"""Run from the repository root with PYTHONPATH=. and one visible A800."""
import importlib
import inspect
import statistics

import torch

from fla.ops.common import chunk_o
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
original = chunk.chunk_bwd_dqkwg
source = inspect.getsource(inspect.unwrap(original))
source = source[source.index('def chunk_bwd_dqkwg('):]
line = 'BV = min(max(triton.next_power_of_2(V), 16), CONST_TILING)'
assert source.count(line) == 1
variants = {}
for name, replacement in [('base', line), ('bv32', 'BV = 32')]:
    scope = dict(vars(chunk_o))
    exec(source.replace(line, replacement), scope)
    variants[name] = scope['chunk_bwd_dqkwg']


def run(width):
    torch.manual_seed(42)

    def rand(d):
        return torch.randn(2, 8192, 16, d, device='cuda', dtype=torch.bfloat16).requires_grad_()
    q, k, v = rand(128), rand(128), rand(width)
    g = torch.nn.functional.logsigmoid(torch.randn(*q.shape[:3], device='cuda')).requires_grad_()
    beta = torch.rand_like(g, dtype=q.dtype).requires_grad_()
    dy = torch.randn_like(v)

    def step():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        return torch.autograd.grad(o, (q, k, v, g, beta), dy)
    gradients = {}
    for name, fn in variants.items():
        chunk.chunk_bwd_dqkwg = fn
        seen = []

        class Probe:
            def __getitem__(self, grid):
                def launch(**kw):
                    seen.append((kw['BK'], kw['BV'], grid))
                    return chunk_o.chunk_bwd_kernel_dqkwg[grid](**kw)
                return launch
        fn.__globals__['chunk_bwd_kernel_dqkwg'] = Probe()
        gradients[name] = step()
        fn.__globals__['chunk_bwd_kernel_dqkwg'] = chunk_o.chunk_bwd_kernel_dqkwg
        assert len(seen) == 1
        print('actual_tiles', width, name, seen)
    for a, b in zip(*gradients.values()):
        err = ((a.float()-b.float()).norm()/a.float().norm()).item()
        print('gradient_rrms', width, err)
        assert err < 0.006
    for _ in range(20):
        step()
    samples = []
    for i in range(12):
        pair = {}
        for name in (('base', 'bv32') if i % 2 == 0 else ('bv32', 'base')):
            chunk.chunk_bwd_dqkwg = variants[name]
            start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            start.record()
            for _ in range(12):
                step()
            end.record()
            end.synchronize()
            pair[name] = start.elapsed_time(end)/12
        delta = 100*(1-pair['bv32']/pair['base'])
        samples.append((pair['base'], pair['bv32'], delta))
        print('pair', width, i, *samples[-1], flush=True)
    print('median_base_ms_candidate_ms_paired_pct', width, *map(statistics.median, zip(*samples)))
    chunk.chunk_bwd_dqkwg = original


for width in (128, 256):
    run(width)
