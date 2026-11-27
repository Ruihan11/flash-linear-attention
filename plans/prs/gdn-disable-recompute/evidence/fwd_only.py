"""Forward-only timing under no_grad for dense V=256, with the autotune choice of each forward kernel."""
import statistics
import sys

import torch

import fla
from fla.ops.common import chunk_delta_h, chunk_o
from fla.ops.gated_delta_rule import chunk_fwd, chunk_gated_delta_rule, wy_fast

order = sys.argv[1] if len(sys.argv) > 1 else 'fwd-first'


def timed(fn, count=12):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def best(k):
    while not hasattr(k, 'best_config'):
        k = k.fn
    return str(getattr(k, 'best_config', None))


torch.manual_seed(42)
B, T, H, K, V = 2, 8192, 16, 128, 256


def rand(d):
    return torch.randn(B, T, H, d, device='cuda', dtype=torch.bfloat16).requires_grad_()


q, k, v = rand(K), rand(K), rand(V)
g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device='cuda')).requires_grad_()
beta = torch.rand_like(g, dtype=q.dtype).requires_grad_()
dy = torch.randn_like(v)


def fwd():
    return chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]


def step():
    return torch.autograd.grad(fwd(), (q, k, v, g, beta), dy)


if order == 'bwd-first':
    for _ in range(16):
        step()
with torch.no_grad():
    for _ in range(16):
        fwd()
    xs = [timed(fwd) for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 8)]
if len(sys.argv) > 2:
    print('samples', [round(x, 3) for x in xs])
print(fla.__file__.split('/fla/')[0], order, 'fwd_ms', round(statistics.median(xs), 5),
      'min', round(min(xs), 5), 'max', round(max(xs), 5))
for name, kern in [('fwd_h', chunk_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64), ('fwd_o', chunk_o.chunk_fwd_kernel_o),
                   ('wu', wy_fast.recompute_w_u_fwd_kernel), ('kkt', chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel)]:
    print('  ', name, best(kern))
