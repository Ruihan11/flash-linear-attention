import importlib
import statistics
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import device

chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
sync = torch.npu.synchronize if device == 'npu' else torch.cuda.synchronize
mem = torch.npu if device == 'npu' else torch.cuda
calls = {'wu': 0, 'h': 0}


def count(name, key):
    orig = getattr(chunk, name)

    def wrapped(*args, **kwargs):
        calls[key] += 1
        return orig(*args, **kwargs)
    setattr(chunk, name, wrapped)


count('recompute_w_u_fwd', 'wu')
count('chunk_gated_delta_rule_fwd_h', 'h')


def timed(fn, n=12):
    sync()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - start) * 1000 / n


def run(name, B, T, H, HV, K, V, dtype, cu=None, layers=(1, 4)):
    torch.manual_seed(42)

    def rand(h, d):
        return torch.randn(B, T, h, d, dtype=dtype).to(device).requires_grad_()
    q, k, v = rand(H, K), rand(H, K), rand(HV, V)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV)).to(device).requires_grad_()
    beta = torch.rand(B, T, HV).to(device).requires_grad_()
    dy = torch.randn_like(v)
    cu = torch.tensor(cu, device=device) if cu else None
    inputs = (q, k, v, g, beta)

    def step(flag, depth=1):
        x = v
        for _ in range(depth):
            x, _ = chunk_gated_delta_rule(q, k, x, g, beta, use_qk_l2norm_in_kernel=True,
                                          cu_seqlens=cu, disable_recompute=flag)
        return torch.autograd.grad(x, inputs, dy)

    grads = {}
    for flag in (False, True):
        calls.update(wu=0, h=0)
        grads[flag] = step(flag)
        print('calls', name, 'saved' if flag else 'recompute', dict(calls), flush=True)
    again = step(False)
    for label, other in (('saved_vs_recompute', grads[True]), ('recompute_twice', again)):
        equal = [torch.equal(a, b) for a, b in zip(grads[False], other)]
        err = [((a.float() - b.float()).norm() / a.float().norm()).item() for a, b in zip(grads[False], other)]
        print(label, name, equal, ['%.2e' % e for e in err], flush=True)

    for _ in range(16):
        step(False)
        step(True)
    pairs = []
    for i in range(8):
        pair = {f: timed(lambda f=f: step(f)) for f in ((False, True) if i % 2 == 0 else (True, False))}
        pairs.append((pair[False], pair[True]))
    med = statistics.median
    print('fwdbwd recompute/saved', name, *[round(med(x), 4) for x in zip(*pairs)], 'paired_pct',
          round(med(100 * (1 - b / a) for a, b in pairs), 2), 'wins', sum(b < a for a, b in pairs), flush=True)

    if V == K:      # stacking feeds the output back in as v, which needs matching head shapes
        for depth in layers:
            peak = {}
            for flag in (False, True):
                sync()
                mem.reset_peak_memory_stats()
                step(flag, depth)
                sync()
                peak[flag] = mem.max_memory_allocated() / 2**20
            print('peak_mib', name, 'layers', depth, 'recompute', round(peak[False], 1),
                  'saved', round(peak[True], 1), 'delta', round(peak[True] - peak[False], 1), flush=True)


print(torch.__version__, device, flush=True)
bf, fp = torch.bfloat16, torch.float16
run('small-D64', 4, 2048, 4, 4, 64, 64, bf)
run('dense-D128', 2, 8192, 16, 16, 128, 128, bf)
run('packed-D128', 1, 16384, 16, 16, 128, 128, bf, [0, 4095, 8193, 12289, 16384])
run('gva-H8-HV16', 2, 8192, 8, 16, 128, 128, bf)
run('fp16-D128', 2, 8192, 16, 16, 128, 128, fp)
run('dense-V256', 2, 8192, 16, 16, 128, 256, bf)
