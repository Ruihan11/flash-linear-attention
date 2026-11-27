"""Default path (disable_recompute unset): dump outputs/grads and steady timings. Usage: default_path.py <out.pt>"""
import statistics
import sys

import torch

import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def timed(fn, count=12):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


print(fla.__file__, flush=True)
out = {}
for name, B, T, H, HV, K, V, dtype, cu in [
    ('dense-V256', 2, 8192, 16, 16, 128, 256, torch.bfloat16, None),
    ('dense-V128', 2, 8192, 16, 16, 128, 128, torch.bfloat16, None),
    ('packed-gva-fp16', 1, 4096, 2, 4, 64, 64, torch.float16, [0, 15, 1000, 2500, 4096]),
]:
    torch.manual_seed(42)

    def rand(h, d):
        return torch.randn(B, T, h, d, device='cuda', dtype=dtype).requires_grad_()
    q, k, v = rand(H, K), rand(H, K), rand(HV, V)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, device='cuda')).requires_grad_()
    beta = torch.rand_like(g, dtype=dtype).requires_grad_()
    dy = torch.randn_like(v)
    cu = torch.tensor(cu, device='cuda') if cu else None

    def fwd():
        return chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)[0]

    def step():
        return torch.autograd.grad(fwd(), (q, k, v, g, beta), dy)
    o = fwd()
    out[name] = [x.detach().cpu() for x in (o, *torch.autograd.grad(o, (q, k, v, g, beta), dy))]
    for _ in range(16):
        step()
    with torch.no_grad():
        f = statistics.median(timed(fwd) for _ in range(8))
    fb = statistics.median(timed(step) for _ in range(8))
    print('default_ms', name, 'fwd', round(f, 5), 'fwdbwd', round(fb, 5), flush=True)
torch.save(out, sys.argv[1])
