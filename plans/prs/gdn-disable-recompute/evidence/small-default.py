"""Default-path fwdbwd on the smallest registry shape. Run once per tree with PYTHONPATH=<tree>."""
import statistics

import torch

import fla
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.manual_seed(42)
B, T, H, D = 8, 1024, 8, 64
dt = torch.bfloat16
q, k, v = [torch.randn(B, T, H, D, device='cuda', dtype=dt).requires_grad_() for _ in range(3)]
g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device='cuda')).requires_grad_()
beta = torch.rand(B, T, H, device='cuda').sigmoid().requires_grad_()
dy = torch.randn_like(v)


def step():
    o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
    return torch.autograd.grad(o, (q, k, v, g, beta), dy)


def timed(n=50):
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        step()
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / n


for _ in range(200):
    step()
xs = [timed() for _ in range(20)]
print(fla.__file__.split('/fla/')[0], 'fwdbwd_ms median', round(statistics.median(xs), 5),
      'min', round(min(xs), 5), 'max', round(max(xs), 5), flush=True)
