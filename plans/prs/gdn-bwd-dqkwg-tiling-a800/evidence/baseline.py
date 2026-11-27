#!/usr/bin/env python
import statistics

import torch

from fla.ops.common.chunk_o import chunk_bwd_dqkwg
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def event_ms(fn, repeats=30):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def workload(V):
    B, T, H, HV, K, BT = 2, 8192, 16, 16, 128, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, K, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=dtype, requires_grad=True)
    k.requires_grad_()
    do = torch.randn_like(v)
    h = torch.randn(B, T//BT, HV, K, V, device='cuda', dtype=dtype)
    dh = torch.randn_like(h)
    w = torch.randn(B, T, HV, K, device='cuda', dtype=dtype)
    dv = torch.randn_like(v)
    g = torch.randn(B, T, HV, device='cuda', dtype=torch.float32, requires_grad=True)
    beta = torch.rand(B, T, HV, device='cuda', dtype=dtype).sigmoid().requires_grad_()

    def run():
        return chunk_bwd_dqkwg(q, k, v, do, h, dh, w, g=g, dv=dv, scale=K ** -0.5, chunk_size=BT)

    def full_forward():
        return chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]

    def full_train():
        return torch.autograd.grad(full_forward(), (q, k, v, g, beta), do, retain_graph=True)
    for _ in range(10):
        run()
        full_forward()
        full_train()
    torch.cuda.synchronize()
    kernel_samples = [event_ms(run) for _ in range(7)]
    full_fwd_samples = [event_ms(full_forward, 10) for _ in range(7)]
    full_train_samples = [event_ms(full_train, 5) for _ in range(7)]

    def summary(samples):
        q1, q3 = statistics.quantiles(samples, n=4, method='inclusive')[
            0], statistics.quantiles(samples, n=4, method='inclusive')[2]
        return f'samples_ms={[round(x, 6) for x in samples]} median_ms={statistics.median(samples):.6f} iqr_ms={q3-q1:.6f}'
    print(f'V={V} kernel {summary(kernel_samples)}', flush=True)
    print(f'V={V} full_fwd {summary(full_fwd_samples)}', flush=True)
    print(f'V={V} full_fwdbwd {summary(full_train_samples)}', flush=True)


print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
for V in (128, 256):
    workload(V)
