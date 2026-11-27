"""Measure explicit baseline and candidate Triton configs without changing production code."""
import time

import torch

from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg


def make(V):
    B, T, H, HV, K, BT = 2, 8192, 16, 16, 128, 64
    dtype = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=dtype)
    do = torch.randn_like(v)
    h = torch.randn(B, T // BT, HV, K, V, device='cuda', dtype=dtype)
    dh = torch.randn_like(h)
    g = torch.randn(B, T, HV, device='cuda', dtype=torch.float32)
    dq, dk = torch.empty_like(q), torch.empty_like(k)
    dw, dv = torch.empty(B, T, HV, K, device='cuda', dtype=dtype), torch.empty_like(v)
    dg = torch.empty(1, B, T, HV, device='cuda', dtype=torch.float32)
    grid = (T // BT, B * HV)
    args = dict(q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh,
                dq=dq, dk=dk, dw=dw, dv=dv, dg=dg, cu_seqlens=None,
                chunk_indices=None, scale=K ** -0.5, B=B, T=T, H=H, HV=HV,
                K=K, V=V, BT=BT, USE_G=True, USE_G_GAMMA=False,
                USE_DW=True, STATE_V_FIRST=False, IS_VARLEN=False)
    return chunk_bwd_kernel_dqkwg.fn.fn, grid, args


def run(V, name, BV, stages):
    kernel, grid, args = make(V)
    start = time.perf_counter()
    binary = kernel[grid](**dict(args, BK=128, BV=BV), num_warps=8, num_stages=stages)
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - start) * 1000
    start_event, end_event = torch.cuda.Event(True), torch.cuda.Event(True)
    start_event.record()
    for _ in range(24):
        kernel[grid](**dict(args, BK=128, BV=BV), num_warps=8, num_stages=stages)
    end_event.record()
    end_event.synchronize()
    warm_ms = start_event.elapsed_time(end_event) / 24
    print(name, 'V', V, 'BK', 128, 'BV', BV, 'warps', 8, 'stages', stages,
          'cold_ms', round(cold_ms, 3), 'warm_ms', round(warm_ms, 6),
          'regs', binary.n_regs, 'spills', binary.n_spills,
          'shared', binary.metadata.shared, flush=True)


if __name__ == '__main__':
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(), flush=True)
    for V in (128, 256):
        run(V, 'baseline', 128, 2)
        run(V, 'candidate', 64, 3)
