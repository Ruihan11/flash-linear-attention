import statistics

import torch
from triton.runtime.errors import OutOfResources

from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg


def event_ms(fn, repeats=24):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def run(V, BK, BV, warps, stages):
    B, T, H, HV, K, BT = 2, 8192, 16, 16, 128, 64
    d = torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device='cuda', dtype=d)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=d)
    do = torch.randn_like(v)
    h = torch.randn(B, T//BT, HV, K, V, device='cuda', dtype=d)
    dh = torch.randn_like(h)
    w = torch.randn(B, T, HV, K, device='cuda', dtype=d)
    dv = torch.randn_like(v)
    g = torch.randn(B, T, HV, device='cuda', dtype=torch.float32)
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)
    dg = torch.empty(1, B, T, HV, device='cuda', dtype=torch.float32)
    grid = ((K+BK-1)//BK*(T+BT-1)//BT, B*HV)

    def fn():
        return chunk_bwd_kernel_dqkwg.fn.fn[grid](q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh, dq=dq, dk=dk, dw=dw, dv=dv, dg=dg, cu_seqlens=None, chunk_indices=None, scale=K**-0.5, B=B, T=T, H=H, HV=HV, K=K, V=V, BT=BT, BK=BK, BV=BV, USE_G=True, USE_G_GAMMA=False, USE_DW=True, STATE_V_FIRST=False, IS_VARLEN=False, num_warps=warps, num_stages=stages)
    try:
        binary = fn()
        torch.cuda.synchronize()
        xs = [event_ms(fn) for _ in range(4)]
        return statistics.median(xs), binary.n_regs, binary.n_spills
    except torch.OutOfMemoryError:
        raise
    except OutOfResources as e:
        return type(e).__name__, str(e), None


print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
for V in (128, 256):
    print('V', V)
    for BV in (32, 64, 128):
        for warps in (2, 4, 8):
            for stages in (2, 3, 4):
                print(BV, warps, stages, run(V, 128, BV, warps, stages), flush=True)
