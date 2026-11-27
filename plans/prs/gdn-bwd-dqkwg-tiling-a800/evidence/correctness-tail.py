import torch

from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg
from tests.ops.test_gdn_kernels import chunk_bwd_dqkwg_ref


def check(dtype, use_g, use_w, H, HV, V):
    B, T, K, BT = 1, 128, 64, 64
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=dtype)
    do = torch.randn_like(v)
    h = torch.randn(B, T // BT, HV, K, V, device='cuda', dtype=dtype)
    dh = torch.randn_like(h)
    w = torch.randn(B, T, HV, K, device='cuda', dtype=dtype) if use_w else None
    dv = torch.randn_like(v) if use_w else None
    g = torch.randn(B, T, HV, device='cuda', dtype=torch.float32) if use_g else None
    ref = chunk_bwd_dqkwg_ref(q, k, v, do, h, dh, w, dv, g, K ** -0.5, BT)
    dq = torch.empty(B, T, HV, K, device='cuda', dtype=dtype)
    dk = torch.empty_like(dq)
    dw = torch.empty_like(w) if w is not None else None
    dg = torch.empty(1, B, T, HV, device='cuda', dtype=torch.float32) if g is not None else None
    grid = (T // BT, B * HV)
    chunk_bwd_kernel_dqkwg.fn.fn[grid](
        q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh,
        dq=dq, dk=dk, dw=dw, dv=dv, dg=dg, cu_seqlens=None,
        chunk_indices=None, scale=K ** -0.5, B=B, T=T, H=H, HV=HV,
        K=K, V=V, BT=BT, BK=128, BV=64, USE_G=use_g,
        USE_G_GAMMA=False, USE_DW=use_w, STATE_V_FIRST=False,
        IS_VARLEN=False, num_warps=8, num_stages=3,
    )
    torch.cuda.synchronize()
    if H != HV:
        dq = dq.view(B, T, H, HV // H, K).sum(3)
        dk = dk.view(B, T, H, HV // H, K).sum(3)
    actual = (dq, dk, dw, dg.sum(0) if dg is not None else None)
    errors = []
    for expected, observed in zip(ref, actual):
        if expected is None:
            continue
        errors.append(((expected.float() - observed.float()).norm() / expected.float().norm()).item())
    assert max(errors) < 0.006, (dtype, use_g, use_w, H, HV, V, errors)
    print(dtype, 'use_g', use_g, 'use_w', use_w, 'H/HV', H, HV, 'V', V,
          'rrms', [round(error, 8) for error in errors], flush=True)


if __name__ == '__main__':
    for dtype in (torch.bfloat16, torch.float16):
        for use_g in (False, True):
            for use_w in (False, True):
                check(dtype, use_g, use_w, 2, 4, 70)
