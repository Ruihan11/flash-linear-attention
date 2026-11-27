#!/usr/bin/env python
import statistics

import torch

from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg
from tests.ops.test_gdn_kernels import chunk_bwd_dqkwg_ref


def event_ms(fn, repeats=30):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def summary(xs):
    q1, q3 = statistics.quantiles(xs, n=4, method='inclusive')[0], statistics.quantiles(xs, n=4, method='inclusive')[2]
    return f'{[round(x, 6) for x in xs]} median={statistics.median(xs):.6f} iqr={q3-q1:.6f}'


def make(V):
    B, T, H, HV, K, BT = 2, 8192, 16, 16, 128, 64
    d = torch.bfloat16
    q = torch.randn(B, T, H, K, device='cuda', dtype=d)
    k = torch.randn_like(q)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=d)
    do = torch.randn_like(v)
    h = torch.randn(B, T//BT, HV, K, V, device='cuda', dtype=d)
    dh = torch.randn_like(h)
    w = torch.randn(B, T, HV, K, device='cuda', dtype=d)
    dv = torch.randn_like(v)
    g = torch.randn(B, T, HV, device='cuda', dtype=torch.float32)
    return q, k, v, do, h, dh, w, dv, g, K**-0.5, BT


def launch(args, BK, BV, collect=True):
    q, k, v, do, h, dh, w, dv, g, scale, BT = args
    B, T, H, K = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    HV = v.shape[2]
    NK = (K+BK-1)//BK
    NT = (T+BT-1)//BT
    dq = torch.empty_like(q).reshape(B, T, HV, K)
    dk = torch.empty_like(k).reshape(B, T, HV, K)
    dw = torch.empty_like(w)
    dg = torch.empty(NK, *g.shape, dtype=torch.float32, device='cuda')
    grid = (NK*NT, B*HV)

    def run():
        chunk_bwd_kernel_dqkwg[grid](q=q, k=k, v=v, g=g, g_gamma=None, h=h, do=do, dh=dh, dq=dq, dk=dk, dw=dw, dv=dv, dg=dg, cu_seqlens=None, chunk_indices=None,
                                     scale=scale, B=B, T=T, H=H, HV=HV, K=K, V=v.shape[-1], BT=BT, BK=BK, BV=BV, USE_G=True, USE_G_GAMMA=False, USE_DW=True, STATE_V_FIRST=False)
        return dq, dk, dw, dg.sum(0)
    run()
    torch.cuda.synchronize()
    return run, (dq, dk, dw, dg.sum(0))


def main():
    torch.manual_seed(42)
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
    candidates = [(128, 32), (128, 64), (128, 128), (64, 64), (64, 128)]
    for V in (128, 256):
        args = make(V)
        q, k, v, do, h, dh, w, dv, g, scale, BT = args
        ref = chunk_bwd_dqkwg_ref(q, k, v, do, h, dh, w, dv, g, scale, BT)
        for BK, BV in candidates:
            fn, out = launch(args, BK, BV)
            ratios = []
            for expected, actual in zip(ref, out):
                ratios.append(((expected.float()-actual.float()).norm()/expected.float().norm()).item())
            samples = [event_ms(fn) for _ in range(6)]
            print(f'V={V} BK={BK} BV={BV} ratios={[round(x, 8) for x in ratios]} {summary(samples)}', flush=True)


if __name__ == '__main__':
    main()
