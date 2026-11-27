import json
import sys
from pathlib import Path

import torch
import triton

from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd, recompute_w_u_fwd_kernel
from fla.utils import assert_close
from tests.ops.test_gdn_kernels import _make_wy_inverse, recompute_w_u_fwd_ref


def measure(fn, n=10):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    torch.cuda.synchronize()
    start.record()
    for _ in range(n):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / n


def run(mode, V):
    torch.manual_seed(42)
    B, T, H, K = 2, 8192, 16, 128

    def rand(*shape):
        return torch.randn(*shape, device='cuda', dtype=torch.bfloat16)
    q, k = [rand(B, T, H, K).requires_grad_() for _ in range(2)]
    v = rand(B, T, H, V).requires_grad_()
    dy = torch.randn_like(v)
    g = torch.nn.functional.logsigmoid(rand(B, T, H).float()).requires_grad_()
    beta = rand(B, T, H).sigmoid().requires_grad_()

    def forward():
        return chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)[0]

    def train():
        return torch.autograd.grad(forward(), (q, k, v, g, beta), dy)
    if mode in ('baseline', 'tiles', 'counters'):
        A = _make_wy_inverse(B, T, H, 64, v.dtype)
        wg = torch.randn_like(g)*0.1
        with torch.no_grad():
            ref = recompute_w_u_fwd_ref(k, v, beta, A, wg)
            w, u = recompute_w_u_fwd(k, v, beta, A, wg)

            def check():
                for name, target, value in zip(('w', 'u'), ref, (w, u)):
                    assert_close(name, target, value, 0.005)
                    print('rms', V, name, ((target.float()-value.float()).norm()/target.float().norm()).item())
            check()
            if mode in ('tiles', 'counters'):
                def launch(tile):
                    return recompute_w_u_fwd_kernel[(T//64, B*H)](k=k, v=v, beta=beta, w=w, u=u, A=A, g=wg, cu_seqlens=None, chunk_indices=None, T=T, H=H, HV=H, K=K, V=V, BT=64, BK=64, BV=tile)
                tiles = [32, 64, 128]
                for tile in tiles:
                    binary = launch(tile)
                    check()
                    print('resource', V, tile, binary.n_regs, binary.n_spills,
                          getattr(recompute_w_u_fwd_kernel.fn, 'best_config', None))
                if mode == 'counters':
                    for tile in tiles:
                        with torch.cuda.nvtx.range(f'BV{tile}'):
                            launch(tile)
                    return
                for i in range(6):
                    for tile in tiles[i % 3:]+tiles[:i % 3]:
                        print('tile_ms', V, i, tile, measure(lambda: launch(tile), 30))
                return
    for _ in range(3):
        train()
    torch.cuda.synchronize()
    if mode == 'trace':
        for i in range(7):
            print('steady_ms', V, i, measure(forward), measure(train))
        path = Path('profile/gdn-wy-vtile')
        path.mkdir(parents=True, exist_ok=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            train()
            torch.cuda.synchronize()
        file = path/f'v{V}.json'
        prof.export_chrome_trace(str(file))
        kernels = [e for e in json.loads(file.read_text())['traceEvents'] if e.get('cat') == 'kernel']
        wy = [e for e in kernels if 'recompute_w_u_fwd_kernel' in e['name']]
        assert len(wy) == 2, [e['name'] for e in kernels]
        print('trace', V, len(wy), [e['dur'] for e in wy], sum(e['dur']
              for e in kernels), getattr(recompute_w_u_fwd_kernel.fn, 'best_config', None))
    else:
        for i in range(2):
            print('timing_ms', V, i, measure(forward), measure(train))


if __name__ == '__main__':
    print(torch.__version__, triton.__version__, torch.cuda.get_device_name())
    for width in [128, 256]:
        run(sys.argv[1], width)
