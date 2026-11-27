"""Intercept the kernel launch, not operator dispatch. Replay with the same tensors."""
import statistics
import sys

import torch
from triton.runtime.errors import OutOfResources

from fla.ops.common import chunk_o
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

small = 64 if '--bv64' in sys.argv else 32
kernel = chunk_o.chunk_bwd_kernel_dqkwg


class Probe:
    tile = None
    capture = False

    def __getitem__(self, grid):
        def run(**kw):
            if self.tile is not None:
                kw['BV'] = self.tile
            binary = kernel[grid](**kw)
            if self.capture:
                self.calls.append((grid, kw, binary.n_regs, binary.n_spills, str(kernel.fn.best_config)))
            return binary
        return run


def timed(fn, count):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def compare(scope, fns, count):
    pairs = []
    for i in range(8):
        order = (0, 1) if i % 2 == 0 else (1, 0)
        pair = {j: timed(fns[j], count) for j in order}
        pairs.append((pair[0], pair[1]))
    med = statistics.median
    print(scope, f'base/bv{small}', *[round(med(xs), 5) for xs in zip(*pairs)], 'paired_pct',
          round(med(100*(1-b/a) for a, b in pairs), 3), 'wins', sum(b < a for a, b in pairs), 'pairs', pairs, flush=True)


def run(width, heads, packed):
    torch.manual_seed(42)
    b, t = (1, 16384) if packed else (2, 8192)

    def rand(h, d):
        return torch.randn(b, t, h, d, device='cuda', dtype=torch.bfloat16).requires_grad_()
    q, k, v = rand(heads, 128), rand(heads, 128), rand(16, width)
    g = torch.nn.functional.logsigmoid(torch.randn(b, t, 16, device='cuda')).requires_grad_()
    beta = torch.rand_like(g, dtype=q.dtype).requires_grad_()
    dy = torch.randn_like(v)
    cu = torch.tensor([0, 4095, 8193, 12289, t], device='cuda', dtype=torch.long) if packed else None

    def step():
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return torch.autograd.grad(o, (q, k, v, g, beta), dy)
    print('shape', width, heads, 16, packed, flush=True)
    probe = Probe()
    chunk_o.chunk_bwd_kernel_dqkwg = probe
    try:
        grads = []
        probe.capture = True
        for tile in (None, small):
            probe.calls = []
            probe.tile = tile
            grads.append(step())
            assert len(probe.calls) == 1
            grid, kw, regs, spills, config = probe.calls.pop()
            print('launch', kw['BK'], kw['BV'], grid, regs, spills, config, flush=True)
        err = [((a.float()-b.float()).norm()/a.float().norm()).item() for a, b in zip(*grads)]
        print('grad_rrms', err, flush=True)
        assert max(err) < 0.006 and all(torch.isfinite(x).all() for x in (*grads[0], *grads[1]))
        probe.capture = False

        def full(tile):
            probe.tile = tile
            step()

        def micro(tile):
            kernel[grid](**dict(kw, BV=tile))
        for _ in range(16):
            full(None)
            full(small)
        compare('fwdbwd', [lambda:full(None), lambda:full(small)], 12)
        if '--stages' in sys.argv:
            jit = kernel.fn.fn
            kwargs = dict(kw, USE_G=True, USE_G_GAMMA=False, USE_DW=True, IS_VARLEN=False)
            for stages in (2, 4):
                def fixed(tile):
                    return jit[grid](**dict(kwargs, BV=tile), num_warps=8, num_stages=stages)
                valid = []
                for tile in (128, 32):
                    try:
                        obj = fixed(tile)
                    except OutOfResources as error:
                        print('infeasible', tile, stages, str(error), flush=True)
                        continue
                    valid.append(tile)
                    print('fixed_resources', tile, stages, obj.n_regs, obj.n_spills, obj.metadata.shared, flush=True)
                if len(valid) == 2:
                    compare(f'fixed_stages_{stages}', [lambda:fixed(128), lambda:fixed(32)], 24)
                elif valid:
                    print('single_ms', valid[0], stages, statistics.median(
                        timed(lambda: fixed(valid[0]), 24) for _ in range(8)))
        compare('kernel', [lambda:micro(128), lambda:micro(small)], 24)
    finally:
        chunk_o.chunk_bwd_kernel_dqkwg = kernel


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.get_device_name(), flush=True)
    cases = [(256, 16, False)] if '--stages' in sys.argv else [(256, 16, False),
                                                               (128, 16, False), (256, 16, True), (256, 8, False)]
    for width, heads, packed in cases:
        run(width, heads, packed)
