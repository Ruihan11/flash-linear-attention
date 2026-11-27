"""Load the upstream module's kernel with one optional extra config in a temporary module."""
import importlib.util
import sys
import tempfile
from pathlib import Path

from fla.ops.common import chunk_o


def candidate():
    source = Path(chunk_o.__file__).read_text()
    start = source.index('@triton.heuristics', source.index('def chunk_fwd_kernel_o'))
    end = source.index('@triton.heuristics', source.index('def chunk_bwd_kernel_dqkwg'))
    code = source[:source.index('@triton.heuristics')] + source[start:end]
    code = code.replace("    ],\n    key=", "    ] + [triton.Config({'NARROW': True}, num_warps=8, num_stages=3)],\n    key=")
    code = code.replace('IS_VARLEN: tl.constexpr,\n):',
                        'IS_VARLEN: tl.constexpr,\n    NARROW: tl.constexpr = False,\n):\n    TILE_V: tl.constexpr = 64 if NARROW else BV')
    code = code.replace('for i_v in range(tl.cdiv(V, BV)):', 'for i_v in range(tl.cdiv(V, TILE_V)):').replace(
        'i_v * BV + tl.arange(0, BV)', 'i_v * TILE_V + tl.arange(0, TILE_V)')
    path = Path(tempfile.mkdtemp()) / 'vtile_probe.py'
    path.write_text(code)
    spec = importlib.util.spec_from_file_location('vtile_probe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.chunk_bwd_kernel_dqkwg


if __name__ == '__main__':
    import runpy
    p = runpy.run_path(str(Path(__file__).with_name('paired-shapes.py')))
    new = candidate()
    original = p['kernel']

    def run_new(width, heads, packed):
        import torch

        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        torch.manual_seed(42)
        b, t = (1, 16384) if packed else (2, 8192)
        cu = torch.tensor([0, 4095, 8193, 12289, t], device='cuda') if packed else None
        def rand(h, d): return torch.randn(b, t, h, d, device='cuda', dtype=torch.bfloat16).requires_grad_()
        q, k, v = rand(heads, 128), rand(heads, 128), rand(16, width)
        g = torch.nn.functional.logsigmoid(torch.randn(b, t, 16, device='cuda')).requires_grad_()
        beta = torch.rand_like(g, dtype=q.dtype).requires_grad_()
        dy = torch.randn_like(v)

        def step(kernel):
            chunk_o.chunk_bwd_kernel_dqkwg = kernel
            o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
            return torch.autograd.grad(o, (q, k, v, g, beta), dy)
        a, b = step(original), step(new)
        err = [((x.float()-y.float()).norm()/x.float().norm()).item() for x, y in zip(a, b)]
        print(width, heads, packed, 'selected', new.fn.best_config, 'gradient_rrms', err, flush=True)
        assert all(x < 0.006 for x in err)
        p['compare']('autoselect', [lambda:step(original), lambda:step(new)], 12)
    for args in [(256, 16, False), (128, 16, False), (256, 16, True), (256, 8, False)]:
        run_new(*args)
