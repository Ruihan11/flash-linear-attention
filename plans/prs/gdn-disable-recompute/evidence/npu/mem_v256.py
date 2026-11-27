# Per-layer retained memory of disable_recompute on the committed code, K != V.
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import device

sync = torch.npu.synchronize if device == 'npu' else torch.cuda.synchronize
mem = torch.npu if device == 'npu' else torch.cuda


def run(name, B, T, H, HV, K, V, dtype=torch.bfloat16, layers=(1, 4, 12)):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=dtype).to(device).requires_grad_()
    k = torch.randn(B, T, H, K, dtype=dtype).to(device).requires_grad_()
    v = torch.randn(B, T, HV, V, dtype=dtype).to(device).requires_grad_()
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV)).to(device).requires_grad_()
    beta = torch.rand(B, T, HV).to(device).requires_grad_()
    dy = torch.randn_like(v)

    def step(flag, depth):
        x = v
        for _ in range(depth):
            x, _ = chunk_gated_delta_rule(q, k, x, g, beta, use_qk_l2norm_in_kernel=True, disable_recompute=flag)
        return torch.autograd.grad(x, (q, k, v, g, beta), dy)

    step(False, 1), step(True, 1)
    delta = {}
    for depth in layers:
        peak = {}
        for flag in (False, True):
            sync()
            mem.reset_peak_memory_stats()
            step(flag, depth)
            sync()
            peak[flag] = mem.max_memory_allocated() / 2**20
        delta[depth] = peak[True] - peak[False]
        print('peak_mib', name, 'layers', depth, 'recompute', round(peak[False], 1), 'saved', round(peak[True], 1),
              'delta', round(delta[depth], 1), flush=True)
    a, b = layers[0], layers[-1]
    print('per_layer_mib', name, round((delta[b] - delta[a]) / (b - a), 1), flush=True)


print(torch.__version__, device, flush=True)
run('dense-V256', 2, 8192, 16, 16, 128, 256)
run('dense-D128', 2, 8192, 16, 16, 128, 128)
