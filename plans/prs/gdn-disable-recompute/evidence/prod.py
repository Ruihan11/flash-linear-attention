"""Paired fwdbwd through the public disable_recompute parameter. Run with PYTHONPATH=. on one idle A800."""
import statistics

import torch

from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def timed(fn, count=12):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def run(name, B, T, H, HV, K, V, dtype, cu=None):
    torch.manual_seed(42)

    def rand(h, d):
        return torch.randn(B, T, h, d, device='cuda', dtype=dtype).requires_grad_()
    q, k, v = rand(H, K), rand(H, K), rand(HV, V)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV, device='cuda')).requires_grad_()
    beta = torch.rand_like(g, dtype=dtype).requires_grad_()
    dy = torch.randn_like(v)
    cu = torch.tensor(cu, device='cuda') if cu else None

    def step(flag):
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, disable_recompute=flag)
        return torch.autograd.grad(o, (q, k, v, g, beta), dy)
    print('grad_equal', name, [torch.equal(a, b) for a, b in zip(step(False), step(True))], flush=True)
    for _ in range(16):
        step(False)
        step(True)
    pairs = []
    for i in range(8):
        pair = {r: timed(lambda r=r: step(r)) for r in ((False, True) if i % 2 == 0 else (True, False))}
        pairs.append((pair[False], pair[True]))
    med = statistics.median
    print('fwdbwd recompute/saved', name, *[round(med(x), 5) for x in zip(*pairs)], 'paired_pct',
          round(med(100 * (1 - b / a) for a, b in pairs), 3), 'wins', sum(b < a for a, b in pairs), flush=True)


print(torch.__version__, torch.cuda.get_device_name(), flush=True)
bf, fp = torch.bfloat16, torch.float16
run('dense-V256', 2, 8192, 16, 16, 128, 256, bf)
run('dense-V128', 2, 8192, 16, 16, 128, 128, bf)
run('packed-V256', 1, 16384, 16, 16, 128, 256, bf, [0, 4095, 8193, 12289, 16384])
run('gva-H8-HV16-V256', 2, 8192, 8, 16, 128, 256, bf)
run('small-D64', 4, 2048, 4, 4, 64, 64, bf)
run('fp16-V256', 2, 8192, 16, 16, 128, 256, fp)
