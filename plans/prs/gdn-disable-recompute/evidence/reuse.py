"""Stage 0: reuse forward w/u/h/v_new in the GDN chunk backward without touching production files.

Run from the repository root with PYTHONPATH=. and one visible A800.
Pass --timing to add the paired fwdbwd comparison (needs an idle GPU).
"""
import importlib
import statistics
import sys

import torch

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
orig_intra = chunk.chunk_gated_delta_rule_fwd_intra
orig_wu = chunk.recompute_w_u_fwd
orig_h = chunk.chunk_gated_delta_rule_fwd_h

state = {'reuse': False, 'wu': None, 'h': None, 'calls': {'wu': 0, 'h': 0}}


def intra(**kw):
    w, u, A = orig_intra(**kw)
    state['wu'] = (w, u)
    return w, u, A


def wu(**kw):
    if state['reuse']:
        return state['wu']
    state['calls']['wu'] += 1
    return orig_wu(**kw)


def fwd_h(**kw):
    # the backward call is the one that does not ask for the final state
    if state['reuse'] and state['h'] is not None and not kw['output_final_state']:
        h, v_new = state['h']
        state['h'] = None
        return h, v_new, None
    state['calls']['h'] += 1
    h, v_new, final = orig_h(**kw)
    state['h'] = (h, v_new)
    return h, v_new, final


chunk.chunk_gated_delta_rule_fwd_intra = intra
chunk.recompute_w_u_fwd = wu
chunk.chunk_gated_delta_rule_fwd_h = fwd_h


def timed(fn, count):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def run(V, timing):
    torch.manual_seed(42)
    B, T, H, K = 2, 8192, 16, 128

    def rand(d):
        return torch.randn(B, T, H, d, device='cuda', dtype=torch.bfloat16).requires_grad_()
    q, k, v = rand(K), rand(K), rand(V)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device='cuda')).requires_grad_()
    beta = torch.rand_like(g, dtype=q.dtype).requires_grad_()
    dy = torch.randn_like(v)

    def step(reuse):
        state['reuse'] = reuse
        state['h'] = None
        o, _ = chunk_gated_delta_rule(q, k, v, g, beta, use_qk_l2norm_in_kernel=True)
        return torch.autograd.grad(o, (q, k, v, g, beta), dy)

    grads = {}
    for reuse in (False, True):
        state['calls'] = {'wu': 0, 'h': 0}
        grads[reuse] = step(reuse)
        print('calls', V, 'reuse' if reuse else 'base', state['calls'], flush=True)
    print('grad_equal', V, [torch.equal(a, b) for a, b in zip(grads[False], grads[True])], flush=True)
    for reuse in (False, True):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        step(reuse)
        torch.cuda.synchronize()
        print('peak_mib', V, 'reuse' if reuse else 'base', round(torch.cuda.max_memory_allocated() / 2**20, 1), flush=True)
    if not timing:
        return
    for _ in range(16):
        step(False)
        step(True)
    pairs = []
    for i in range(8):
        pair = {r: timed(lambda r=r: step(r), 12) for r in ((False, True) if i % 2 == 0 else (True, False))}
        pairs.append((pair[False], pair[True]))
        print('pair', V, i, *pairs[-1], flush=True)
    med = statistics.median
    print('fwdbwd base/reuse', V, *[round(med(x), 5) for x in zip(*pairs)], 'paired_pct',
          round(med(100 * (1 - b / a) for a, b in pairs), 3), 'wins', sum(b < a for a, b in pairs), flush=True)


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.get_device_name(), flush=True)
    for V in (256, 128):
        run(V, '--timing' in sys.argv)
