"""Stage 1a: reuse forward w/u/h/v_new in the GDN chunk backward without touching production files.

Run from the repository root with PYTHONPATH=. and one visible idle A800:
    stage1.py options   bitwise equality of outputs and gradients across option combinations
    stage1.py memory    peak memory of 1/4/12 stacked operators, default vs reuse
    stage1.py matrix    paired fwdbwd timing on the remaining shapes

Saved tensors live on a LIFO stack from the forward call to the matching backward call,
which mimics save_for_backward for stacked operators. The default path stores nothing.
"""
import importlib
import statistics
import sys

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import chunk_gated_delta_rule

chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
orig_intra = chunk.chunk_gated_delta_rule_fwd_intra
orig_wu = chunk.recompute_w_u_fwd
orig_h = chunk.chunk_gated_delta_rule_fwd_h

state = {'reuse': False, 'phase': 'fwd', 'saved': [], 'calls': {'wu': 0, 'h': 0}}


def intra(**kw):
    state['phase'] = 'fwd'
    w, u, A = orig_intra(**kw)
    if state['reuse']:
        state['saved'].append([w, u])
    return w, u, A


def wu(**kw):
    state['phase'] = 'bwd'
    if state['reuse']:
        return tuple(state['saved'][-1][:2])
    state['calls']['wu'] += 1
    return orig_wu(**kw)


def fwd_h(**kw):
    if state['reuse'] and state['phase'] == 'bwd':
        h, v_new = state['saved'].pop()[2:]
        return h, v_new, None
    state['calls']['h'] += 1
    h, v_new, final = orig_h(**kw)
    if state['reuse']:
        state['saved'][-1] += [h, v_new]
    return h, v_new, final


chunk.chunk_gated_delta_rule_fwd_intra = intra
chunk.recompute_w_u_fwd = wu
chunk.chunk_gated_delta_rule_fwd_h = fwd_h


def make(B=2, T=8192, H=16, HV=16, K=128, V=256, dtype=torch.bfloat16, lens=None,
         h0=False, v_first=False, gate=False, chunk_size=64, layers=1):
    torch.manual_seed(42)
    if lens is not None:
        B, T = 1, sum(lens)

    def rand(h, d):
        return torch.randn(B, T, h, d, device='cuda', dtype=dtype).requires_grad_()
    q, k, v = rand(H, K), rand(H, K), rand(HV, V)
    if gate:
        g = torch.randn(B, T, HV, device='cuda').requires_grad_()
    else:
        g = F.logsigmoid(torch.randn(B, T, HV, device='cuda')).requires_grad_()
    beta = torch.rand(B, T, HV, device='cuda', dtype=dtype).requires_grad_()
    dy = torch.randn_like(v)
    inputs = [q, k, v, g, beta]
    kw = dict(use_qk_l2norm_in_kernel=True, state_v_first=v_first, chunk_size=chunk_size)
    if lens is not None:
        kw['cu_seqlens'] = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), device='cuda', dtype=torch.long)
    dht = None
    if h0:
        N = len(lens) if lens is not None else B
        shape = (N, HV, V, K) if v_first else (N, HV, K, V)
        init = torch.randn(*shape, device='cuda').requires_grad_()
        dht = torch.randn_like(init)
        inputs.append(init)
        kw.update(initial_state=init, output_final_state=True)
    if gate:
        A_log = (0.1 * torch.randn(HV, device='cuda')).requires_grad_()
        dt_bias = (0.1 * torch.randn(HV, device='cuda')).requires_grad_()
        inputs += [A_log, dt_bias]
        kw.update(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias)

    def step(reuse):
        state['reuse'] = reuse
        state['saved'].clear()
        x, ht = v, None
        for _ in range(layers):
            x, ht = chunk_gated_delta_rule(q, k, x, g, beta, **kw)
        outs, seeds = ([x, ht], [dy, dht]) if h0 else ([x], [dy])
        grads = torch.autograd.grad(outs, inputs, seeds)
        assert not state['saved']
        return outs, grads

    return step


def timed(fn, count):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(count):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / count


def options():
    small = dict(B=2, T=1024, H=4, HV=4, K=64, V=64)
    cases = {
        'dense': small,
        'h0-final': dict(small, h0=True),
        'v-first-h0': dict(small, h0=True, v_first=True),
        'gate-in-kernel': dict(small, gate=True),
        'gate-in-kernel-h0': dict(small, gate=True, h0=True),
        'chunk32': dict(small, chunk_size=32),
        'tail-T1000': dict(small, T=1000),
        'packed': dict(small, lens=[300, 417, 200, 107]),
        'packed-h0': dict(small, lens=[300, 417, 200, 107], h0=True),
        'gva-H2-HV4': dict(small, H=2, HV=4),
        'fp16': dict(small, dtype=torch.float16),
        'K128-V256': dict(small, K=128, V=256),
        'combined': dict(small, H=2, HV=4, lens=[300, 417, 200, 107], h0=True, v_first=True, gate=True, chunk_size=32),
    }
    failed = []
    for name, cfg in cases.items():
        step = make(**cfg)
        res = {}
        for reuse in (False, True):
            state['calls'] = {'wu': 0, 'h': 0}
            res[reuse] = step(reuse)
            assert state['calls'] == ({'wu': 0, 'h': 1} if reuse else {'wu': 1, 'h': 2}), state['calls']
        base, new = [[*o, *g] for o, g in (res[False], res[True])]
        equal = [torch.equal(a, b) for a, b in zip(base, new)]
        finite = all(torch.isfinite(x).all().item() for x in base)
        ok = all(equal) and finite
        if not ok:
            failed.append(name)
        print('option', name, 'ok' if ok else 'FAIL', 'outputs+grads',
              len(equal), 'equal', equal, 'finite', finite, flush=True)
    print('options_failed', failed, flush=True)


def memory():
    shapes = {
        'main-V256': dict(B=2, T=8192, H=16, HV=16, K=128, V=256),
        'small-D64': dict(B=4, T=2048, H=4, HV=4, K=64, V=64),
    }
    for name, cfg in shapes.items():
        B, T, HV, K, V = cfg['B'], cfg['T'], cfg['HV'], cfg['K'], cfg['V']
        expected = (B * T * HV * (K + 2 * V) + B * (T // 64) * HV * K * V) * 2 / 2**20
        print('expected_saved_mib_per_layer', name, round(expected, 1), flush=True)
        deltas = {}
        for layers in (1, 4, 12):
            step = make(layers=layers, **cfg)
            peaks, grads = {}, {}
            for reuse in (False, True):
                step(reuse)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                held = torch.cuda.memory_allocated()
                grads[reuse] = step(reuse)[1]
                torch.cuda.synchronize()
                peaks[reuse] = torch.cuda.max_memory_allocated() / 2**20
                print('peak_mib', name, 'layers', layers, 'reuse' if reuse else 'base', round(peaks[reuse], 1),
                      'held_before_mib', round(held / 2**20, 1), flush=True)
            deltas[layers] = peaks[True] - peaks[False]
            equal = [torch.equal(a, b) for a, b in zip(grads[False], grads[True])]
            finite = all(torch.isfinite(x).all().item() for x in grads[False])
            print('peak_delta_mib', name, 'layers', layers, round(
                deltas[layers], 1), 'grad_equal', equal, 'finite', finite, flush=True)
            del step, grads
            torch.cuda.empty_cache()
        print('per_layer_increment_mib', name, '1->4', round((deltas[4] - deltas[1]) / 3, 1),
              '4->12', round((deltas[12] - deltas[4]) / 8, 1), flush=True)


def matrix():
    shapes = {
        'packed-V256': dict(H=16, HV=16, K=128, V=256, lens=[4095, 4098, 4096, 4095]),
        'gva-H8-HV16-V256': dict(B=2, T=8192, H=8, HV=16, K=128, V=256),
        'small-D64': dict(B=4, T=2048, H=4, HV=4, K=64, V=64),
        'fp16-V256': dict(B=2, T=8192, H=16, HV=16, K=128, V=256, dtype=torch.float16),
    }
    med = statistics.median
    for name, cfg in shapes.items():
        step = make(**cfg)
        res = {r: step(r) for r in (False, True)}
        equal = [torch.equal(a, b) for a, b in zip(res[False][1], res[True][1])]
        print('grad_equal', name, equal, flush=True)
        del res
        for _ in range(16):
            step(False)
            step(True)
        pairs = []
        for i in range(8):
            pair = {r: timed(lambda r=r: step(r), 12) for r in ((False, True) if i % 2 == 0 else (True, False))}
            pairs.append((pair[False], pair[True]))
            print('pair', name, i, *pairs[-1], flush=True)
        print('fwdbwd base/reuse', name, *[round(med(x), 5) for x in zip(*pairs)], 'paired_pct',
              round(med(100 * (1 - b / a) for a, b in pairs), 3), 'wins', sum(b < a for a, b in pairs), flush=True)


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.get_device_name(), sys.argv[1], flush=True)
    {'options': options, 'memory': memory, 'matrix': matrix}[sys.argv[1]]()
