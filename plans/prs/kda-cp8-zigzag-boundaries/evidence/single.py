import pytest
import torch

from fla.utils.testing import get_err_ratio
from tests.cp import test_kda as kda


@pytest.mark.parametrize('cut', [511, 512, 513])
def test_single(single_rank_group, cut):
    q, k, v, g, beta, do, a, dt = kda._make_inputs(0, 8192, 12, 128, torch.bfloat16, True, True, -5.)
    offsets = [0, cut, 8192]
    ref = kda._reference(q, k, v, g, beta, do, offsets, a, dt, True, -5.)
    inputs = [t.requires_grad_() for t in (q, k, v, g, beta)]
    o, _ = kda.chunk_kda(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=torch.tensor(offsets, device=q.device, dtype=torch.long),
                         A_log=a[:12], dt_bias=dt, lower_bound=-5., safe_gate=True, use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True)
    o.backward(do)
    for name, target, actual in zip(('Output', 'dq', 'dk', 'dv', 'dg', 'db'), ref, (o.detach(), *(x.grad for x in inputs))):
        count = (~torch.isfinite(actual)).sum().item() + (~torch.isfinite(target)).sum().item()
        delta = target[:, cut-8:cut+8].float() - actual[:, cut-8:cut+8].float()
        print(f'METRIC single {cut} {name} {get_err_ratio(target, actual):.8g} {delta.abs().max().item():.8g} {count}', flush=True)
        assert count == 0
        kda.assert_close(name, target, actual, ratio=8e-3, warning=name in ('dg', 'db'))
