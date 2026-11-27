import pytest
import torch

from fla.utils.testing import get_err_ratio
from tests.cp import test_kda as kda


def measure(rank, world_size, cut, layout):
    check = kda.assert_close

    def compare(name, ref, value, **kwargs):
        nonfinite = (~torch.isfinite(ref)).sum().item() + (~torch.isfinite(value)).sum().item()
        delta = ref[:, cut-8:cut+8].float() - value[:, cut-8:cut+8].float()
        print(f'METRIC {layout} {cut} {name} {get_err_ratio(ref, value):.8g} {delta.abs().max().item():.8g} {nonfinite}', flush=True)
        assert nonfinite == 0
        check(name, ref, value, **kwargs)

    kda.assert_close = compare
    kda._run_chunk(rank, world_size, H=12, D=128, lengths=[cut, 8192-cut], dtype=torch.bfloat16,
                   use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0, layout=layout)


@pytest.mark.parametrize('cut', [512, 511, 513])
@pytest.mark.parametrize('layout', ['zigzag', 'contiguous'])
def test_offset(run_distributed, layout, cut):
    run_distributed(worker=measure, world_size=8, cut=cut, layout=layout)
