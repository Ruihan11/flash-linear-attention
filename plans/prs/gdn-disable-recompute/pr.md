# [Perf] Add disable_recompute to chunk GDN

## Summary

`chunk_gated_delta_rule` gains an opt-in `disable_recompute` flag, matching the existing flag on KDA, GDN-2, DPLR and RWKV7. When enabled, the forward saves `w`, `v_new` and `h`, and the backward skips `recompute_w_u_fwd` and the second `chunk_gated_delta_rule_fwd_h`. On an A800 this lowers forward+backward time by 11.0–13.9% with bitwise-identical outputs and gradients.

The cost is memory held from forward to backward: `w`, `v_new` and `h` per call. For BF16 `B=2, T=8192, HV=16, K=128, V=256` that is 576 MiB per layer (measured by stacking 1, 4 and 12 calls; 12 calls raise peak allocation from 5224 to 11818 MiB). The default is `False` and the default path is unchanged.

Limitations: the flag only affects the Triton path. FlashQLA is a fused implementation without a recompute step and ignores it. `GatedDeltaNet` does not expose the flag yet. It cannot be combined with `return_intermediate_states`.

## Test plan

Hardware: NVIDIA A800-SXM4-80GB (SM80), driver 580.65.06, PyTorch 2.10.0+cu130, CUDA 13.0, Triton 3.6.0. Base `6511ea3e`.

- New `test_chunk_disable_recompute` (5 cases) asserts `torch.equal` between the two paths for `o`, the final state and the gradients of `q, k, v, g, beta, h0` (and `A_log` with the in-kernel gate), and patches `recompute_w_u_fwd` to raise so the saved path is known to be taken. Cases cover dense and packed inputs, grouped value heads, both state layouts, the in-kernel gate, `chunk_size=32`, `K != V`, a non-multiple `T`, FP16 and BF16. A sixth test covers the `return_intermediate_states` rejection. All six fail on the base and pass here.
- New `tests/cp/test_gdn.py::test_chunk_disable_recompute` (2 cases, two ranks) compares the saved path under context parallel with the full-sequence reference.
- `FLA_CI_ENV=0 pytest -q tests/ops/test_gdn.py tests/ops/test_gdn_kernels.py tests/ops/test_gdp.py tests/layers/test_gated_deltanet.py tests/models/test_modeling_gated_deltanet.py`: 203 passed, 31 skipped.
- `FLA_CI_ENV=0 pytest -q tests/cp/test_gdn.py -k cp2`: 8 passed.
- With the default `disable_recompute=False`, outputs and gradients are `torch.equal` to the base on three shapes (dense `V=256`, dense `V=128`, packed GVA FP16).
- `ruff check` and the pre-commit hooks pass on the three changed files.
- Not run: the cp4 and cp8 cases, the other files listed by `scripts/find_dependent_tests.py fla/ops/gated_delta_rule/chunk.py`, and any SM90+, Blackwell or Ascend device. The FlashQLA path cannot run on SM80.

## Benchmark / NCU

Same GPU, same process, `use_qk_l2norm_in_kernel=True`, seed 42. Each row is the median of 8 alternating pairs, 12 forward+backward calls per timed block, CUDA events. All 8 pairs favoured the saved path in every row.

| Workload | Recompute (ms) | Saved (ms) | Time reduction |
| --- | ---: | ---: | ---: |
| BF16 dense `B=2, T=8192, H=HV=16, K=128, V=256` | 6.105 | 5.280 | 13.5% |
| BF16 dense, `V=128` | 4.145 | 3.598 | 13.2% |
| BF16 packed, four sequences, `T=16384, V=256` | 5.836 | 5.022 | 13.9% |
| BF16 grouped heads `H=8, HV=16, V=256` | 6.107 | 5.300 | 13.2% |
| BF16 `B=4, T=2048, H=4, K=V=64` | 1.799 | 1.601 | 11.0% |
| FP16 dense, `V=256` | 6.057 | 5.234 | 13.6% |

Default path, `python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base 6511ea3e`: forward 0.99–1.00x and forward+backward 0.99–1.00x on five of six shapes. `B=8, T=1024, H=8, D=64` forward+backward read 0.97x (1.757 → 1.804 ms). Four alternating reruns of that shape gave base medians 1.753–1.813 ms and candidate medians 1.773–1.809 ms, so the two ranges overlap.

NCU was not collected. The saving is two whole kernel launches, and only SM80 was measured.

## Breaking changes

None. One new keyword argument with default `False`.
