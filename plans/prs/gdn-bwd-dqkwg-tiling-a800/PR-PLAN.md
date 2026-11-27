# GDN backward tiling autotune plan

## Context

The latest upstream main is `399504c6`. A800 SM80 measurements show that `chunk_bwd_kernel_dqkwg` benefits from a smaller V tile on BF16 `V=256` workloads, but the benefit depends on Triton selecting a compatible pipeline depth.

Observed A800 results on the existing full GDN path were approximately 3.1–4.3% faster for `V=256` with `BV=64` across dense, packed, and GVA workloads. The same candidate regressed `V=128` by about 1.3%. Forcing `BV=32` and the baseline to use the same two-stage pipeline made the smaller tile slower, so a fixed A800 tile branch would encode the wrong cause and could regress other shapes.

The intended outcome is a small, independently reviewable performance PR that lets the existing Triton autotune mechanism choose a valid V-tile and pipeline configuration for supported inputs. The public `chunk_bwd_dqkwg` signature and non-NVIDIA backend registrations remain unchanged.

## Approach

1. Prototype one additional `BV=64` configuration for the shared NVIDIA `chunk_bwd_kernel_dqkwg` autotune path, with the measured warp and stage values.
2. Keep the wrapper's existing architecture and dtype guards as the upper-bound policy. The candidate must not bypass the current Ampere, Hopper, Ada, or fallback tile caps.
3. Ensure the candidate is only offered when its shared-memory requirements are valid. Do not add a universal configuration that can fail compilation on smaller-memory devices.
4. Validate the selected configuration through the normal GDN dispatch before changing production code.
5. If the candidate causes cold-start regressions, compilation failures, or shape regressions, retain the experiment as killed and do not submit the PR.

## Files to modify

Production changes, if the validation gate passes:

- `fla/ops/common/chunk_o.py`
  - Extend the existing `chunk_bwd_kernel_dqkwg` autotune configuration.
  - Reuse the existing `check_shared_mem`, `NUM_WARPS`, and `autotune_cache_kwargs` policies.
  - Keep the wrapper's `BK/BV` interface and all backend dispatch contracts intact.

Focused coverage, only where current cases do not exercise the new selection boundary:

- `tests/ops/test_gdn_kernels.py`
  - Extend the existing `chunk_bwd_dqkwg` reference coverage for the relevant V widths, dtype, GVA, gate, and optional `w`/`dv` paths.

Validation-only files in the experiment directory:

- `plans/prs/gdn-bwd-dqkwg-tiling-a800/evidence/`
  - Keep benchmark scripts and raw logs untracked.
- `plans/prs/gdn-bwd-dqkwg-tiling-a800/README.md`
  - Record final scope, measured timings, selected configs, and platform limits.

Do not modify `fla/ops/common/backends/triton_ascend/chunk_o.py`, TileLang backends, or public API definitions in this PR.

## Reuse

- Reuse `chunk_bwd_kernel_dqkwg` and its existing `fla_cache_autotune` decorator in `fla/ops/common/chunk_o.py`.
- Reuse the wrapper's current architecture and dtype checks for the maximum supported tile size.
- Reuse `chunk_bwd_dqkwg_ref` and `test_chunk_bwd_dqkwg` in `tests/ops/test_gdn_kernels.py` for numerical validation.
- Reuse the public `chunk_gated_delta_rule` path and the `chunk_gdn` benchmark entry for end-to-end timing.
- Treat the separate Ascend implementation and TileLang registration as compatibility boundaries, not as implementation targets.

## Steps

- [x] Rebase the experiment measurements against `main@399504c6` and confirm the working tree contains no production changes.
- [x] Add the smallest experiment-only configuration candidate and log the actual `BV`, `num_warps`, `num_stages`, registers, spills, shared memory, and cold compile/autotune time.
- [x] Compare the default and candidate configurations on A800 for dense `V=128/256`, packed `V=256`, and GVA `H=8,HV=16` using alternating CUDA-event pairs.
- [x] Run the direct kernel correctness comparison for BF16 and FP16, `use_g/use_w` combinations, GVA, and non-divisible V tails.
- [x] Run the full GDN forward/backward path through normal dispatch and verify that candidate selection does not alter gradients beyond the existing test budget.
- [x] Check shared callers of `chunk_bwd_dqkwg`, including Delta Rule, Simple GLA, Comba, and preconditioned GDN, for configuration or correctness regressions.
- [x] Evaluate the cold-start cost and cache-key behavior. Reject the change if it adds unacceptable compilation work or reuses an incompatible cached configuration.
- [x] If all gates pass, implement the production configuration change and add only the focused regression coverage needed for the new boundary.
- [x] Update the experiment record and prepare a PR description that states the A800-only evidence and leaves Hopper, Ada, Blackwell, and Ascend claims unmade without hardware evidence.

## Verification

Run targeted correctness first:

```bash
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg
```

Run the public GDN regression suite and dependent-test discovery after the targeted gate:

```bash
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn.py
.venv/bin/python scripts/find_dependent_tests.py fla/ops/common/chunk_o.py
```

Run the standard operator benchmark on the same A800 with paired baseline and candidate processes. Record absolute milliseconds, paired percentage changes, dispersion, selected Triton resources, and cold-start cost:

```bash
.venv/bin/python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base 399504c6
```

Run formatting and static checks only on changed production and test files:

```bash
.venv/bin/pre-commit run --files fla/ops/common/chunk_o.py tests/ops/test_gdn_kernels.py
```

The PR is ready only if the full GDN path improves the intended `V=256` workloads, does not materially regress `V=128` or shared callers, passes numerical coverage, and has no unsupported cross-platform performance claim.
