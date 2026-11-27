# [Fix] Preserve the autocast output dtype in biased GroupLinear

## Summary

Under CUDA autocast, `GroupLinear` with FP32 parameters produced an FP32 output when its FP32 bias was added to a BF16 or FP16 projection. Casting the bias to the projection dtype during autocast keeps the result in the same dtype as the equivalent block-diagonal `F.linear`.

## Test plan

On an A800-SXM4-80GB with PyTorch 2.10.0+cu130, the new biased BF16 and FP16 cases failed on the output dtype at baseline `cc7018a4`. The new CPU BF16 biased case also fails at baseline because the output is promoted to FP32. The two CUDA bias-free controls passed. On the candidate, `FLA_CI_ENV=0 python -m pytest -q tests/modules/test_grouplinear.py` passed all 13 cases. The added cases check output dtype, forward values and gradients for inputs and parameters against `F.linear`, with the existing 0.005 relative RMS tolerance. Coverage includes CPU BF16 autocast plus CUDA BF16 and FP16 autocast with and without bias. `scripts/find_dependent_tests.py` identified this test file as the dependent suite.

## Benchmark / NCU

N/A for this module-level dtype correction.

## Breaking changes

Consumers that relied on implicit FP32 promotion of the biased autocast output can explicitly call `.float()` on the result. The parameter layout and state dict are unchanged.

## Checklist

- [x] I have read [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/main/CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](https://github.com/fla-org/flash-linear-attention/blob/main/AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](https://github.com/fla-org/flash-linear-attention/tree/main/.agents/skills).
- [x] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.
