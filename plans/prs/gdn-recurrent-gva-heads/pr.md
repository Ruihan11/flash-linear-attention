# [Fix] Validate GVA head counts in recurrent GDN

## Summary

`fused_recurrent_gated_delta_rule` now raises `ValueError` before dispatch when the number of value heads (`HV`) is not divisible by the number of key heads (`H`). This matches the documented grouped value attention contract and the existing check in `chunk_gated_delta_rule`.

The recurrent kernel selects key head `i_hv // (HV // H)`. With `H=4, HV=2` the divisor is zero. With `H=4, HV=6`, the last two value heads map to key head indices 4 and 5, outside the valid range 0–3. The exception reports both input head counts.

## Test plan

The two regression cases call the public API with BF16 inputs and intercept `FusedRecurrentFunction.apply`. With the implementation from `f8bfd49f`, both cases failed because invalid inputs reached `apply`. With the patch, both raised the expected `ValueError` before dispatch.

On an NVIDIA A800-SXM4-80GB, `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py -k fused_recurrent` produced **33 passed, 100 deselected**. This includes the existing legal grouped-head case that compares default beta against explicit ones for both output and final state.

## Benchmark / NCU

N/A for this Python argument-validation change.

## Breaking changes

Calls with invalid GVA head counts now raise `ValueError` at the public entry point.

## Checklist

- [x] I have read [CONTRIBUTING.md](https://github.com/fla-org/flash-linear-attention/blob/main/CONTRIBUTING.md) and follow its conventions (code style, docstrings, commit prefixes).
- [x] I have read [AGENTS.md](https://github.com/fla-org/flash-linear-attention/blob/main/AGENTS.md) and, where my change matches its scope, the relevant skill under [.agents/skills](https://github.com/fla-org/flash-linear-attention/tree/main/.agents/skills).
- [ ] Dependent tests pass locally or in CI, and new behavior is covered by tests where applicable (tick as N/A for changes with no testable code, e.g. docs-only).
- [x] Kernel changes include same-hardware before/after benchmark numbers, dense + varlen where applicable (tick as N/A when no kernel code changed).
- [ ] This PR is minor/cosmetic-only (typo, formatting, style-only tweaks) — tick only if it is, and justify below.
- [ ] I understand this PR exceeds 500 changed lines and have explained its single purpose and why it should stay together below.
