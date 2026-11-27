# GDN recurrent 的 GVA 头数校验

选题已在 `main@8973fdd7` 完成入口验证。Gated DeltaNet 的 GVA 把多个 value head 映射到一个 Q/K head，要求每个 Q/K head 分到相同的正整数个 value head。`chunk_gated_delta_rule` 已在 Python 入口校验整除关系，`fused_recurrent_gated_delta_rule` 的文档也写明了这一约束。[chunk 校验](../../../fla/ops/gated_delta_rule/chunk.py) · [recurrent 实现](../../../fla/ops/gated_delta_rule/fused_recurrent.py)

## 实验记录

在 A800 上创建 BF16 CUDA 张量，固定 `B=1,T=2,H=4,K=V=32`。分别调用两个公共入口，用 `unittest.mock.patch` 在各自的 autograd `apply` 处截断执行，观察公开入口是否接受输入。

| value heads | chunk | fused recurrent |
| --- | --- | --- |
| 2 | `ValueError` | 到达 dispatch |
| 6 | `ValueError` | 到达 dispatch |
| 8 | 到达 dispatch | 到达 dispatch |

## 开发计划

- [x] **从 `main` 创建主题分支。** 审批后核对工作树和基线，在当前已核验的 `main@f8bfd49f` 上执行 `git switch -c fix/gdn-recurrent-gva-heads`。两项改动都在该分支完成。
- [x] **固定失败用例并修复入口。** 在 `tests/ops/test_gdn.py` 用参数表覆盖 `(H, HV)=(4,2)` 和 `(4,6)`。调用公开 recurrent 接口，临时拦截 `FusedRecurrentFunction.apply`，使基线测试能够安全地报告错误输入已到达底层。然后在 `fla/ops/gated_delta_rule/fused_recurrent.py` 中读取 Q/K 头数和 value 头数，按 chunk 入口已有的整除条件与 `ValueError` 文案，在调用 `apply` 前拒绝这两组输入。断言错误信息包含实际头数。
- [x] **验证拒绝路径与合法路径。** 在 A800 上运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py -k 'fused_recurrent'`。两个非法比例应在入口抛错，已有等头数和合法 GVA 数值测试应通过，包括 `test_fused_recurrent_default_beta_gva`。运行 `python scripts/find_dependent_tests.py fla/ops/gated_delta_rule/fused_recurrent.py`，检查涉及的 layer 调用方，复核分支 diff 并将命令、结果写回本记录。

## 验证结果

在 `main@f8bfd49f` 创建分支 `fix/gdn-recurrent-gva-heads`。修复前两个非法比例均触发拦截 `FusedRecurrentFunction.apply` 的测试桩；修复后均在 Python 入口得到包含实际头数的 `ValueError`。

A800 上运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py -k fused_recurrent`，33 passed，100 deselected。覆盖合法 GVA、默认 beta、输出和状态的数值对照。`python scripts/find_dependent_tests.py fla/ops/gated_delta_rule/fused_recurrent.py` 列出 11 个相关测试文件。直接 layer 调用方包括 GatedDeltaNet、MoM 和 GatedDeltaProduct。Ruff 与 `git diff --check` 均通过。

## 状态

**merged** — PR [#1851](https://github.com/fla-org/flash-linear-attention/pull/1851) 已合并，merge commit 为 `399504c60f865e8fb76c949eeb869cab9d3a676d`。


建议让 recurrent 入口按 chunk 的规则拒绝非法头数。在 `tests/ops/test_gdn.py` 增加两个非法比例的 `ValueError` 断言，并运行已有合法 GVA recurrent 测试。候选标题为 `[Fix] Validate GVA head counts in recurrent GDN`。

本次 GitHub 查重覆盖 `num_v_heads GatedDeltaNet` 和 `fused_recurrent_gated_delta_rule divisible` 等关键词。相邻在审工作包括 [grouped backward 梯度 #1747](https://github.com/fla-org/flash-linear-attention/pull/1747) 和 [decode 输出缓冲区 #1808](https://github.com/fla-org/flash-linear-attention/pull/1808)。
