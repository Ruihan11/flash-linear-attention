# GDN chunk 训练路径的算法优化研讨

状态：in progress（研讨阶段，只读代码和已有日志，未改生产文件）

## Context

`chunk_bwd_dqkwg` 的分块调参实验已归档为 killed：A800 上 `V=256` 稳态收益 3.2–4.3%，但首次 autotune 多 1.06 秒，且结论依赖 Ampere 的 shared-memory 预算。[调参实验](../gdn-bwd-dqkwg-tiling-a800/README.md)

本计划换一个方向：找数学上等价、不依赖具体架构的改动。先总结社区已合并的算法类性能 PR 的做法，再把同样的手法逐条对照 GDN chunk 前反向，挑出一个约 100 行的候选。分支 `perf/gdn-backward-vtile-autotune`，基于 `main@399504c6`，后续按最终候选改名。

## 社区算法类性能 PR 的五种做法

| 做法 | 先例 | 具体手法 | 验证方式 |
| --- | --- | --- | --- |
| 删掉被覆盖或无人读取的工作 | #1825（+3/−2）、#1416、#1717（+7/−6）、#1532、#1200 | 输出被 kernel 全量覆盖时 `zeros_like` 改 `empty_like`，去掉多余的 host sync | 与 base `torch.equal`，varlen 保留清零 |
| 复用已经算出的中间量 | #1048、#1595（+30/−20）、#916、#641、#1199 | #1048 在 Ascend `dqkwg` 中把 `ds = do @ vᵀ` 只算一次，同时供 `dq` 和 `dk` 使用。#1595 复用已有扫描的累计值，省掉一次归约 | 输出逐位一致 |
| 不显式构造大矩阵 | #1752（+350/−8）、#1601、在审 #1826 | #1752 把 `[K,K]` 变换改为按行条带直接作用，A100 `K=256` 从 225 ms 降到 2.7 ms。#1826 把 KDA 改为 Dual-WY，省掉 V 侧 WY 反向 | 对 FP32 递推参考比误差，附各形状回退说明 |
| 按资源决定融合或拆分 | #972、#1592（+38/−5）、#1210 | 融合减少读写；寄存器 spill 时反过来拆开 | 寄存器/spill 数据加同卡 A/B |
| 按梯度需求跳过计算 | #1296（+70/−29）、在审 #1417 | 不需要的梯度不算 | 各梯度需求组合对照 |

共同点：标题直接写被省掉的工作，改动等价或在现有容差内，描述里列出不受益和回退的形状。改变数值路径的（#1826、在审 #1703 ReWY）都附了对 FP32 参考的误差表，并且体量在 500–1000 行。

## GDN `fwdbwd` 的时间分布

来自已有 A800 trace，BF16，`B=2,T=8192,H=HV=16,K=128,V=256`，一次前向加反向共 6.08 ms。[trace 日志](../gdn-wy-vtile-a800/trace.log)，原始文件 `profile/gdn-wy-vtile/v256.json`。

| kernel | 占比 | 调用次数 | 说明 |
| --- | ---: | ---: | --- |
| `chunk_bwd_kernel_dqkwg` | 20.2% | 1 | |
| `fwd_kernel_h_blockdim64` | 17.8% | 2 | 前向一次，反向重算一次，各约 8.9% |
| `bwd_kernel_dhu_blockdim64` | 17.6% | 1 | 串行沿 chunk 递推 |
| `prepare_wy_repr_bwd_kernel` | 10.6% | 1 | |
| `chunk_fwd_kernel_o` | 9.0% | 1 | |
| `recompute_w_u_fwd_kernel` | 9.0% | 2 | 前向一次，反向重算一次，各约 4.5% |
| `chunk_bwd_kernel_dv_local` | 3.9% | 1 | |
| 其余（l2norm、kkt_solve、逐元素、cumsum） | 11.9% | | |

反向里重算 `w/u` 和 `h/v_new` 合计约 13.4%，是单项最大的可省工作。

## 逐条对照得到的候选

| 候选 | 对应做法 | 上限（V=256） | 体量 | 判断 |
| --- | --- | ---: | --- | --- |
| A. 反向按梯度需求跳过 | 按梯度需求 | 未测 | 小 | **需先读代码**：`ChunkGatedDeltaRuleFunction.backward` 是否已检查 `needs_input_grad` 尚未核对 |
| B. 可选的"保存中间量、反向不重算" | 复用中间量 | 约 13.4% | 约 60–100 行 | 收益最大。仓库已有先例：KDA 和 precond KDA 的 `disable_recompute` 参数。代价是多存 `h`（本负载 268 MB）和 `w/u/v_new`（335 MB），并新增一个公开参数，需要维护者认可 |
| C. `dqkwg` 中 `ds = do @ vᵀ` 跨 K 分块去重 | 复用中间量（#1048） | `K≤BK` 时为 0 | 中 | 每个 K 分块的 program 都完整算一遍 `ds` 并重读 `v/do`。`K=128,BK=128` 时只有一个分块，没有冗余。只对 `K=256` 或 `BK=64`（FP32、Ada）有效，需要另测 |
| D. `A` 的清零 | 删掉被覆盖的工作 | 小于 1% | 小 | kernel 只写下三角块，读取方按整行加载，上三角依赖清零。要去掉必须给读取方加掩码（#1826 的做法），收益太小，不做 |
| E. GDN 改 Dual-WY | 不构造大矩阵 | #1826 在 KDA 上 5–9% | 约 1000 行 | 维护者正在 KDA 上做，GDN 版本大概率由其延续。体量和撞车风险都不适合 |
| F. 融合 `dqkwg` 与 `dv_local` | 融合 | 小于 3.9% | 中 | #1592 显示 SM8x 上融合反向会 spill，不做 |

撞车检查：#1703（GDN opt-in ReWY 反向，Draft）、#1465（TileLang chunk 反向）、#1747（GVA dq/dk 舍入）都改 GDN 反向。候选 B 改的是 `chunk.py` 的保存与重算逻辑，与 #1703 同文件，需要在动手前看其 diff。

## Approach

推荐顺序是 A → B → C，前一个成立就停。

1. 先核对 A，只读代码，十分钟内有结论。若反向在不需要某些梯度时仍全量计算，这是最小、最容易合并的 PR。
2. 若 A 无空间，做 B 的可行性实验：用实验目录内的包装函数把前向的 `w/u/h/v_new` 传给反向，量出实际 `fwdbwd` 收益和峰值显存增量，再决定是否向维护者提议参数。
3. C 只在 `K=256` 负载上有明确冗余时才做，先量 `K=256` 下 `dqkwg` 的占比。

D、E、F 不做，原因记录在上表。

## Files to modify

研讨阶段只写本目录。若候选晋级：

- A 或 B：`fla/ops/gated_delta_rule/chunk.py`，测试在 `tests/ops/test_gdn.py`
- C：`fla/ops/common/chunk_o.py`，测试在 `tests/ops/test_gdn_kernels.py`

不改 Ascend 和 TileLang backend。

## Reuse

- `disable_recompute` 的接口和保存逻辑：`fla/ops/precond_kda/chunk.py`、`fla/ops/kda/`
- #1048 的 `ds` 复用结构：`fla/ops/common/backends/triton_ascend/chunk_o.py`
- 配对计时脚本：`plans/prs/gdn-bwd-dqkwg-tiling-a800/evidence/paired-shapes.py`
- trace 采集：`plans/prs/gdn-wy-vtile-a800/bench.py`
- 参考实现：`tests/ops/test_gdn_kernels.py::chunk_bwd_dqkwg_ref`，公开入口测试 `tests/ops/test_gdn.py`

## Steps

- [x] 检索已合并和在审的 `[Perf]` PR，归纳算法类做法
- [x] 读 `chunk_gated_delta_rule_bwd` 调用链和 `chunk_bwd_kernel_dqkwg`，结合已有 trace 列出时间分布
- [x] 核对候选 A：读 `ChunkGatedDeltaRuleFunction.backward` 和各 wrapper 对 `needs_input_grad` 的处理
- [x] 读 #1703、#1465、#1747 的 diff，确认与 A/B/C 的重叠范围
- [x] 读 KDA `disable_recompute` 的引入 PR（#744）和维护者对显存换时间的态度
- [ ] 候选 B 可行性实验：实验内包装，A800 上 dense `V=128/256`、packed、GVA 的配对 `fwdbwd` 和 `torch.cuda.max_memory_allocated` 增量，梯度与默认路径 `torch.equal`
- [ ] 候选 C 预检：`K=256,V=256` 的 trace，量 `dqkwg` 占比和 NK
- [ ] 按结果选一个候选，写独立的实现计划，更新 `plans/README.md`

## 只读核对的结果

**候选 A：空间很小，不单独做。** `ChunkGatedDeltaRuleFunction.backward` 没有读取 `ctx.needs_input_grad`，总是算出 `dq/dk/dv/dg/db/dh0`。但 GDN 层里这五个输入都来自可训练的投影，常规训练全部需要。各梯度共用 `dhu` 递推和 `dqkwg`，冻结单个输入省不掉主要 kernel。未实测冻结场景的耗时。

**重叠范围。** #1703（Draft）把 `chunk_gated_delta_rule_bwd` 尾部的 `chunk_bwd_dqkwg` 加 `prepare_wy_repr_bwd` 抽成带 `@dispatch` 的 `chunk_gated_delta_rule_bwd_dqkvbg`，入参包含 `w/v_new/h`。候选 B 改的是同一函数开头的重算和 `forward` 的保存列表，同文件不同段，逻辑不冲突，谁后合并谁 rebase。#1747 改 `chunk_o.py` 中 `dq/dk` 的分配 dtype 和求和后的舍入，与候选 C 同一 wrapper，会冲突。#1465 改 `wy_fast.py` 和各算子的 backend 注册，不碰 `gated_delta_rule/chunk.py`。

**`disable_recompute` 先例。** KDA 和 precond KDA 都有这个公开参数，由维护者在 #744 引入，默认 `False`。precond KDA 的做法是 `forward` 里算出 `w/u/kg/v_new/h`，默认置 `None`，开启时存进 `save_for_backward`，反向用 `if not disable_recompute` 包住重算。文档写的是“对训练小模型有利，代价是显存增加”。#744 的描述是自动生成的摘要，没有维护者对这个取舍的讨论。GDN 的 `chunk_gated_delta_rule_fwd` 目前只在 `return_intermediate_states` 时返回 `h`，不返回 `w/u/v_new`。

## Verification

研讨阶段的产出是本文件的结论和 `evidence/` 下的日志。候选晋级后沿用既有门槛：

```bash
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn_kernels.py
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn.py
.venv/bin/python scripts/find_dependent_tests.py fla/ops/gated_delta_rule/chunk.py
.venv/bin/python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base 399504c6
```

等价改动要求与 base 的输出和梯度 `torch.equal`。A800 计时只作为 SM80 证据。

## 需要你决定

1. 候选 B 会新增公开参数并用显存换时间。你是否接受这种取舍，还是只考虑不增加显存、不改接口的改动（那样只剩 A 和 C）？
2. 100 行的目标是否仍然是硬约束？
3. 目标负载是否继续用 `K=128,V=256`？候选 C 只在 `K=256` 这类负载上有意义。
