# GDN chunk 反向复用前向中间量（`disable_recompute`）

状态：in progress（阶段 0 通过，阶段 1a 通过，1b、1c 完成，未提交、未推送）。分支 `perf/gdn-disable-recompute`，基于 `origin/main@6511ea3e`。相对 `399504c6` 只多了 #1740（GDN2 的 CP 测试），没有改动 GDN chunk、共享 kernel 或 `tests/ops/test_gdn.py`。

## Context

[算法优化研讨](../gdn-algorithmic-optimization/README.md)筛出的主要候选。`chunk_gated_delta_rule_bwd` 开头会重算两样前向已经算过的东西：`recompute_w_u_fwd` 得到 `w/u`，`chunk_gated_delta_rule_fwd_h` 得到 `h/v_new`。把前向结果存进 `save_for_backward`，反向就可以跳过这两个 kernel。

## 是否值得做：纸面判断

这是阶段 0 之前的判断，实测结果见下一节。依据如下。

| 依据 | 数据 | 来源 |
| --- | --- | --- |
| 可省时间上限，`V=256` | (545/2 + 1080/2) / 6076 µs = **13.4%** | [A800 trace](../gdn-wy-vtile-a800/trace.log) |
| 可省时间上限，`V=128` | (364/2 + 709/2) / 4105 µs = **13.1%** | 同上 |
| 仓库先例 | KDA、precond KDA、GDN2、Triadic GDN、DPLR、RWKV7 的 chunk 入口都有同名公开参数 `disable_recompute`，默认 `False` | `grep -rl disable_recompute fla/ops` |
| 既有测试写法 | `tests/ops/test_gdn2.py::test_chunk_varlen`、`tests/ops/test_kda.py` 把 `disable_recompute` 作为参数化维度 | 同上 |
| 不依赖架构 | 省掉的是整次 kernel 调用，不涉及 tile、warp、shared memory | 代码结构 |

GDN 是这组算子里唯一没有该参数的 delta 类 chunk 入口，所以这个 PR 更像"补齐已有接口"，不是提出新取舍。

三个会削弱结论的因素，实验必须回答：

1. **13.4% 是占比推出的上限。** 保存的张量要多活一段时间，分配器行为会变，实际收益可能更低。
2. **显存代价在单算子基准里看不出来。** 默认路径的反向本来就会临时分配 `w/u/h/v_new`，单次调用的峰值可能几乎不变。真实代价是多层模型里每层的这些张量从前向一直保留到反向。本负载每层约 603 MB（`h` 268 MB，`w` 67 MB，`u` 和 `v_new` 各 134 MB，BF16），12 层约 7.2 GB。必须用多层堆叠来量。
3. **已有文档把适用范围写成"小模型训练"。** 收益和显存要在小形状上也量一遍，PR 描述沿用同样的定位。

## 阶段 0 结果（通过）

GPU 1，A800-SXM4-80GB，计时前后均空闲。`6511ea3e`，PyTorch 2.10.0+cu130，BF16，`B=2,T=8192,H=HV=16,K=128`，seed 42，8 组交错配对，每组 12 次 `fwdbwd`。[脚本](evidence/reuse.py) · [日志](evidence/stage0.log)

| V | 默认 | 复用 | 配对降幅 | 胜出组数 | 五个梯度 `torch.equal` |
| --- | ---: | ---: | ---: | ---: | --- |
| 256 | 6.110 ms | 5.288 ms | **13.44%** | 8/8 | 全部一致 |
| 128 | 4.146 ms | 3.597 ms | **13.23%** | 8/8 | 全部一致 |

复用路径少一次 `recompute_w_u_fwd` 和一次 `chunk_gated_delta_rule_fwd_h`（`h` 调用 2 → 1，`wu` 重算 1 → 0）。实测降幅与 trace 推出的上限（13.4% / 13.1%）基本相同，超过 6% 门槛。

单次调用的峰值显存没有上升：`V=256` 为 2730 → 2538 MiB，`V=128` 为 1834 → 1706 MiB。这符合预期，单算子基准看不出显存代价，每层的保留成本要靠阶段 1 的多层堆叠来量。

## GPU 使用规则

每次测试前运行 `evidence/wait-gpu.sh` 取一张空闲卡（利用率 ≤ 5%，已用显存 ≤ 500 MiB）。没有空闲卡就睡 10 分钟再查。计时结束后再查一次，中途被占用的结果作废重跑。

## 阶段 1 前的只读核对

以下来自读代码，都还没有运行验证。

| 问题 | 结论 | 对 PR 的影响 |
| --- | --- | --- |
| KDA 在 CP 下怎么处理 `disable_recompute` | 支持。`fla/ops/kda/chunk_bwd.py` 的 `else` 分支直接取保存的张量，只额外做一次 `expand_h0`。`tests/cp/test_kda.py`、`test_gdn2.py`、`test_dplr.py` 都把它作为参数 | GDN 照做：开启时跳过两次重算，保留 `expand_h0`，后面的 `bwd_dhu_pre_process` 不变。需要 2 张卡验证 |
| FlashQLA 后端收到新参数会怎样 | `chunk_gated_delta_rule_verifier` 和实现都带 `**kwargs`，新参数会落进 `kwargs` 被忽略。该后端只在 SM90/SM100/SM120 且 `K=V=128` 时启用，A800 上不会走到 | 见下方决定 1。A800 无法实测这条路径 |
| 其他调用方 | `fla/ops/gated_delta_product/chunk.py` 直接调用 `chunk_gated_delta_rule_bwd`，不传新参数 | 新参数默认 `None` 时走重算，GDP 行为不变。用 `tests/ops/test_gdp.py` 确认 |
| 入口的参数传递 | `chunk_size`、`use_gate_in_kernel`、`A_log`、`dt_bias` 从 `kwargs` 里取，`Function.apply` 用位置参数 | `disable_recompute` 写成显式关键字参数，追加到 `apply` 末尾，`backward` 的返回元组多一个 `None` |
| 在审 PR | #1703（ReWY，Draft，11-18 后未更新）改 `chunk_gated_delta_rule_bwd` 的后半段。#1453（GDN/KDA 的 CUDA Graph）也在审 | 实现前读两者最新 diff。本改动只动反向开头的重算段和前向的保存段 |

### 需要你决定，已给出默认选择

1. **FlashQLA 上的行为。** 默认选择：不改 `flash_qla.py`，在文档字符串里写明该参数只影响 Triton 路径。FlashQLA 是融合实现，没有"重算"这一步，让 verifier 拒绝会把调用退回更慢的 Triton 路径。另一种做法是 verifier 拒绝，多 2 行。
2. **CP 测试是否进本 PR。** 默认选择：进，给 `tests/cp/test_gdn.py` 加一个 `disable_recompute` 参数并只对 2 卡用例开启，约 10 行。若你想守住 100 行，可以只在本地跑 CP 对照，不提交测试。
3. **层和模型配置的开关。** 默认选择：不进本 PR，`GatedDeltaNet` 层暂时无法开启该参数，PR 描述里写明。

## 阶段 1a 结果（通过）

A800-SXM4-80GB，`6511ea3e`，PyTorch 2.10.0+cu130，seed 42。仍通过实验脚本替换模块内函数，未改生产文件。[脚本](evidence/stage1.py)

**选项覆盖。** 13 项的输出、最终状态和全部梯度在两条路径间 `torch.equal`，数值均有限：dense、`initial_state` 加 `output_final_state`、`state_v_first`、`use_gate_in_kernel`（含 `dA_log/ddt_bias`）、`chunk_size=32`、`T=1000`、packed、packed 加 `h0`、GVA `H=2,HV=4`、FP16、`K=128,V=256`、组合项。[日志](evidence/options.log)

**多层显存。** 每层增量与推算完全一致。[日志](evidence/memory.log)

| 负载 | 每层多保留 | 1 层峰值 默认 → 复用 | 4 层 | 12 层 |
| --- | ---: | --- | --- | --- |
| BF16 `B=2,T=8192,H=16,K=128,V=256` | 576 MiB | 2023 → 2281 MiB | 2896 → 4882 MiB | 5224 → 11818 MiB |
| BF16 `B=4,T=2048,H=4,K=V=64` | 16 MiB | 77 → 89 MiB | 126 → 186 MiB | 257 → 445 MiB |

堆叠后峰值上升，12 层主负载约 2.26 倍。阶段 0 里单次调用峰值下降只是因为默认路径在反向里临时分配同样的张量。这个代价与 KDA 文档"小模型训练、显存换时间"的定位一致。

**计时矩阵。** 8 组交错配对，每组 12 次 `fwdbwd`，全部 8/8 胜出。第一次在 GPU 1 上跑完后利用率读数为 54%，15 秒后回到 0% 且卡上无其他进程，随后在空闲的 GPU 2 上重跑做对照。[GPU 1](evidence/matrix.log) · [GPU 2](evidence/matrix-repeat.log)

| 负载 | 默认 → 复用（GPU 1） | 配对降幅 GPU 1 / GPU 2 |
| --- | --- | --- |
| packed BF16 `V=256`，四条序列 | 5.844 → 5.026 ms | 13.90% / 13.89% |
| GVA BF16 `H=8,HV=16,V=256` | 6.116 → 5.305 ms | 13.25% / 13.16% |
| 小形状 BF16 `B=4,T=2048,H=4,K=V=64` | 1.777 → 1.592 ms | 10.45% / 10.78% |
| FP16 dense `V=256` | 6.066 → 5.242 ms | 13.57% / 13.53% |

没有回退的形状，没有触发停止条件。

**与在审 PR 的重叠（读最新 diff）。** #1703（`e9fc25f2`，Draft）只改 `chunk_gated_delta_rule_bwd` 后半段的 `dqkwg` 与 `prepare_wy_repr_bwd`，与本改动的重算段不重叠。#1453（`10aca841`，Draft，10-16 后未更新）给同一批调用加 `**graph_kwargs`，并改 `save_for_backward`、`apply` 参数和 `backward` 返回元组，与本改动有文本冲突，后合并的一方需要 rebase。

## 阶段 1b/1c 结果

生产改动在工作区，未提交。`git diff --stat`：`chunk.py` +56/−25，`tests/ops/test_gdn.py` +63，`tests/cp/test_gdn.py` +23，合计 167 行，超过 110 行的目标。

- 新测试 6 项在候选上通过，在 base 上全部失败。CP 两卡 8 项通过（GPU 3、4）。
- 回归 203 passed，31 skipped，耗时 51 分钟。[日志](evidence/suites.log)
- 公开参数的配对降幅 11.0–13.9%，与脚本替换一致，梯度逐位一致。[日志](evidence/prod.log)
- 默认路径与 base 的输出和梯度 `torch.equal`。`fwd` 稳定值两边都是约 1.744 ms。
- registry 基准五个形状为 0.99–1.00x。`B=8,T=1024,H=8,D=64` 的 `fwdbwd` 读数为 0.97x，四轮交错复测 base 中位数 1.753–1.813 ms，候选 1.773–1.809 ms，区间重叠，但候选在四轮中有三轮更高。
- 未跑：cp4、cp8，依赖扫描列出的其余测试文件，SM90 及以上设备。

PR 描述见 [pr.md](pr.md)。


## Approach

分两个阶段，第一阶段不过就归档。

**阶段 0：单形状快速判定。** 不改生产代码。在实验脚本里替换 `fla.ops.gated_delta_rule.chunk` 模块内的三个名字：包住 `chunk_gated_delta_rule_fwd_intra` 记下前向的 `w/u`，把 `recompute_w_u_fwd` 换成直接返回记下的值，包住 `chunk_gated_delta_rule_fwd_h` 让第二次调用返回第一次的 `h/v_new`。只测 dense BF16 `B=2,T=8192,H=HV=16,K=128,V=256`。

晋级条件：配对 `fwdbwd` 耗时下降不低于 **6%**，且五个梯度与默认路径 `torch.equal`。低于 6% 说明实际收益不到上限的一半，归档为 killed。

**阶段 1：完整实验。** 阶段 0 通过后才做，矩阵见 Steps。

**生产改动的形状（阶段 1 通过后）。** 照 KDA 的写法：

- `chunk_gated_delta_rule` 和 `ChunkGatedDeltaRuleFunction.forward` 增加 `disable_recompute: bool = False`
- `chunk_gated_delta_rule_fwd` 多返回 `w/u/v_new`，`h` 已经在返回值里
- 开启时把 `w/u/v_new/h` 存入 `save_for_backward`，关闭时存 `None`
- `chunk_gated_delta_rule_bwd` 接受这四个可选张量，开启时跳过两次重算
- 与 `return_intermediate_states` 同时开启时报错，和 KDA 一致
- 文档字符串沿用 KDA 的措辞

- `disable_recompute=True` 且不需要梯度（`torch.no_grad()`）时不保存任何中间量

预计生产约 40–60 行，测试约 40–50 行，合计 80–110 行。默认路径的行为和显存不变。

**测试的形状。** 在 `tests/ops/test_gdn.py` 新增一个参数化测试 `test_chunk_disable_recompute`：同一组输入分别用 `disable_recompute=False/True` 调用，断言输出、最终状态和六个梯度（含 `dh0`）`torch.equal`。参数覆盖 dense、packed、GVA、`state_v_first`、`use_gate_in_kernel`、`chunk_size=32`、`T` 不被 chunk 整除、FP16 和 BF16。另加一条 `return_intermediate_states` 与 `disable_recompute` 同时开启时报错的断言。默认路径与 FP32 参考的对照沿用现有测试，不重复。

## Files to modify

实验阶段只写本目录。晋级后：

- `fla/ops/gated_delta_rule/chunk.py`
- `tests/ops/test_gdn.py`
- `tests/cp/test_gdn.py`（取决于决定 2）

不改 kernel、`fla/layers/gated_deltanet.py`、Ascend 和 TileLang backend。层和模型配置的开关留给后续 PR。

## Reuse

- 接口、保存逻辑和文档措辞：`fla/ops/kda/chunk.py`（`save_for_backward` 段）、`fla/ops/kda/chunk_bwd.py`（`if disable_recompute is False:` 段）
- 测试参数化写法：`tests/ops/test_gdn2.py::test_chunk_varlen`
- 配对计时的 `timed/compare`：`plans/prs/gdn-bwd-dqkwg-tiling-a800/evidence/paired-shapes.py`
- trace 采集：`plans/prs/gdn-wy-vtile-a800/bench.py`
- 标准负载：`benchmarks/ops/registry.py` 的 `chunk_gdn`

## Steps

阶段 0

- [x] 计时前确认所用 A800 空闲。写计划时 8 张卡利用率 86–100%，建分支时 GPU 2–7 已空闲（0 MiB，0%）。每次计时前后各查一次 `nvidia-smi`，中途被占用的结果作废重跑
- [x] 写 `evidence/reuse.py`（模块内替换，不动生产文件），记录默认和复用两条路径实际发起的 kernel 次数，确认复用路径少了一次 `recompute_w_u_fwd_kernel` 和一次 `fwd_kernel_h`
- [x] dense `V=256`：五个梯度 `torch.equal`，8 组交错配对 `fwdbwd`，每组 12 次调用
- [x] 按 6% 门槛决定继续或归档，结果写回本文件和 `plans/README.md`

阶段 1

- [x] 只读核对 CP、FlashQLA、其他调用方和在审 PR，结论见"阶段 1 前的只读核对"

1a. 实验脚本阶段，仍不改生产文件。先做会改变 PR 定位的两项

- [x] **多层显存。** 扩展 `evidence/reuse.py`：把算子顺序堆叠 1、4、12 层（上一层输出作为下一层的 `v`），一次反向，记录两条路径的 `torch.cuda.max_memory_allocated`，给出每层增量并与推算的 603 MB 对比。形状用 `V=256` 主负载和小形状 `B=4,T=2048,H=4,K=V=64`。日志 `evidence/memory.log`
- [x] **选项覆盖，只查逐位一致。** `initial_state` 加 `output_final_state`（含 `dh0`）、`state_v_first=True`、`use_gate_in_kernel=True`、`chunk_size=32`、`T` 不被 chunk 整除、packed、GVA `H=2,HV=4`、FP16。任何一项不一致就停下查原因。日志 `evidence/options.log`
- [x] **计时矩阵。** BF16、8 组交错配对、每组 12 次：packed `V=256`（四条序列）、GVA `H=8,HV=16,V=256`、小形状 `B=4,T=2048,H=4,K=V=64`、FP16 dense `V=256`。dense `V=128/256` 已在阶段 0 完成。日志 `evidence/matrix.log`
- [x] 把三项结果写回本文件。若每层显存增量明显高于推算，或某个形状回退，先与你讨论再继续

1b. 生产实现

- [x] 读 #1703 和 #1453 的最新 diff，确认改动段不重叠
- [x] 改 `fla/ops/gated_delta_rule/chunk.py`，按"生产改动的形状"实现
- [x] 加 `test_chunk_disable_recompute`，先在未实现的 base 上确认它因参数不存在而失败，再在候选上通过
- [x] CP：在 2 张空闲卡上跑 `tests/cp/test_gdn.py`，对比开启前后的梯度。按决定 2 处理是否提交测试
- [x] `git diff --stat` 核对体量，超过 110 行时先缩减测试参数

1c. 生产路径验证与 PR

- [x] 默认路径不变：`disable_recompute=False` 时，候选与 `6511ea3e` 的输出和梯度 `torch.equal`，`fwd` 和 `fwdbwd` 配对耗时在噪声内
- [x] 通过公开参数重跑阶段 0 和 1a 的计时矩阵，确认与脚本替换得到的降幅一致
- [x] 跑 Verification 里的全部命令。`tests/ops/test_gdn.py` 全文件用后台进程加日志，之前约需 5 分钟以上
- [x] 写 `pr.md`：标题 `[Perf] Add disable_recompute to chunk GDN`，按仓库模板列出各形状绝对耗时、配对降幅、每层显存增量、未测平台
- [x] 更新 `plans/README.md`。推送到 fork 和开 PR 之前先问你

## Verification

```bash
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn.py
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdn_kernels.py
.venv/bin/python scripts/find_dependent_tests.py fla/ops/gated_delta_rule/chunk.py
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/ops/test_gdp.py tests/layers/test_gated_deltanet.py tests/models/test_modeling_gated_deltanet.py
FLA_CI_ENV=0 .venv/bin/python -m pytest -q tests/cp/test_gdn.py -k cp2
.venv/bin/python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base 6511ea3e
.venv/bin/pre-commit run --files fla/ops/gated_delta_rule/chunk.py tests/ops/test_gdn.py
```

通过标准：

- 默认 `disable_recompute=False` 时输出、梯度与 base `torch.equal`，`fwd` 耗时无变化
- 开启时梯度与默认路径 `torch.equal`。若出现非逐位一致，先查原因，不放宽为容差比较
- PR 描述写明每个形状的绝对耗时、配对降幅、每层显存增量、不受益的形状
- A800 数据只作为 SM80 证据。省掉的是整次 kernel 调用，其他架构预期同向，但不写成已测结论
- FlashQLA 路径（SM90 及以上，`K=V=128`）在本机无法运行，PR 描述里列为未测
- registry 基准只覆盖默认路径，用来证明默认行为没有回退。开启后的收益以配对计时为准

## 停止条件

- 阶段 0 降幅低于 6%
- 梯度无法做到逐位一致且原因不在本改动可控范围内
- `cp_context` 下开启后梯度与默认路径不一致，此时缩小范围为"CP 下开启时报错"，仍有问题则与你讨论
- 每层显存增量或某个形状的回退使"小模型训练"的定位站不住

## 昇腾 910B 测试

未运行。步骤、NPU 版计时脚本和通过标准见 [npu-test.md](npu-test.md)。
