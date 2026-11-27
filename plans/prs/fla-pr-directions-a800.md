# 用 8 × A800 参与 Flash Linear Attention

**首选方向是 KDA 上下文并行（CP）的跨 rank 正确性。** 本机可以在同一套软件栈上构造单卡参考，先用两卡定位序列切分问题，再用八卡检验长上下文。已有合并先例直接体现了这种资源的价值。PR #1824 在 8 × A100 上发现，65,536-token 测试的单卡参考本身就会耗尽 40 GB 显存。作者分段计算参考并校验跨段反传，保留了原有算子、输入和误差阈值。[已合并 PR #1824](https://github.com/fla-org/flash-linear-attention/pull/1824)

调研快照为 2026-11-25，本地 `main` 为 `cc7018a4`。通过 GitHub CLI 查看 11 月 1–25 日合并的 77 个 PR、当时返回的前 100 个开放 PR 及开放 issue 列表，精读与本机相关的案例。以下趋势是据此做的选题判断，开放状态以链接当前页面为准。[合并记录](https://github.com/fla-org/flash-linear-attention/pulls?q=is%3Apr+is%3Amerged+merged%3A2026-11-01..2026-11-25) · [开放 PR](https://github.com/fla-org/flash-linear-attention/pulls)

## 从哪一层读代码

- `fla/models/` 把模型配置、训练与生成接入 Transformers。`fla/layers/` 组合投影、短卷积、状态缓存和 attention 算子。遇到“整段 prefill 与分段 decode 不一致”，先在 layer 入口固定路由、比较每步缓存，再判断该改哪一层。[MoM 实现](../../fla/layers/mom.py) · [缓存问题 #1611](https://github.com/fla-org/flash-linear-attention/issues/1611)
- `fla/ops/<op>/` 是算子的公共入口及实现。以 KDA 为例，长序列训练关注 chunk 路径、保存的中间状态与反向梯度，逐 token 解码关注 recurrent 状态更新。对照参考时要记录实际走的后端，以及输入是否按 `cu_seqlens` 打包。[KDA 算子](../../fla/ops/kda/) · [KDA 测试](../../tests/ops/test_kda.py)
- `fla/ops/common/` 和 `fla/ops/cp/` 的改动会传到多个算子。CP 将同一序列切给不同 rank，切点若落在序列内部，要传递状态及其梯度；若恰逢两条独立序列的边界，则必须隔离。单测同时比较拼回的输出和梯度才覆盖这个契约。[CP 设计](../../fla/ops/cp/README.md) · [测试矩阵](../../tests/cp/README.md)

## 社区正在合并什么，还有哪些人在做

| 方向 | 近期合并证据 | 开放工作与选题含义 |
| --- | --- | --- |
| 状态、梯度、变长正确性 | HGRN 补齐首末状态反传 [#1630](https://github.com/fla-org/flash-linear-attention/pull/1630)，KDA 加入共线 key 与 fused varlen 训练测试 [#1828](https://github.com/fla-org/flash-linear-attention/pull/1828) | Delta Rule 的 `dbeta` 已有 [#1838](https://github.com/fla-org/flash-linear-attention/pull/1838)，零长度序列索引已有 [#1837](https://github.com/fla-org/flash-linear-attention/pull/1837)。独立参考和漏掉的梯度分量值得查，先搜具体符号。 |
| 多卡 CP、长序列 | CUDA 长上下文参考 [#1824](https://github.com/fla-org/flash-linear-attention/pull/1824)，Ascend KDA CP [#1558](https://github.com/fla-org/flash-linear-attention/pull/1558) | `K > 64` 的 summary kernel 已有 [#1752](https://github.com/fla-org/flash-linear-attention/pull/1752)，zigzag 保存状态已有 [#1680](https://github.com/fla-org/flash-linear-attention/pull/1680)。可贡献八卡边界复现和交叉验证。 |
| 架构相关的性能与资源上限 | SM8x CyFA backward 降低寄存器溢出 [#1592](https://github.com/fla-org/flash-linear-attention/pull/1592)，LayerNorm 合并 autotune 分桶修复 [#1800](https://github.com/fla-org/flash-linear-attention/pull/1800) | L2Norm 动态行数已有 [#1795](https://github.com/fla-org/flash-linear-attention/pull/1795)，Hopper KDA 调参已有 [#1714](https://github.com/fla-org/flash-linear-attention/pull/1714)。SM80 适合定位资源和编译问题，通用性能结论还要补目标硬件数据。 |
| 新算子与后端 | Triadic KDA 先合入参考实现 [#1746](https://github.com/fla-org/flash-linear-attention/pull/1746)，Ascend Parallax 后端 [#1709](https://github.com/fla-org/flash-linear-attention/pull/1709) | Triadic KDA chunk 已有 [#1751](https://github.com/fla-org/flash-linear-attention/pull/1751)，B300 QLA 有专门的 [RFC #1844](https://github.com/fla-org/flash-linear-attention/issues/1844)。适合有算法合作方或目标设备时参与。 |

## 三个具体切入点

1. **首选，验证 packed KDA 的八卡 zigzag 边界。** 当前 `tests/cp/test_kda.py` 的 zigzag 覆盖 2、4 rank，8 rank 用例是 contiguous 单序列。先跑已有对照，再固定总长 8192，在八卡 zigzag 的 16 个半分片中，把两条独立序列的分界从 512 移到 511、513。保持输入种子和 dtype，分别与单卡 recurrent 参考比较。记录首个错误 token、所在 rank，以及输出和每项梯度的误差。若失败，缩到能触发的最少 rank，提交修复与单个回归用例；若都通过，且没有发现八卡特有的状态传播分支，这组数据更适合用于现有 PR 的验证，单纯添加昂贵 CI case 的价值有限。[测试参数与断言](../../tests/cp/test_kda.py) · [相邻改动 #1667](https://github.com/fla-org/flash-linear-attention/pull/1667)
2. **单卡备选，验证真实的短续写缓存错误。** MoM issue #1611 给出了固定路由、`conv_size=4` 的对照。比较整段 5 token 与 `2+2+1`，检查第 5 个输出、q/k/v 卷积缓存和 recurrent state。该报告基于 10 月提交，且评论中已有贡献者认领。适合在当前 main 独立复核，并优先给原作者补充多 memory 路由的边界数据。[issue #1611](https://github.com/fla-org/flash-linear-attention/issues/1611) · [现有 layer 测试](../../tests/layers/test_mom.py)
3. **性能备选，给 CP summary 已有实现补 Ampere 对照。** #1752 作者已经给出 A100 上 `K=256` 的前反向和编译耗时。A800 实验最有价值的问题是单 kernel 收益是否传递到多卡 CP：固定总 token、head 数与布局，对照 `K=64/128/256`，分别记录 summary 耗时、首次 autotune 时间和完整 CP step 耗时。若端到端变化小，再看各 rank 是否等待通信。独立发现新的瓶颈后再定新 PR 范围。[原始报告 #1683](https://github.com/fla-org/flash-linear-attention/issues/1683) · [在审实现 #1752](https://github.com/fla-org/flash-linear-attention/pull/1752) · [CP benchmarks](../../benchmarks/cp/)

## 本机验证顺序

`nvidia-smi` 显示 8 × NVIDIA A800-SXM4-80GB，每卡 81920 MiB，SM80，驱动 580.65.06，卡间拓扑均为 NV8。本地 `.venv` 为 PyTorch `2.10.0+cu130`、CUDA runtime `13.0`、Triton `3.6.0`。硬件适合独立验证 Ampere 数值、显存与单机多卡通信。仓库要求 NVIDIA 性能结论以 SM90+ 数据中心卡为依据，A800 数据作为补充；涉及精度或算法替换先走 RFC。[NVIDIA 性能指南](../../.agents/skills/fla-nvidia-performance/SKILL.md) · [仓库约定](../../AGENTS.md)

在仓库根目录按顺序运行。先确认 GPU 空闲。CP 测试会自行启动 ranks，每台机器只启动一个 pytest 进程。[CP 测试说明](../../tests/cp/README.md)

```bash
source .venv/bin/activate
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
git rev-parse HEAD
CUDA_VISIBLE_DEVICES=0,1 FLA_CI_ENV=0 python -m pytest -qs tests/cp/test_kda.py -k cp2-zigzag-sequence-cut
CUDA_VISIBLE_DEVICES=0,1,2,3 FLA_CI_ENV=0 python -m pytest -qs tests/cp/test_kda.py -k cp4-zigzag-sequence-cut
FLA_CI_ENV=0 python -m pytest -qs tests/cp/test_kda.py -k cp8-single-sequence
```

上面三条是仓库**已有对照**。八卡 packed zigzag 是待验证实验。KDA CP 的 `dg` 和 `db` 当前仅告警，保存逐项误差，尤其关注切点附近的偏差；即便 pytest 成功也应单列这些诊断。性能对照另外固定 base/head SHA、GPU、软件版本、输入和 dispatch，预热后分开记录 fwd/bwd、dense/varlen 与显存峰值。只有测了完整训练路径，才报告训练吞吐。[贡献指南](../../CONTRIBUTING.md#benchmarking)

开工当天用算子名、函数名和问题关键词重查 open PR 与 issue 并阅读认领评论。已有相同修复时，独立验证结果可直接服务原 PR。能指出哪个用户输入失败、哪一项断言捕获失败、修复后哪些调用方通过，才有足够材料确定自己的首个修复 PR。
