# KDA 八卡 zigzag 序列边界实验与 PR 流程

**决定是否提 PR 的依据**是能否找到可复现的跨序列状态泄漏、梯度错误，或能明确说明维护价值的测试缺口。已有八卡长序列参考和两卡、四卡 zigzag 测试可复用。调研和实验基线为 `cc7018a4`。[选题背景](../fla-pr-directions-a800.md) · [现有分布式测试](../../../tests/cp/test_kda.py)

## 实测结果与决定（2026-11-25）

**归档这组选题。** 在八张 A800-SXM4-80GB 上，三种切点 × 两种布局的六组测试均通过，耗时 133.33 秒。另有两个已有 CP 对照和三个同输入单卡 chunk 对照通过，共 11 个用例。八卡六个用例的 `o/dq/dk/dv` 全局相对 RMS 分别为 `0.006654/0.006653/0.007778/0.006302`。这些数值采用仓库原有 `get_err_ratio` 的计算精度。所有新增对照的参考及实测张量均有限。

| 切点 | 八卡 zigzag `dg` | 八卡 contiguous `dg` | 单卡 chunk `dg` |
| --- | --- | --- | --- |
| 511 | 0.008476 | 0.008527 | 0.008527 |
| 512 | 0.008476 | 0.008578 | 0.008476 |
| 513 | 0.008527 | 0.008476 | 0.008578 |

`dg` 在全部九组对照中均超过 `0.008` 诊断线。切点前后 16 个 token 的 `dg` 最大绝对误差在八卡 zigzag 为 `0.00244–0.00345`，单卡为 `0.00244–0.00339`。两种布局与单卡的偏差量级相近。当前种子、门控和形状下，数据不足以支持针对八卡 zigzag 的修复 PR。

运行环境为 PyTorch `2.10.0+cu130`、Triton `3.6.0`、驱动 `580.65.06`、NCCL、`FLA_CI_ENV=0` 及 `OMP_NUM_THREADS=1`。两卡及四卡首次运行因 `Python.h` 缺失而在编译阶段退出，向 `CPATH` 加入本机现有的 `$HOME/.local/python312-dev/usr/include`、其 `python3.12` 和 `x86_64-linux-gnu/python3.12` 子目录后，对照分别用时 328.93 秒和 18.92 秒。八卡运行使用 `CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,1,0`，单卡使用 GPU 2。归档的复现脚本分别为 [offsets.py](evidence/offsets.py) 和 [single.py](evidence/single.py)。[八卡日志](evidence/eight.log) 与 [单卡日志](evidence/single.log) 保留每组六项误差、切点窗口和非有限值计数。

## 实验问题与对照

固定 `T=8192`，八卡 zigzag 将全局 token 顺序切成 16 段，每段 512 token。rank `r` 持有第 `r` 和第 `15-r` 段。全局位置 512 恰好从 rank 0 转到 rank 1。两条独立序列在此处相接时，新序列应从零状态开始；切点向左或向右移一个 token，同一序列就需要把状态传过 rank 边界。一个 token 的移动可同时检验状态重置和跨卡延续。[布局实现](../../../fla/ops/cp/context.py)

| 用例 | rank / layout | 两条序列的长度 | 检查意图 |
| --- | --- | --- | --- |
| Z0 | 8 / zigzag | `[512,7680]` | rank 1 的第一个 token 开启新序列 |
| Z− | 8 / zigzag | `[511,7681]` | 第二条序列的首个 token 在 rank 0，下一个在 rank 1 |
| Z+ | 8 / zigzag | `[513,7679]` | 第一条序列的最后一个 token 在 rank 1，紧接着应重置状态 |
| C | 8 / contiguous | 与对应 Z 相同 | 保持输入和序列边界，只改变布局，区分通用 packed 问题与 zigzag 路径问题 |

首轮沿用 `test_chunk` 的输入生成器，rank 0 使用种子 42 并广播。固定 `H=12`、`D=128`、BF16、in-kernel Q/K L2Norm、in-kernel gate、`safe_gate=True`、`lower_bound=-5.0`、默认状态布局与重算。所有组使用相同的上游梯度 `do`。每个用例分别按自己的 `cu_seqlens` 构造参考；改变逻辑切点会改变正确输出。

## 在 A800 上执行

1. 在仓库根目录启用 `.venv`，记录 base SHA、GPU 型号和可见卡号、驱动、PyTorch/CUDA/Triton 版本。用 `nvidia-smi` 确认有足够的空闲设备。已有用例先做两卡冒烟，随后做四卡对照。pytest fixture 自行启动进程，每台主机只启动一次 pytest。[运行说明](../../../tests/cp/README.md)

   ```bash
   source .venv/bin/activate
   export FLA_CI_ENV=0
   git rev-parse HEAD
   nvidia-smi
   CUDA_VISIBLE_DEVICES=0,1 python -m pytest -qs 'tests/cp/test_kda.py::test_chunk[cp2-zigzag-sequence-cut]'
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest -qs 'tests/cp/test_kda.py::test_chunk[cp4-zigzag-boundary-aligned]'
   ```

2. 方案通过后，将 Z0、Z−、Z+ 及对应 C 用例临时加入 `test_chunk` 的参数表，分别命名为 `cp8-zigzag-offset-{511,512,513}` 与 `cp8-contiguous-offset-{511,512,513}`。沿用 `_run_chunk`、`shard_tensor`、`gather_global` 和 `_reference`。先 `--collect-only` 确认 6 个 node ID，再用同一筛选执行。若有失败，单独重跑失败 ID 两次，保存三次结果。

   ```bash
   python -m pytest --collect-only -q tests/cp/test_kda.py -k 'cp8 and offset'
   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m pytest -qs tests/cp/test_kda.py -k 'cp8 and offset' --tb=short
   ```

3. rank 0 对两条序列分别计算单卡 recurrent 参考及反向，再拼回全局顺序。先检查参考与实测的 `o/dq/dk/dv/dg/db` 均有限，再使用现有 `assert_close`。`o/dq/dk/dv` 沿用 `ratio=8e-3` 的严格断言，`dg/db` 沿用诊断告警。另取切点前后各 8 个 token 记录逐项最大绝对误差，避免单个边界错误被全局 RMS 稀释。[断言实现](../../../fla/utils/testing.py)

每次运行留存 SHA、命令、case ID、实际 rank 数、后端、耗时、跳过原因、各张量相对 RMS、边界窗口误差与非有限值计数。失败时记录首个异常 token 的全局位置、所属序列、半分片与 rank。原始日志保存在仓库外。

## 结果如何决定下一步

- **硬断言失败或出现非有限值。** 先用全局 token 编号做 `shard → gather` 往返校验。随后固定第二条序列，单独扰动第一条序列的输入，比较第二条序列的输出是否改变。反向只对第二条序列施加 `do`，检查第一条的输入梯度是否为零。这能把跨序列泄漏与 BF16 的正常数值差异区分开。比较对应 C 组，沿 CP 元数据、前向初始状态、反向状态梯度依次定位。用四卡、两卡缩小复现时按半分片长度重新选取切点。
- **只有 `dg/db` 告警。** 在同输入上增加单卡 chunk 对 recurrent 的误差作参照，检查偏差是否集中在切点、是否跟随 rank 或布局移动。保留现有阈值和原始数值，等独立证据能定位 CP 特有错误后再提出修复。
- **全部通过。** 记录实际覆盖的路径和八卡耗时。若两卡即可覆盖相同逻辑，优先将验证结果用于相关在审 PR。只有能说明现有测试漏掉了哪一种状态传播行为、并能控制 CI 成本时，才考虑独立测试 PR。

## 从复现到合并

1. **查重与归属。** 开工当天在 `fla-org/flash-linear-attention` 的 issues 和 PR 中搜索 `KDA CP zigzag`、`sequence boundaries` 和最终定位的函数名，核对认领评论及最新 head。尤其检查 zigzag 状态保存、intra-card 边界和 CP summaries 三个在审方向。重合时向原 PR 提供准确的 revision、命令和结论。[#1680](https://github.com/fla-org/flash-linear-attention/pull/1680) · [#1667](https://github.com/fla-org/flash-linear-attention/pull/1667) · [#1752](https://github.com/fla-org/flash-linear-attention/pull/1752)
2. **冻结复现。** 记录最新上游 base SHA，在该版本运行相同测试，保留失败日志和最小输入。一个 case 应能区分错误行为与正确行为。公共 API、计算精度或算法需要改变时先按仓库规则讨论 RFC。[仓库约定](../../../AGENTS.md)
3. **实现与回归。** 获准建分支后，从已验证 base 创建按问题命名的主题分支。把最小失败用例加入已有参数矩阵，修正根因。在 base 和候选代码上执行同一回归用例，证明先失败、后通过。共享 CP helper 有改动时，用 `rg` 核对所有调用方，并运行 `python scripts/find_dependent_tests.py <changed-path>` 获取受影响测试。
4. **验证候选版本。** 先跑复现、成对布局对照和 `FLA_CI_ENV=0 python -m pytest -q tests/cp/test_kda.py`，再跑涉及的算子与其他 CP 调用方。对改动文件执行 `pre-commit run --files <changed-paths>`。改到 kernel 时，提供同设备、同输入、同后端的 base/head 耗时，并按仓库要求补目标平台证据。纯测试改动报告测试耗时和所需卡数即可。[贡献指南](../../../CONTRIBUTING.md)
5. **准备和发布。** 用 `git diff --check` 和完整 diff 核对范围。向用户展示文件列表、diff 和建议 commit message，经确认后提交。获得 push 与开 PR 的授权后推送主题分支，按当前模板填写触发条件、影响范围、根因、复现方式、验证 SHA、实际结果及平台限制。有剩余验证项时开 draft，将本地计划与实验日志留在提交之外。[PR 模板](../../../.github/pull_request_template.md)
6. **CI、评审、合并后核验。** 核对上游 PR 的 base/head 和 CI 命中的测试。CI 失败时在相同 revision 复现，逐项回应 review，修订后重跑受影响用例并更新证据。维护者合并后记录 PR 链接与 merge SHA，确认最小回归用例已进入上游，再回填关联 issue 的验证结论。对外发布评论和变更均按用户授权执行。
