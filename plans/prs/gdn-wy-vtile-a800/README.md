# GDN 训练中 WY 重建的 V 分块实验

## A800 实测结果

2026-11-25，`8973fdd7`，A800-SXM4-80GB，PyTorch 2.10.0+cu130，Triton 3.6.0。两组负载均保留 `BV=64`。下表为单 kernel 的 GPU 事件计时，单位 µs，每格列出六轮中位数和四分位距，每轮 30 次调用。

| V | BV=32 | BV=64 | BV=128 |
| --- | --- | --- | --- |
| 128 | 206.10（0.30） | 199.29（0.06） | 202.34（0.32） |
| 256 | 297.11（0.38） | 283.05（0.16） | 283.89（0.32） |

`V=256` 的七轮稳态训练时延为 6.128–6.136 ms，前向为 1.748–1.762 ms。trace 中前反向各调用 WY 一次，合计 545.0 µs，占 kernel 时间的 8.97%。`V=128` 对应为 363.5 µs 和 8.86%。初始两轮 `V=256` 训练计时为 6.77 和 6.15 ms，另采七轮判断稳态波动。

目标形状三个 BV 按顺序使用 173、205、243 个寄存器/线程，编译器报告 spill 均为 0。宽度 128 比 64 少两次 V 循环，autotune 也从 2 warps 改为 4 warps。结果支持保留现有配置，但不足以把耗时差归因于单个硬件指标。NCU 读取 DRAM 计数器返回 `ERR_NVGPUCTRPERM`。[计数器日志](ncu.log)

正确性门槛为 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn_kernels.py -k recompute_w_u_fwd`，8 passed。seed 42 的两个主负载、三个 tile 的 `w/u` 均通过原有 0.005 容差。第 4 步以更快候选为前提，本轮在微基准结束后归档。目标负载 trace 中累计耗时最高的是 `chunk_bwd_kernel_dqkwg`，约 1.23 ms，占 20.2%。

从仓库根目录分别执行 `bash plans/prs/gdn-wy-vtile-a800/run.sh baseline`、`trace`、`tiles` 三个模式。原始 trace 在 `profile/gdn-wy-vtile/v{128,256}.json`。[基线](baseline.log) · [稳态计时与 trace 摘要](trace.log) · [逐轮结果与 autotune 配置](tiles.log) · [测试输出](tests.log)

## 实验问题

在 A800 上测量 `recompute_w_u_fwd_kernel` 占一次 `chunk_gated_delta_rule` 前向加反向的时间比例，再判断调整 `BV` 能否降低训练算子耗时。当前 Triton 路径在前向生成 `w/u`，反向再重建一次。基线 `BK=BV=64`，第一轮固定 `BK=64`，比较 `BV=32/64/128`。[kernel 与封装](../../../fla/ops/gated_delta_rule/wy_fast.py) · [前向调用](../../../fla/ops/gated_delta_rule/chunk_fwd.py) · [反向调用](../../../fla/ops/gated_delta_rule/chunk.py)

## 负载、参考与文件

主实验使用单张空闲 A800、BF16、`B=2,T=8192,H=HV=16,K=128`。`V=256` 是目标，`V=128` 是对照。固定输入、上游梯度、门控、chunk size 和 dispatch。记录实际选中的后端、autotune 配置、GPU 与软件版本。实验脚本、命令、各轮结果和结论归入本目录，大型 trace 存到 `profile/gdn-wy-vtile/`。

复用 `tests/ops/test_gdn_kernels.py` 的 dense 和 varlen Torch 参考，检查 `w/u`，用 `tests/ops/test_gdn.py` 验证公开算子的输出、状态和支持的梯度。通过直接调用现有 kernel 传入候选 `BV` 做初筛。只有值得集成的候选才需要修改 `fla/ops/gated_delta_rule/wy_fast.py`。若现有参数矩阵缺少非对称 `K/V` 或 V 尾块，再扩展现有用例。[kernel 测试](../../../tests/ops/test_gdn_kernels.py)
## 步骤与判定标准

- [x] **冻结基线和正确性门槛。** 记录 base `8973fdd7`，运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn_kernels.py -k recompute_w_u_fwd` 和两组主负载的参考对照。保存输入 seed、容差和实际运行的 case。先测两次相同基线，估计同卡时延波动。
- [x] **测量 WY 占比。** 对两个主负载预热至 JIT 和 autotune 完成，用同步 GPU 计时分别取得 `fwd`、`fwdbwd` 时延。另采一份稳态 CUDA trace，按训练迭代统计 WY 调用次数以及累计 GPU 时间占比 `p`。用 `p × (1 − 1/1.2)` 估算 WY 加速 1.2 倍时完整算子的耗时降幅。若低于实测噪声，归档基线和 trace，结束本轮。
- [x] **比较三种 V 分块。** 使用同一输入和相同 autotune 搜索空间，分别预热每个 `BV`。每个候选先对齐 `w/u`，再至少做五轮交错计时，报告各轮时延、中位数和四分位距。对最快配置和 `BV=64` 采样寄存器数、spill 和 DRAM 吞吐，说明 V 循环次数变化如何影响耗时。若性能差落在噪声内，保留基线并记录原因。
- [x] **验证完整训练路径。** 将胜出配置接入实际 GDN 前反向，在同卡上复测两组主负载和 packed varlen。晋级要求 `V=256` 的配对 `fwdbwd` 时间收益超过基线噪声，`V=128` 对照及受影响的其他负载回退处于噪声内。运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py`、WY 测试和共享调用方的相关测试，用 `python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base 8973fdd7` 补充仓库标准负载。归档所有结果；若准备提交通用 NVIDIA 性能 PR，按仓库指南补 SM90+ 同卡前后对照。[性能指南](../../../.agents/skills/fla-nvidia-performance/SKILL.md)
