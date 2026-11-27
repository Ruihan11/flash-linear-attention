# GDN `chunk_bwd_dqkwg` 的 BK/BV 性能扫描

## Decision on 399504c6 — killed

On A800 with BF16 and K=128, the 10-config tuner selected BV=64, 8 warps and 3 stages for dense, packed and grouped-head V=256 calls. Across eight alternating pairs the respective median full forward/backward time reductions were 4.17%, 3.22% and 4.30%. Dense V=128 selected the original config and changed by less than 0.01%. Five input gradient relative RMS differences were zero in each sample. In the paired log, the inherited `base/bv32` heading refers to baseline versus automatic selection. The preceding `selected` line identifies the actual configuration. [Paired samples](evidence/autoselect.log)

With fresh Triton directories, separate processes and FLA config lookup disabled, the first dqkwg call took 20.97 and 20.93 seconds with the nine original configs, and 21.95 and 22.06 seconds with ten. The repeat used a single in-memory tuning entry. The average additional 1.06 seconds per measured cold shape would require roughly 4,150 dense training calls at 0.255 ms saved per call to amortize. The approved first-call regression gate ended promotion. [Cold-search samples](evidence/cold-search.log)

On A800 the baseline kernel suite passed 5 cases, the explicit candidate passed 8 GVA tail cases with both 16-bit dtypes and optional inputs, and four representative shared-caller reference tests passed with the prototype tuner. The baseline full public GDN suite passed 104 cases with 29 skipped. The experiment reached its stop condition before production integration and registry benchmarking. [Tail comparisons](evidence/correctness-tail.log) · [Public suite](evidence/full-gdn-current.log)

## Context

最新本地 `main` 为 `f8bfd49f`。此前 A800 trace 显示 GDN 训练中 `chunk_bwd_kernel_dqkwg` 是主要热点，在 `B=2,T=8192,H=HV=16,K=128,V=256,BF16` 负载上约占 GPU kernel 时间的 20%。当前 wrapper 在 Ampere 上根据 dtype 和 shared-memory 条件固定 `BK/BV`，kernel 内部沿 V tile 重复加载 `v/do/h/dh`，并累加 `dq/dk/ds`。[实现](../../../fla/ops/common/chunk_o.py)

目标是判断分块配置是否存在可重复的性能收益，再决定是否形成独立性能 PR。实验结果、失败候选和原始摘要全部保存在本目录。

## Architecture matrix

| Platform | Available here | Route | Tile policy | Evidence status |
| --- | --- | --- | --- | --- |
| Ampere A800 SM80 | Yes, 8 × A800 | shared NVIDIA Triton | `CONST_TILING=128` for 16-bit tensors, otherwise 64. `BK/BV` are powers of two capped by this value. | Will measure locally |
| Hopper | No | shared NVIDIA Triton | `CONST_TILING=128`, `NUM_WARPS=[2,4]`, plus a Triton-version guard for gated backward. | Unverified |
| Ada | No | shared NVIDIA Triton | `CONST_TILING=64`. | Unverified |
| Blackwell | No | shared NVIDIA Triton | Shared kernel with Blackwell-specific warp race guard for `bwd_dv_local`, and Triton-version-dependent configuration. | Unverified |
| Ascend | No | separate `TritonAscendBackend` registration | `_get_bc/_get_bk/_get_bv` select tiles from an explicit UB budget and safety margin. | Out of scope for this experiment |

The local environment is PyTorch `2.10.0+cu130`, CUDA `13.0`, Triton `3.6.0`, driver `580.65.06`, and eight NVIDIA A800-SXM4-80GB GPUs with compute capability 8.0. The shared wrapper and Ascend backend do not use the same tile-selection mechanism. Any result below is an Ampere result and cannot justify a Hopper, Blackwell, Ada, or Ascend claim.


先回答“这个优化能否通用”再做配置扫描。该入口有两条实现路线，必须分开判断：NVIDIA/CPU 等平台走 `fla/ops/common/chunk_o.py` 的共享 Triton kernel，Ascend 走独立的 `fla/ops/common/backends/triton_ascend/chunk_o.py` 注册实现。共享 wrapper 只用 `check_shared_mem` 区分 Hopper、Ampere、Ada 和 fallback，Blackwell 还受 `NUM_WARPS` 与 Triton 版本约束。因此 A800 的候选结果只能作为 Ampere 假设，不能直接推广到 Hopper 或 Blackwell。

实验顺序如下：

1. 先做代码级通用性审查，列出候选影响的 backend、架构分支、dtype、GVA 和 varlen 路径。
2. 用 A800 做共享 Triton kernel 的初筛，观察收益是否来自算法结构，还是来自 Ampere 的 shared-memory/register 预算。
3. 若候选涉及共享 wrapper 的默认配置，必须在 Hopper 和 Blackwell 上有可访问设备或 CI 证据后再考虑通用改动。否则将改动限定为明确的 `IS_NVIDIA_*` 架构分支，并保留其他架构配置。
4. Ascend 不复用 NVIDIA 的 `BK/BV` 结论。若要声称 Ascend 受益，需要在 Ascend backend 的 `_get_bk/_get_bv/_get_bc` 和 UB 预算下独立设计、独立 trace 和独立 PR。当前没有 Ascend 设备，因此本实验不覆盖 Ascend，也不修改 Ascend 文件。

NCU 不是第一步的必需工具。同步 GPU event 已足以判断候选是否有稳定时延收益。只有在 A800 候选超过噪声、需要解释收益来源，或要区分寄存器、spill、occupancy 与 DRAM 瓶颈时才使用 NCU。若 GPU performance counters 没有权限，记录 `ERR_NVGPUCTRPERM`，使用 Triton 编译资源和 CUDA trace，不把 NCU 指标写成结论。Hopper/Blackwell 的 NCU 采集需要在对应硬件上单独进行，不能由 A800 指标替代。

## Files to modify

实验阶段预计只新增本目录的脚本和日志。若候选晋级，才修改以下生产文件和测试文件。

- `fla/ops/common/chunk_o.py`，仅在共享 NVIDIA Triton 路径且已有架构证据支持时修改
- `fla/ops/common/backends/triton_ascend/chunk_o.py`，本轮不修改。Ascend 若形成候选，另立独立实验和 PR
- `tests/ops/test_gdn_kernels.py`，仅当现有参数矩阵无法覆盖胜出配置的边界
- `tests/ops/test_gdn.py`，用于完整公开算子回归

## Reuse

- `chunk_bwd_dqkwg` 及其 Ampere/Hopper/Ada/Blackwell 分支逻辑复用 `fla/ops/common/chunk_o.py`。先记录 `check_shared_mem`、`IS_NVIDIA_HOPPER`、`IS_NVIDIA_BLACKWELL` 和 Triton 版本对配置的影响。
- Ascend 的 `_get_bk/_get_bv/_get_bc`、UB budget 和 `TritonAscendBackend` 注册复用 `fla/ops/common/backends/triton_ascend/chunk_o.py` 作为通用性边界参考，不把 NVIDIA tile 直接移植过去。
- `chunk_bwd_dqkwg_ref` 和 `test_chunk_bwd_dqkwg` 复用 `tests/ops/test_gdn_kernels.py` 的逐 kernel Torch reference。现有 cases 覆盖 `H=HV`、GVA、gate、`w`、`dv`、BF16 和 FP16。
- `benchmarks/ops/registry.py` 的 `chunk_gdn` 入口用于标准 workload 对照。
- `tests/ops/test_gdn.py` 用于公开 GDN 输出、state 和支持梯度的验证。
- A800 结果按 `fla-optimization-loop` 和 `fla-nvidia-performance` 记录，性能结论标注为 Ampere 初筛证据。若没有 Hopper/Blackwell 设备或 CI 证据，不宣称跨 NVIDIA 架构收益。Ascend 结果按 `fla-ascend-performance` 单独处理。

## Steps

- [x] **先做通用性矩阵。** 在代码和可用环境中记录 Hopper、Blackwell、Ampere 的配置分支，检查共享 wrapper 是否在每种架构采用相同 kernel 与候选参数。记录 Ascend 是独立注册实现。没有对应硬件时标记为未验证，不用 A800 结果代替。
- [x] **冻结 baseline。** 记录 `f8bfd49f`、A800 型号、驱动、PyTorch/CUDA/Triton、GPU 编号、seed 和当前实际 `BK/BV`。运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg`，确认 5 个参数 case 全部执行并通过。
- [x] **建立 A800 热点基线。** 固定 `B=2,T=8192,H=HV=16,K=128,V=128/256,BF16,BT=64`，预热 JIT/autotune 后交错测量 `chunk_bwd_kernel_dqkwg` 和完整 `chunk_gated_delta_rule` 的 fwd/fwdbwd。每个形状至少 7 轮，记录中位数、四分位距、实际 autotune 配置和编译资源。
- [x] **扫描 A800 的 BK/BV。** 先测 `(128,32)`, `(128,64)`, `(128,128)`，再测 `(64,64)`, `(64,128)`。每个候选先比较 `dq/dk/dw/dg` 与 reference，再做至少 6 轮交错计时。记录 registers/thread、spill、launch grid、完整算子时延和相对 baseline 的变化。
- [x] **决定是否需要 NCU。** 若候选没有超过 event timing 噪声，停止 profiler，不用 NCU。若候选稳定胜出且需要解释原因，再在 A800 对 baseline/候选采集 NCU。若未来要声称 Hopper/Blackwell 收益，在对应 GPU 上重新做 event timing 和 NCU，不能复用 A800 计数器。若 GPU counters 无权限，记录错误并使用编译资源与 CUDA trace。
- [x] **扩大正确性覆盖。** 对胜出候选运行现有 `test_chunk_bwd_dqkwg` 全集，补查 `H=2,HV=4` GVA、`use_g/use_w` 开关、FP16、`V=128/256` 和非整除尾 tile。然后运行 `FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py` 以及依赖扫描得到的相关测试。若候选进入共享 wrapper，必须按架构分别检查 Hopper/Blackwell 配置；Ascend 本轮不纳入晋级。
- [x] **决定实现范围。** 只有 A800 稳定收益超过噪声且完整训练路径不回退时，才继续讨论实现。若收益只存在 A800，配置改动必须明确限定 Ampere 并保留 Hopper/Blackwell/Ada fallback。若收益可由共享算法解释但缺少其他架构证据，先不提交通用 PR。Ascend 若要优化，另建使用 NPU trace 和 UB 预算的实验目录。
- [x] **准备 PR。** 若晋级，标题根据实际范围命名为 `[Perf] Tune GDN chunk backward tiling on Ampere` 或经多架构验证后再使用通用标题，说明影响的形状、dtype、backend、架构证据和未验证平台。保留 raw profile 在 `profile/gdn-bwd-dqkwg-opt/`，不加入 Git。


## Results

### Architecture and baseline

Only Ampere was available locally. No Hopper, Ada, Blackwell, or Ascend timing was collected. The shared NVIDIA wrapper uses architecture-specific shared-memory caps, while Ascend dispatches to a separate UB-budgeted implementation. The conclusions below are Ampere-only.

The kernel correctness gate passed 5 cases in 3.11 s. The A800 environment was PyTorch `2.10.0+cu130`, CUDA `13.0`, Triton `3.6.0`, driver `580.65.06`, GPU 2, BF16, seed 42. After warmup, current wrapper timings were:

| V | `chunk_bwd_kernel_dqkwg` | full GDN fwd | full GDN fwdbwd |
| --- | ---: | ---: | ---: |
| 128 | 0.739 ms* | 1.212 ms | 4.178 ms |
| 256 | 1.187 ms | 1.751 ms | 6.145 ms |

`*` The V=128 kernel samples were still settling from compilation/autotune and ranged from 0.692 to 0.803 ms. The full operator measurements were stable. The detailed baseline is in [baseline.log](evidence/baseline.log).

### Candidate scan

Each candidate used the same `B=2,T=8192,H=HV=16,K=128,BT=64`, BF16 inputs, seed 42, and direct kernel launch. The six samples below are kernel event times in ms. The existing Torch reference remained within the existing relative error budget for `dq`, `dk`, `dw`, and `dg` for every candidate.

| V | BK/BV | samples | median | result |
| --- | --- | --- | ---: | --- |
| 128 | 128/32 | 0.683, 0.679, 0.679, 0.679, 0.679, 0.679 | 0.679 | fastest in this microbenchmark |
| 128 | 128/64 | 0.732, 0.728, 0.728, 0.726, 0.727, 0.728 | 0.728 | slower |
| 128 | 128/128 | 0.693, 0.691, 0.691, 0.691, 0.691, 0.691 | 0.691 | close to baseline |
| 128 | 64/64 | 0.893, 0.891, 0.891, 0.891, 0.893, 0.883 | 0.891 | slower |
| 128 | 64/128 | 0.766, 0.765, 0.765, 0.765, 0.765, 0.765 | 0.765 | slower |
| 256 | 128/32 | 1.027, 1.025, 1.025, 1.025, 1.025, 1.023 | 1.025 | fastest sequential median |
| 256 | 128/64 | 0.986, 0.982, 1.039, 1.074, 1.074, 1.075 | 1.056 | lower median, variable |
| 256 | 128/128 | 1.284, 1.276, 1.190, 1.187, 1.185, 1.181 | 1.188 | matches baseline after warmup |
| 256 | 64/64 | 1.405, 1.403, 1.403, 1.403, 1.402, 1.402 | 1.403 | slower |
| 256 | 64/128 | 2.043, 2.039, 2.039, 2.039, 2.038, 2.039 | 2.039 | much slower |

### Paired full-operator verification

Two independent processes timed dense BF16 forward plus backward on GPU 2. Each ran 12 paired rounds per width, with 12 calls per variant and alternating A/B order. Each run reused identical inputs within each pair (seed 42). The instrumented run confirmed `(BK,BV)=(128,128)` for baseline and `(128,32)` for candidate with identical launch grids. The two Python wrappers differed only in the value tile. The paired reduction is the median of `100*(base_ms-candidate_ms)/base_ms`.

| V | Run 1 base → candidate (ms) | Paired reduction | Run 2 base → candidate (ms) | Paired reduction |
| --- | --- | --- | --- | --- |
| 128 | 4.151 → 4.138 | 0.33% | 4.153 → 4.140 | 0.33% |
| 256 | 6.121 → 5.903 | 3.55% | 6.125 → 5.907 | 3.59% |

The five full-operator input gradients agreed at the recorded relative RMS precision for this seed. The measurements cover A800, dense equal-head inputs, and these two widths. Packed varlen and GVA validation remain open before promotion. [Script](evidence/paired.py) · [Run 1](evidence/paired.log) · [Run 2](evidence/repeat.log)

### Follow-up: tile width and pipeline depth

On `main@f8bfd49f`, the full operator dispatch remained intact; only the target launch's `BV` changed. With `BK=128`, BF16, seed 42, eight alternating pairs and `BV=64` versus the default 128, median paired fwdbwd time fell 4.11% for dense `V=256`, 3.14% for four packed sequences, and 4.26% for dense GVA (`H=8,HV=16`). Dense `V=128` regressed 1.27%. All five gradient RRMS values printed zero for these samples. [Mainline pairs](evidence/main-bv64.log) · [Experiment script](evidence/paired-shapes.py)

The original scan's 1.025 ms vs 1.188 ms is a **time reduction**, consistent with the earlier BV=32 paired result. At `V=256`, `BV=32` actually lost 3.9% with both launches fixed at 8 warps and two stages. Four stages fit with `BV=32` (86,032 bytes), but `BV=128` required 344,080 bytes versus the A800 limit of 166,912. `BV=64, stages=3` was faster still. [Stage control](evidence/stages.log) · [Config sweep](evidence/config-matrix.log)

Next experiment: keep `BK` and launch grid fixed, compare adding a single `(BV=64, warps=8, stages=3)` alternative against the existing candidates. Measure cold autotune cost as well as warm throughput before proposing a shared rule. No production changes or cross-platform claims.

原 kernel 门槛 5/5 通过。此前公开 `tests/ops/test_gdn.py` 全文件运行超过 300 秒，完整覆盖仍待验证。见 [gdn.log](evidence/gdn.log) 和 [dependent-tests.log](evidence/dependent-tests.log)。

```bash
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py
python scripts/find_dependent_tests.py fla/ops/common/chunk_o.py
python -m benchmarks.ops.run --op chunk_gdn --modes fwd fwdbwd --base f8bfd49f
pre-commit run --files fla/ops/common/chunk_o.py tests/ops/test_gdn_kernels.py tests/ops/test_gdn.py
```

所有计时使用同一 GPU、同一进程配置、相同输入 seed 和相同 dispatch。先报告架构通用性矩阵，再报告每个形状的绝对时延、相对变化、波动范围、正确性结果和未运行的 profiler 项目。A800 仅支持 Ampere 结论；Hopper、Blackwell 和 Ascend 必须分别有硬件/CI 或 NPU trace 证据。
