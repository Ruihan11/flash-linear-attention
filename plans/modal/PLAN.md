# 在 Modal 上运行 FLA 测试

状态：**paused**（2026-11-27）。runner 和说明已写好，远端 pytest 尚未执行过。卡在 Modal 账号未绑定支付方式，步骤 5–8 未开始。

## Context

目标是在 `/home/rhli15/dev/flash-linear-attention` 外置的 Modal 示例目录中，建立可复用的 Modal 运行入口，使 FLA 的 CUDA/Triton 测试可以在云端 GPU 容器中执行。仓库本身已经定义了 CUDA extra、pytest 配置、测试依赖和 H100 CI 流程，但没有 Modal 入口。

当前已确认：

- `pyproject.toml` 的 CUDA extra 安装 `torch>=2.7.0`、`triton>=3.3`，测试 extra 安装 `pytest`、`pytest-xdist`。
- `INSTALL.md` 建议 CUDA 使用 `.[cuda,test]`，Hopper 使用 Triton 3.9.0 或更新版本。
- `CONTRIBUTING.md` 的完整本地测试命令是 `FLA_CI_ENV=0 pytest tests/`。
- GitHub 的 H100 smoke workflow 按测试文件分进程运行，使用 `pytest -s -v --exitfirst`，冷 Triton cache 会显著增加首次运行时间。
- Modal 客户端已登录，本地可调用 `/home/rhli15/dev/cutedsl_playground/.venv/bin/modal`。本计划不读取凭证，也不在审批前提交远程任务。

## Approach

先在 `plans/modal/runner/` 建立唯一的可复用 Modal 测试入口。所有 FLA 测试都通过同一个 runner 传入测试路径、pytest 选择器和必要参数，后续不按测试类型新建 Modal 脚本。容器使用 NVIDIA CUDA 开发镜像或兼容的 Debian 镜像，安装仓库 CUDA/test extra，把当前仓库作为本地目录挂载或打包进镜像，设置 `FLA_CI_ENV=0`，在远端工作目录执行 pytest，并打印版本、GPU、测试结果和 Triton cache 信息。

优先用同一个 runner 验证一个小的 GDN kernel smoke test，再用相同入口验证完整的 GDN operator 文件。只改变命令行参数，不复制 runner。Modal 的远端执行应保持单进程，避免把 pytest-xdist 和 GPU kernel autotune cache 混在首个验证中。

## Files to modify

- `plans/modal/runner/`（唯一的 Modal 测试入口目录）
  - `modal_test_runner.py`（固定的 App、镜像、源码传递和 pytest 参数入口）
  - `README.md`（统一运行说明和参数示例）

该目录之外不新增 Modal 启动脚本。仓库生产代码、`pyproject.toml`、GitHub workflow 和测试文件暂不修改。

## Reuse

- 依赖声明：`pyproject.toml` 的 `[cuda]` 和 `[test]` extra。
- CUDA 安装说明：`INSTALL.md`。
- 测试配置和内存 poisoning：`tests/conftest.py`。
- H100 smoke 的分文件执行策略：`.github/workflows/nvidia-h100-smoke.yml`。
- Modal 的 GPU Function、Image 和本地入口模式：`plans/modal/modal-examples/06_gpu_and_ml/openai_whisper/fine_tune_asr.py`、`02_building_containers/install_cuda.py`。

## Steps

- [x] 确定远端 GPU、CUDA/PyTorch/Triton 版本和本地源码传递方式。
- [x] 检查 Modal 当前 API 对本地目录挂载、镜像构建和 GPU 类型的支持。
- [x] 在 `plans/modal/runner/` 内写唯一的 `modal_test_runner.py`，只支持测试路径、pytest 参数和超时等必要选项。
- [x] 在同一目录写 `README.md`，包含登录检查、dry run、单测 smoke、GDN 测试和日志/缓存位置。
- [ ] 提交计划审阅后，启动一次最小远端环境检查。只保留一次成功结果作为验证证据。失败尝试不计入成功次数，但每次失败仍可能产生 Modal 费用，失败后只在能明确修复且仍属于最小 smoke 的情况下重试。
- [ ] 用同一个 runner 在 Modal 上运行 GDN kernel smoke test。
- [ ] 用同一个 runner 在 Modal 上运行 GDN operator 测试文件，并记录跳过项、失败项和首次编译耗时。
- [ ] 如有必要，再在同一个 runner 中增加多 GPU CP 参数入口，不新增第二个启动脚本。普通 FLA operator 测试默认单 GPU。

## Verification

环境检查至少确认：

- `torch.cuda.is_available()`、GPU 名称、计算能力和 CUDA 版本。
- `torch.__version__`、`triton.__version__`、`fla` 从当前源码导入。
- Modal 容器的工作目录和源码版本正确。

功能检查按顺序运行：

```bash
FLA_CI_ENV=0 pytest -q tests/ops/test_gdn_kernels.py -k chunk_bwd_dqkwg
FLA_CI_ENV=0 pytest -q tests/ops/test_gdn.py
```

结果需要记录实际命令、GPU、软件版本、通过/跳过/失败数量和失败日志。未运行的测试不标记为通过。性能测试另行使用配对 CUDA event 计时，不把 Modal 单次冷启动时间当作 kernel 性能。

## Open questions

1. 默认使用 Modal 的 H100 单 GPU吗？这是与仓库 H100 smoke 兼容且适合 Triton 性能验证的选择。
2. 源码默认从当前本地 checkout 传入，还是从公开 Git commit 构建？推荐当前 checkout，便于验证未提交的实验代码。
3. 是否允许实际启动一次 Modal 任务来验证环境？这会产生云端 GPU 和镜像构建费用。用户已授权运行测试，并要求只计一次成功测试，失败不计入成功次数。验证和后续测试都必须复用 `plans/modal/runner/modal_test_runner.py`，成功后立即停止。

## 暂停时的进展（2026-11-27）

已完成步骤 1–4：

- 决定：H100 单卡，`add_local_dir` 运行时挂载当前 checkout，依赖从 `pyproject.toml` 的基础依赖和 `cuda`、`test` extra 读取。
- 产物：`runner/modal_test_runner.py`、`runner/README.md`。本地导入正常。
- 镜像构建成功（22 秒），解析出 `torch==2.15.0`、`triton==3.9.0`、`transformers==5.24.1`、`pytest==9.2.0`，CUDA 13 系列。

未完成：

- 两次 smoke 都在分配 GPU 时被拒，报错 `Please add a payment method to use H100 GPU functions.`。第二次是在设置 100 美元月度上限之后。远端 pytest 从未执行，成功次数为 0。日志在 `runner/logs/smoke-attempt1.txt` 和 `runner/logs/smoke-attempt2.txt`。
- `runner/README.md` 描述的远端行为（源码挂载、Triton cache Volume `fla-triton-cache`、开头打印的版本信息）没有经过真实运行验证。

## 恢复时怎么做

1. 在 Modal 控制台 Settings → Billing 给账号 `rl5409` 绑定支付方式。或者决定先用非 H100 的卡验证 runner（`FLA_MODAL_GPU=A10G`，未试过，是否同样需要绑卡未知，结果不算 SM90 证据）。
2. 决定测哪份代码。runner 挂载当前工作区。暂停时分支是 `perf/gdn-disable-recompute`，`fla/ops/gated_delta_rule/chunk.py`、`tests/ops/test_gdn.py`、`tests/cp/test_gdn.py` 有未提交改动。
3. 在仓库根目录运行步骤 5/6 的 smoke，成功标志是最后一行 `pytest exit code: 0`：

   ```bash
   MODAL=/home/rhli15/dev/cutedsl_playground/.venv/bin/modal
   RUNNER=plans/modal/runner/modal_test_runner.py
   $MODAL run $RUNNER 2>&1 | tee plans/modal/runner/logs/smoke-attempt3.txt
   ```

4. 只允许记一次成功测试，失败不计数。smoke 成功后是否继续步骤 7（`tests/ops/test_gdn.py`）需要先问用户。
