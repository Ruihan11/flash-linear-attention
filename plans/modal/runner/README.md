# Modal 测试入口

`modal_test_runner.py` 是本仓库唯一的 Modal 启动脚本。换测试只改 `--pytest-args`，不新建脚本。

## 前提

- Modal 客户端装在 `/home/rhli15/dev/cutedsl_playground/.venv/bin/modal`（1.6.4），FLA 的 `.venv` 里没有。
- 登录检查：`modal profile current`。
- 在仓库根目录运行。每次运行都会产生 GPU 费用。

## 运行

```bash
MODAL=/home/rhli15/dev/cutedsl_playground/.venv/bin/modal
RUNNER=plans/modal/runner/modal_test_runner.py

# 默认 smoke：GDN kernel 的 dqkwg 用例
$MODAL run $RUNNER

# 任意 pytest 目标和参数
$MODAL run $RUNNER --pytest-args "-q tests/ops/test_gdn.py"
$MODAL run $RUNNER --pytest-args "-s -v --exitfirst tests/ops/test_gdn_kernels.py"

# 换 GPU 或延长超时（默认 H100，30 分钟）
FLA_MODAL_GPU=B200 FLA_MODAL_TIMEOUT_MIN=90 $MODAL run $RUNNER --pytest-args "-q tests/ops/test_gdn.py"

# 多卡，例如 CP 测试
FLA_MODAL_GPU=H100:8 $MODAL run $RUNNER --pytest-args "-q tests/cp/test_gdn.py"
```

本地命令的退出码等于远端 pytest 的退出码。

## 行为

- **源码**：容器启动时挂载当前 checkout 到 `/root/fla`，包含未提交的改动。改代码不重建镜像。`.git`、`.venv`、`plans`、`profile`、`images` 不上传。
- **依赖**：从 `pyproject.toml` 读取基础依赖和 `cuda`、`test` extra。只有这些声明变化时才重建镜像。版本不固定，实际版本以每次运行开头打印的为准。
- **环境**：远端设置 `FLA_CI_ENV=0`，单进程 pytest。
- **Triton cache**：存在 Modal Volume `fla-triton-cache`，挂载到 `/root/.triton`，跨运行复用。清空用 `modal volume rm fla-triton-cache / -r`。
- **日志**：运行开头打印分支、commit、是否有未提交改动、torch/triton/CUDA 版本和 GPU 型号。完整日志在 Modal 控制台的 `fla-tests` App 下。

## 不做的事

- 不做性能计时。冷启动和首次编译时间不能当作 kernel 性能。
- 没有 dry run 模式。不花钱的检查只有本地导入：`python -c "import runpy; runpy.run_path('$RUNNER')"`（用装了 modal 的 Python）。
