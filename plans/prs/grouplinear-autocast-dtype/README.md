# A800 可独立提交的正确性 PR 筛选

## 背景与范围

优先选择能用公开入口和 PyTorch 参考复现、能落在现有测试文件、且社区尚无同范围 PR 的问题。MoM #1611 已有人认领，CP summary #1683 已有 #1752。本轮从训练模块查找独立选题。

## 首选题目

**修复 `GroupLinear` 在 CUDA autocast 下因 bias 而返回 FP32 的问题。** 现有测试使用块对角权重的 `F.linear` 作为参考，但将参数与输入预先转为相同 dtype。实际混合精度训练常保留 FP32 参数。此时 `einsum` 产生 BF16 激活，随后加上一维 FP32 bias，结果提升为 FP32，每个输出元素从 2 字节增至 4 字节。[实现](../../../fla/modules/grouplinear.py) · [现有测试](../../../tests/modules/test_grouplinear.py)

## 复现与查重证据

单张 A800，PyTorch 2.10.0，FP32 输入 `[2,3,12]`，`GroupLinear(12,8,groups=4)`，在 `torch.autocast('cuda', dtype=torch.bfloat16)` 中实测得到以下输出 dtype。

| bias | GroupLinear | 块对角 F.linear |
| --- | --- | --- |
| True | FP32 | BF16 |
| False | BF16 | BF16 |

本地 `cc7018a4` 与查询时上游 `8973fdd7` 的实现和测试文件 Git blob 分别相同。GitHub 按 `GroupLinear`、`GroupLinear autocast` 和 `grouped linear autocast` 查找 issue 与 PR，返回结果中未见相同修复。核对两项在审模型工作的文件列表，唯一相关测试补丁涉及 Ascend FP64 测试设备选择。[模块引入 #1682](https://github.com/fla-org/flash-linear-attention/pull/1682) · [mLSTM #1434](https://github.com/fla-org/flash-linear-attention/pull/1434) · [SwiLA #1702](https://github.com/fla-org/flash-linear-attention/pull/1702)

## 实施范围与复用

- `fla/modules/grouplinear.py` 的 `GroupLinear.forward` 中，以投影结果的 dtype 执行 bias 加法。通过非零 bias 的参考对照确定舍入行为，保留组间独立和 FP32 参数梯度。
- `tests/modules/test_grouplinear.py` 复用现有块对角 `F.linear` 和 `torch.autograd.grad` 对照。现有测试仅比较数值，本次还需断言输出 dtype。

## 实施步骤与验收

- [x] 在最新上游重查 issue 与开放 PR。以相同输入、权重和非零 bias 冻结可复现的基线测试，验证指定失败项是输出 dtype。
- [x] 实施最小修改。在 CUDA BF16 与 FP16 autocast 下检查 bias 开关、输出 dtype、输出值、`dx/dw/db`，并用关闭 autocast 的 FP32 输入作对照。沿用仓库的参考误差预算。
- [x] 运行 `FLA_CI_ENV=0 python -m pytest -q tests/modules/test_grouplinear.py`，用 `python scripts/find_dependent_tests.py fla/modules/grouplinear.py` 确定其他测试。审查依赖当前 FP32 输出的调用方，PR 正文写明受影响的 autocast 用法。
- [x] 候选标题为 `[Fix] Preserve the autocast output dtype in biased GroupLinear`。证据以修改前后 dtype、数值与梯度对照为主。
