# MoM 短续写缓存的 A800 验证计划

## 背景

KDA 八卡边界实验已归档。MoM 的公开报告给出了一个更容易判定的信号：固定路由后，同一段输入整段执行与分段续写会得到不同的卷积缓存，第 5 个 token 的输出也受影响。当前本地 `MomAttention.forward` 仍按每个 memory 的路由长度选择 padding 分支。[issue #1611](https://github.com/fla-org/flash-linear-attention/issues/1611)

## 方法与复用

在单张 A800 上运行同一层和同一组输入。通过 `MomAttention.forward` 观察真实路由、输出及缓存，沿用 `tests/layers/test_mom.py` 的层构造和设备设置。

## 涉及文件

本次实验结果写入本文件。现有 `tests/layers/test_mom.py::test_causal` 用新调用比较前缀，第 5 个 token 继承分段缓存的情形需要另行验证。issue 已有贡献者认领单 memory 复现和修复。本次工作的增量是验证**多个 memory 中只有一个收到短续写**时，错误是否局限在该 memory。

## 实验步骤

- [ ] 固定上游 SHA、环境和种子，从原报告的 BF16、`conv_size=4` 和固定路由开始。每条执行路径使用独立缓存，比较整段与 `2+2+1`，用 `3+1` 对照普通 decode，并核对实际路由。
- [ ] 在两个 memory 都有真实历史的前提下，构造 6 token 的续写，分别路由 2 和 4 个 token。比较同一输入的整段和分段执行，按 memory 保存 q/k/v 卷积缓存、recurrent state、输出与最大绝对误差。将短路由交换给另一个 memory，检查偏差是否随之移动。
- [ ] 追加一个路由到受影响 memory 的 token，记录后续 decode 输出。用两条 memory 都收到 4 个 token 的输入作对照。记录卷积后端、dtype 和触发分支时的实际路由长度。

## 验证与交付

在单张 A800 上执行，保存最小输入、命令、硬件和逐 memory 结果。若当前 main 已通过原报告的用例，先确认路由与后端，再核查相应提交。若多 memory 实验发现独立故障或修复遗漏，可与原贡献者协调回归用例，落入 `tests/layers/test_mom.py`。若与已认领问题一致，交付可供原 PR 复用的验证记录。
