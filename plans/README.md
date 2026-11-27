# 计划索引

本目录记录向 `fla-org/flash-linear-attention` 贡献 PR 的实验和跨平台测试计划。

```
plans/
├── README.md   本索引
├── prs/        每个 PR 候选一个目录，含计划、实测结果、PR 草稿和复现脚本
└── modal/      在 Modal 云端 GPU 上运行 FLA 测试的入口和说明
```

约定：

- 新实验建在 `prs/<实验名>/`，以 `README.md` 记录问题、验证方式、实测结果和结论。PR 草稿放 `pr.md`，脚本和日志放 `evidence/`。
- 状态只用 `in progress`、`killed`、`pushed`、`merged`。实验结束后保留原路径，只更新本索引。
- 各平台的测试结果写在同一个实验目录里，不按平台拆目录。平台专用的步骤单独成文，例如 `npu-test.md`。
- `*.log` 被仓库的 `.gitignore` 忽略，只留在本机。
- 测试 PR 分支时用 `git worktree`，不在主仓库里切分支。`plans/` 只在 `main` 上，被测代码在 PR 分支上，两个目录并存才能同时用到。写法：

  ```bash
  git fetch origin <分支>
  git worktree add ../fla-<实验名> <分支>      # 目录与主仓库同级，共用同一个 .git
  git worktree list                           # 查看
  git worktree remove ../fla-<实验名>          # 测完删除，分支和提交保留
  ```

  同一分支不能同时在两个 worktree 里检出。不要用 `rm -rf` 删目录，否则要再跑 `git worktree prune`。可编辑安装（`pip install -e .`）指向哪个目录，`import fla` 就用哪个目录的代码，跑测试前用 `python -c "import fla; print(fla.__file__)"` 确认。

## 平台

| 平台 | 硬件 | 用途 | 状态 |
| --- | --- | --- | --- |
| A800 | 8 × A800-SXM4-80GB，SM80 | 本机。正确性、配对计时、CP 多卡 | 可用。计时前用 `prs/gdn-disable-recompute/evidence/wait-gpu.sh` 取空闲卡 |
| 910B | 8 × 昇腾 910B3，HBM 64 GB | 验证昇腾后端的正确性和收益 | 可用。conda 环境 `fla-npu-210`（triton-ascend 3.6.0，CANN 9.0.0），旧环境 `fla-npu` 的 triton-ascend 3.2.0 跑不了当前昇腾后端。每次测试前用 `prs/gdn-disable-recompute/evidence/wait-npu.sh` 取空闲卡并设 `ASCEND_RT_VISIBLE_DEVICES` |
| Modal | 云端 H100、B200 等 | SM90 及以上的测试，见 [modal/](modal/PLAN.md) | 已暂停。账号未绑定支付方式，远端 pytest 尚未执行过 |

## PR 实验

平台列写该实验在各平台的实测情况。"未测"表示有计划但没跑，"—"表示不涉及。

| 实验 | 状态 | A800 | 910B | 结果 |
| --- | --- | --- | --- | --- |
| [GDN chunk 反向复用前向中间量](prs/gdn-disable-recompute/README.md) | **pushed** | 已测 | 已测，[结果](prs/gdn-disable-recompute/npu-test.md#实测结果) | 提交 `fa1b94af` 已推到 fork 分支 `perf/gdn-disable-recompute`，未开 PR。`fwdbwd` 配对耗时下降 11.0–13.9%，梯度逐位一致，A800 回归 203 passed。910B 上降 13.4–20.7%，梯度逐位一致，回归 205 passed，CP 两卡 8 passed。生产路径每层多保留 `V=256` 448 MiB、`K=V=128` 256 MiB（原记录的 576 MiB 多算了未保存的 `u`）。[PR 草稿](prs/gdn-disable-recompute/pr.md) |
| [GDN chunk 训练路径的算法优化研讨](prs/gdn-algorithmic-optimization/README.md) | **in progress** | 只读分析 | — | 归纳了社区算法类性能 PR 的五种做法并列出 GDN 候选，候选 B 已独立为上一行的实验。 |
| [MoM 短续写缓存](prs/mom-short-continuation-cache/README.md) | **in progress** | 未测 | — | 上游 issue #1611 已有贡献者认领，本地尚未运行独立验证。 |
| [GDN recurrent 的 GVA 头数校验](prs/gdn-recurrent-gva-heads/README.md) | **merged** | 已测 | — | PR [#1851](https://github.com/fla-org/flash-linear-attention/pull/1851) 已合并，merge commit `399504c6`。A800 recurrent 测试 33 项通过。 |
| [GroupLinear autocast 输出 dtype](prs/grouplinear-autocast-dtype/README.md) | **merged** | 已测 | — | PR [#1847](https://github.com/fla-org/flash-linear-attention/pull/1847) 已合并。修复提交 `70f145f4`，CPU BF16 覆盖提交 `1b149c92`。 |
| [GDN `chunk_bwd_dqkwg` 的 BK/BV 性能扫描](prs/gdn-bwd-dqkwg-tiling-a800/README.md) | **killed** | 已测 | — | `V=256` 的完整前后向耗时下降 3.22–4.30%，首次调优增加 1.06 秒（5.0%），触发冷启动停止条件。 |
| [GDN WY 重建的 V 分块](prs/gdn-wy-vtile-a800/README.md) | **killed** | 已测 | — | 在两种 V 宽度上，现有 `BV=64` 的 WY 时延中位数最低，没有形成优化 PR。 |
| [KDA 八卡 zigzag 序列边界](prs/kda-cp8-zigzag-boundaries/README.md) | **killed** | 已测 | — | 6 个八卡用例通过，另有 5 个对照通过，未发现 zigzag 特有错误。[复现脚本](prs/kda-cp8-zigzag-boundaries/evidence/) |

仓库与社区调研见 [FLA PR 方向与 A800](prs/fla-pr-directions-a800.md)。

## Modal

| 文档 | 状态 | 说明 |
| --- | --- | --- |
| [在 Modal 上运行 FLA 测试](modal/PLAN.md) | **in progress** | 已暂停。镜像构建成功，两次 H100 smoke 都因账号未绑定支付方式被拒。 |
| [runner 使用说明](modal/runner/README.md) | | 唯一入口 `modal/runner/modal_test_runner.py`，换测试只改 `--pytest-args`。 |

`modal/modal-examples/` 是 `modal-labs/modal-examples` 的本地克隆（约 100 MB），只作参考，不进 Git。
