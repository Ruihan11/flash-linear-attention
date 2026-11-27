# 在昇腾 910B 上测试 GDN `disable_recompute`

状态：未运行。本文件是测试步骤，里面的预期都来自读代码和 A800 结果，910B 上没有任何实测。

## 测什么

提交 `fa1b94af`（分支 `perf/gdn-disable-recompute`，基于 `6511ea3e`）给 `chunk_gated_delta_rule` 加了 `disable_recompute`。开启后前向把 `w/v_new/h` 存进 `save_for_backward`，反向少调用一次 `recompute_w_u_fwd` 和一次 `chunk_gated_delta_rule_fwd_h`。

910B 上要回答四个问题：

1. 昇腾是否走同一套编排，开启后确实少了这两次调用
2. 保存与重算两条路径的输出和梯度是否逐位一致
3. `fwdbwd` 耗时降多少
4. 每层多占多少显存

A800 上的对照值：耗时降 11.0–13.9%，逐位一致，主负载每层���保留 576 MiB。[A800 结果](README.md)

## 准备

```bash
git fetch git@github.com:Ruihan11/flash-linear-attention.git perf/gdn-disable-recompute
git switch -c perf/gdn-disable-recompute FETCH_HEAD
git log --oneline -1          # 应为 fa1b94af
npu-smi info                  # 选一张空闲卡，记录型号、驱动、已用显存
python -c "import torch, torch_npu, triton; print(torch.__version__, torch_npu.__version__, triton.__version__)"
python -c "from fla.utils import IS_NPU, device; print(IS_NPU, device)"   # 应为 True npu
```

把以上输出记到结果里。`plans/` 没有进提交，910B 上拿不到，下面的脚本需要从本文件复制。

## 第 1 步：正确性

```bash
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py -k disable_recompute
```

预期 6 项通过。两种失败要区分：

- `AssertionError: disable_recompute=True must not recompute w and u`：昇腾的反向没有走 `chunk.py` 里的跳过分支，问题 1 的答案是否定的，停下来查 dispatch。
- `assert torch.equal(ref, tri), <name>`：两条路径不逐位一致。先用第 2 步脚本看相对误差。如果误差在 1e-3 量级以下，并且同一条路径连跑两次也不逐位一致，说明昇腾 kernel 本身不是逐位确定的，测试需要改成容差比较。这是要带回来讨论的结论，不要在 910B 上直接改测试。

然后跑回归，确认默认路径没有被破坏：

```bash
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdn.py tests/ops/test_gdn_kernels.py
FLA_CI_ENV=0 python -m pytest -q tests/ops/test_gdp.py tests/layers/test_gated_deltanet.py
FLA_CI_ENV=0 python -m pytest -q tests/cp/test_gdn.py -k cp2      # 需要 2 张卡
```

回归建议同时在 base `6511ea3e` 上跑一遍，失败项只有在 base 通过、候选失败时才算本改动引入。A800 上 `tests/ops/test_gdn.py` 全文件约需 50 分钟，用后台进程加日志。

## 第 2 步：调用次数、一致性、计时、显存

保存为 `npu_bench.py`，在仓库根目录运行 `PYTHONPATH=. python npu_bench.py 2>&1 | tee npu_bench.log`。

脚本用同步加墙钟计时，因为 `torch.cuda.Event` 在 NPU 上不可用。`torch.npu.synchronize`、`torch.npu.max_memory_allocated`、`torch.npu.reset_peak_memory_stats` 是按 torch_npu 的接口写的，我没有在 910B 上验证过。如果某个名字不存在，换成该版本 torch_npu 的等价接口，显存也可以改用 `npu-smi info` 的读数。

```python
import importlib
import statistics
import time

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.utils import device

chunk = importlib.import_module('fla.ops.gated_delta_rule.chunk')
sync = torch.npu.synchronize if device == 'npu' else torch.cuda.synchronize
mem = torch.npu if device == 'npu' else torch.cuda
calls = {'wu': 0, 'h': 0}


def count(name, key):
    orig = getattr(chunk, name)

    def wrapped(*args, **kwargs):
        calls[key] += 1
        return orig(*args, **kwargs)
    setattr(chunk, name, wrapped)


count('recompute_w_u_fwd', 'wu')
count('chunk_gated_delta_rule_fwd_h', 'h')


def timed(fn, n=12):
    sync()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    sync()
    return (time.perf_counter() - start) * 1000 / n


def run(name, B, T, H, HV, K, V, dtype, cu=None, layers=(1, 4)):
    torch.manual_seed(42)

    def rand(h, d):
        return torch.randn(B, T, h, d, dtype=dtype).to(device).requires_grad_()
    q, k, v = rand(H, K), rand(H, K), rand(HV, V)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, HV)).to(device).requires_grad_()
    beta = torch.rand(B, T, HV).to(device).requires_grad_()
    dy = torch.randn_like(v)
    cu = torch.tensor(cu, device=device) if cu else None
    inputs = (q, k, v, g, beta)

    def step(flag, depth=1):
        x = v
        for _ in range(depth):
            x, _ = chunk_gated_delta_rule(q, k, x, g, beta, use_qk_l2norm_in_kernel=True,
                                          cu_seqlens=cu, disable_recompute=flag)
        return torch.autograd.grad(x, inputs, dy)

    grads = {}
    for flag in (False, True):
        calls.update(wu=0, h=0)
        grads[flag] = step(flag)
        print('calls', name, 'saved' if flag else 'recompute', dict(calls), flush=True)
    again = step(False)
    for label, other in (('saved_vs_recompute', grads[True]), ('recompute_twice', again)):
        equal = [torch.equal(a, b) for a, b in zip(grads[False], other)]
        err = [((a.float() - b.float()).norm() / a.float().norm()).item() for a, b in zip(grads[False], other)]
        print(label, name, equal, ['%.2e' % e for e in err], flush=True)

    for _ in range(16):
        step(False)
        step(True)
    pairs = []
    for i in range(8):
        pair = {f: timed(lambda f=f: step(f)) for f in ((False, True) if i % 2 == 0 else (True, False))}
        pairs.append((pair[False], pair[True]))
    med = statistics.median
    print('fwdbwd recompute/saved', name, *[round(med(x), 4) for x in zip(*pairs)], 'paired_pct',
          round(med(100 * (1 - b / a) for a, b in pairs), 2), 'wins', sum(b < a for a, b in pairs), flush=True)

    if V == K:      # stacking feeds the output back in as v, which needs matching head shapes
        for depth in layers:
            peak = {}
            for flag in (False, True):
                sync()
                mem.reset_peak_memory_stats()
                step(flag, depth)
                sync()
                peak[flag] = mem.max_memory_allocated() / 2**20
            print('peak_mib', name, 'layers', depth, 'recompute', round(peak[False], 1),
                  'saved', round(peak[True], 1), 'delta', round(peak[True] - peak[False], 1), flush=True)


print(torch.__version__, device, flush=True)
bf, fp = torch.bfloat16, torch.float16
run('small-D64', 4, 2048, 4, 4, 64, 64, bf)
run('dense-D128', 2, 8192, 16, 16, 128, 128, bf)
run('packed-D128', 1, 16384, 16, 16, 128, 128, bf, [0, 4095, 8193, 12289, 16384])
run('gva-H8-HV16', 2, 8192, 8, 16, 128, 128, bf)
run('fp16-D128', 2, 8192, 16, 16, 128, 128, fp)
run('dense-V256', 2, 8192, 16, 16, 128, 256, bf)
```

形状从小到大排，前面的先出结果。如果 `dense-V256` 显存不够或昇腾后端不支持 `K != V`，记录报错并跳过，不影响其余结论。

## 怎么读结果

| 输出行 | 通过标准 | 不满足时 |
| --- | --- | --- |
| `calls ... recompute` | `{'wu': 1, 'h': 2}` | 与 A800 的编排不同，记录实际值 |
| `calls ... saved` | `{'wu': 0, 'h': 1}` | 跳过分支没有生效，停下查 dispatch |
| `recompute_twice` | 全 `True` | 昇腾 kernel 不逐位确定，逐位断言不适用，改看相对误差 |
| `saved_vs_recompute` | 全 `True`，或误差不高于 `recompute_twice` | 两条路径有真实差异，停下查原因 |
| `fwdbwd ... paired_pct` | 为正且 8 组里多数胜出 | 记录数值。墙钟计时噪声比 GPU event 大，区间重叠时重跑一次 |
| `peak_mib ... delta` | 4 层与 1 层的差除以 3 即每层增量 | 与推算对比：`D128` 主负载每层 `h` 256 MiB、`w` 64 MiB、`v_new` 64 MiB，合计 384 MiB |

计时前后各看一次 `npu-smi info`，中途有其他进程占用同一张卡的结果作废重跑。

## 需要带回来的东西

- `npu_bench.log` 全文和 pytest 的通过、失败、跳过数
- 硬件与软件版本：910B 型号、驱动、CANN、torch、torch_npu、triton-ascend
- 逐位一致是否成立。这决定测试里的 `torch.equal` 能否保留
- 如果要解释降幅与 A800 的差异，补一份 NPU trace，看 `recompute_w_u_fwd` 和 `fwd_h` 两个 kernel 在 `fwdbwd` 中的占比。做法见 `.agents/skills/fla-ascend-performance/SKILL.md`

## 已知限制

- 降幅比例不能照搬 A800。昇腾各 kernel 的耗时占比不同，这两个 kernel 占多少要靠 trace 确认。
- "昇腾走同一套编排"是读代码的判断：`gated_delta_rule/backends/` 下只看到 FlashQLA 注册了整个 `chunk_gated_delta_rule`，昇腾注册的是 `recompute_w_u_fwd` 和 `chunk_gated_delta_rule_fwd_h` 这一层。第 2 步的 `calls` 行用来验证它。
- 910B 单卡显存小于 A800 的 80 GB，多层大形状先看余量。
- PR 描述里的昇腾一栏在拿到这些结果前保持"未测"。
