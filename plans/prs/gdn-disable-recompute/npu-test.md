# 在昇腾 910B 上测试 GDN `disable_recompute`

状态：已运行，全部通过。结果见文末[实测结果](#实测结果)。下面的步骤是运行前写的，其中的预期来自读代码和 A800 结果。

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
| `peak_mib ... delta` | 4 层与 1 层的差除以 3 即每层增量 | 与推算对比：`D128` 主负载每层 `h` 128 MiB、`w` 64 MiB、`v_new` 64 MiB，合计 256 MiB。本行原先把 `h` 写成 256 MiB，那是 `V=256` 的值 |

计时前后各看一次 `npu-smi info`，中途有其他进程占用同一张卡的结果作废重跑。

## 取空闲卡

每次测试前先用 `evidence/wait-npu.sh` 取空闲卡，再通过 `ASCEND_RT_VISIBLE_DEVICES` 绑定。不设这个变量时默认用 NPU 0，而 NPU 0 可能被别人占用。

```bash
export ASCEND_RT_VISIBLE_DEVICES=$(plans/prs/gdn-disable-recompute/evidence/wait-npu.sh)     # 1 张
export ASCEND_RT_VISIBLE_DEVICES=$(plans/prs/gdn-disable-recompute/evidence/wait-npu.sh 2)   # CP 两卡
```

空闲的判定是该卡没有进程且 AICore 利用率不高于 5%。HBM 占用不作为条件，910B3 空闲时也有约 3.4 GB 读数。空闲卡不够时脚本每 10 分钟重查一次，设 `ONCE=1` 则直接失败。

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

## 实测结果

8 × 昇腾 910B3（HBM 64 GB），npu-smi 25.0.rc1.2，CANN 9.0.0（仓库 CI 用 9.1.0），Python 3.11.18，torch 2.10.0+cpu，torch_npu 2.10.0.post7，triton-ascend 3.6.0，conda 环境 `fla-npu-210`。被测提交 `fa1b94af`，worktree `/home/rhli15/dev/fla-gdn-disable-recompute`。每次测试前用 `evidence/wait-npu.sh` 取空闲卡，跑完后确认卡上没有其他进程。脚本和日志在 [evidence/npu/](evidence/npu/)。

原有环境 `fla-npu`（triton-ascend 3.2.0）跑不了：昇腾后端传的 `enable_ubuf_saving` 编译选项在 3.2.0 里不存在，5 项报 `KeyError`。这与本改动无关。

### 四个问题的答案

1. **编排相同。** 六个形状的调用次数都是重算 `{'wu': 1, 'h': 2}`、保存 `{'wu': 0, 'h': 1}`。
2. **逐位一致。** `saved_vs_recompute` 和 `recompute_twice` 的五个梯度全部 `torch.equal`，测试里的 `torch.equal` 断言可以保留。
3. **`fwdbwd` 配对耗时降 13.4–20.7%**，六个形状全部 8/8 胜出。
4. **每层多保留的显存等于 `w + v_new + h` 的字节数**：`K=V=128` 为 256 MiB，`V=256` 为 449 MiB，`K=V=64` 小形状为 12 MiB。

### 计时

NPU 2，同步加墙钟计时，8 组交错配对，每组 12 次 `fwdbwd`。`npu_bench.log`

| 形状 | 重算 (ms) | 保存 (ms) | 配对降幅 | 胜出 |
| --- | ---: | ---: | ---: | ---: |
| small-D64 `B=4,T=2048,H=4,K=V=64` | 6.409 | 5.553 | 13.36% | 8/8 |
| dense-D128 `B=2,T=8192,H=HV=16,K=V=128` | 45.084 | 38.741 | 14.10% | 8/8 |
| packed-D128 `T=16384`，四条序列 | 126.982 | 100.684 | 20.71% | 8/8 |
| gva `H=8,HV=16,K=V=128` | 45.266 | 39.013 | 13.89% | 8/8 |
| fp16-D128 | 44.304 | 38.156 | 13.87% | 8/8 |
| dense-V256 `K=128,V=256` | 66.143 | 55.888 | 15.50% | 8/8 |

packed 的降幅高于其他形状，没有采集 trace，原因未查。

### 多层显存

NPU 2，`torch.npu.max_memory_allocated`。每层增量取 (12 层差 − 1 层差) / 11。`mem_v256.log`

| 形状 | 1 层 重算 → 保存 | 4 层 | 12 层 | 每层增量 | 按张量字节数推算 |
| --- | --- | --- | --- | ---: | ---: |
| BF16 `K=128,V=256` | 2742 → 2614 MiB | 3619 → 4840 MiB | 5963 → 10776 MiB | 449.2 MiB | 448 MiB（`h` 256，`v_new` 128，`w` 64） |
| BF16 `K=V=128` | 2038 → 1974 MiB | 2728 → 3432 MiB | 4563 → 7320 MiB | 256.5 MiB | 256 MiB（`h` 128，`v_new` 64，`w` 64） |

这暴露了 A800 记录里的一个问题：[README.md](README.md) 阶段 1a 的 576 MiB 是实验脚本量的，脚本保存了 `w/u/h/v_new` 四个张量。提交 `fa1b94af` 只保存 `w/v_new/h`，不保存 `u`（`V=256` 时 128 MiB），所以生产路径应为 448 MiB。A800 上没有用生产路径重量过。

### 回归

| 卡 | 命令 | 结果 | 耗时 |
| --- | --- | --- | --- |
| NPU 2 | `pytest tests/ops/test_gdn.py -k disable_recompute` | 6 passed | 4 分 13 秒 |
| NPU 6 | `pytest tests/ops/test_gdn.py` | 118 passed，21 skipped | 40 分 56 秒 |
| NPU 7 | `pytest tests/ops/test_gdn_kernels.py tests/ops/test_gdp.py tests/layers/test_gated_deltanet.py` | 79 passed，2 skipped | 57 分 13 秒 |
| NPU 6+7 | `pytest tests/cp/test_gdn.py -k cp2` | 8 passed | 15 分 48 秒 |

均设 `FLA_CI_ENV=0`。跳过的 23 项都是 NVIDIA 专属用例：FlashQLA 16 项、Blackwell guard 3 项、SM120 标志 1 项、TileLang 1 项、`test_chunk_o_deterministic` 2 项。候选没有失败项，所以没有在 base 上跑对照。

### 未做

- `tests/models/test_modeling_gated_deltanet.py`、cp4、cp8。
- NPU trace。
- CANN 9.1.0 上的复测。
