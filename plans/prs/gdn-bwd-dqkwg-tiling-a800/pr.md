# [Perf] Evaluate a smaller value tile for GDN chunk backward on A800

## Archived decision

The final A800 experiment on `399504c6` evaluated a tenth autotune config (`BV=64`, 8 warps, 3 stages). At `V=256` it reduced warm GDN forward/backward time by 3.22–4.30%. At `V=128` the tuner selected the original tile. First-call latency increased from 20.95 s to 22.01 s on the measured dense shape. The experiment was closed under the approved cold-start criterion. The earlier BV=32 draft below is retained as experiment history. [Final evidence](README.md)

## Summary

Experimental PR draft. The candidate sets `BV=32` in `chunk_bwd_dqkwg` for the measured BF16 GDN training workload. The baseline uses `BK=128, BV=128`. Both configurations launch the same `(128,32)` grid. The candidate currently runs through an experiment-local wrapper.

## Benchmark / NCU

On one A800-SXM4-80GB at `f8bfd49f`, two independent processes measured a full `chunk_gated_delta_rule` forward and backward using CUDA events. Each process ran 12 alternating A/B pairs per width, with 12 training calls per timed block. Inputs and upstream gradients were fixed within each pair. The workload used `B=2,T=8192,H=HV=16,K=128,BT=64`, seed 42, PyTorch 2.10.0+cu130 and Triton 3.6.0.

| V | Baseline → candidate, run 1 (ms) | Baseline → candidate, run 2 (ms) | Median paired time reduction, runs 1 / 2 |
| --- | --- | --- | --- |
| 128 | 4.151 → 4.138 | 4.153 → 4.140 | 0.33% / 0.33% |
| 256 | 6.121 → 5.903 | 6.125 → 5.907 | 3.55% / 3.59% |

## Test plan and scope

The five input gradients (`dq/dk/dv/dg/dbeta`) matched the baseline at the recorded relative RMS precision in both dense workloads. Promotion to a production change requires checking packed sequences, grouped value heads, tail dimensions and other callers of this shared kernel. Hardware evidence covers SM80. The timing data establishes the observed latency change. Attributing it to a hardware bottleneck requires separate profiling.

[Measurement script](evidence/paired.py) · [First process](evidence/paired.log) · [Second process and observed tiles](evidence/repeat.log)
