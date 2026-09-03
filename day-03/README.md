# Day 3 – Inference Engines

## Overview

Day 3 compares the hand-rolled static-batching baseline from Day 2 with vLLM
continuous batching. The comparison measures throughput at concurrency levels
1, 4, and 8 and evaluates how effectively each approach scales as concurrency
increases.

## Completed Lab: The Engine Swap

vLLM reached 225.7 tokens/s at concurrency 8, compared with 96.9 tokens/s for
the static batch-8 baseline. From concurrency 1 to 8, vLLM throughput scaled
3.97× while static batching scaled 2.85×, giving continuous batching a 1.39×
scaling advantage. Machine verification returned `GREEN CHECK: PASS`.

See [the full lab review](01-the-engine-swap/) for the measurements and
interpretation.

## Artifacts

- [`ab_report.json`](01-the-engine-swap/ab_report.json) records the baseline
  and vLLM throughput, per-concurrency speedups, and predicted speedup.
- The source [`baselines.json`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/blob/w3d2/day-02/01-inference-anatomy/baselines.json)
  remains with the Day 2 experiment and is referenced rather than duplicated.
