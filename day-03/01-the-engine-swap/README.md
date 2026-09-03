# Lab W3D3: The Engine Swap

## Overview

This lab compares Monday's direct-Transformers static-batching baseline with
vLLM continuous batching at concurrency levels 1, 4, and 8. The original
[`baselines.json`](../../day-02/01-inference-anatomy/baselines.json) remains
with the Day 2 experiment as the single source of baseline data.

## Measured results

| Concurrency / batch size | Monday baseline (tokens/s) | vLLM (tokens/s) | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 34.0 | 56.8 | 1.67× |
| 4 | 50.0 | 161.2 | 3.22× |
| 8 | 96.9 | 225.7 | 2.33× |

| Scaling metric | Result |
| --- | ---: |
| Static batching, 1→8 | 2.85× |
| vLLM, 1→8 | 3.97× |
| Continuous batching scaling advantage | 1.39× |
| Prediction recorded in [`ab_report.json`](ab_report.json) | 1.3× |

## Interpretation

vLLM scaled better because continuous batching can remove finished sequences
and admit waiting requests, reducing wasted slots caused by mixed output
lengths. The scaling ratio is more useful than interpreting the
per-concurrency speedup trend because the latter is a ratio of two different
throughput curves.

## Machine verification

```text
GREEN CHECK: PASS
```

## Artifacts

- [`ab_report.json`](ab_report.json) contains the A/B throughput results,
  per-concurrency speedups, and the recorded prediction.
- The Day 2 [`baselines.json`](../../day-02/01-inference-anatomy/baselines.json)
  is referenced in place rather than copied into this lab.

No W3D3 notebook was present in the saved Colab downloads when this lab folder
was organized. It can be added here when available.
