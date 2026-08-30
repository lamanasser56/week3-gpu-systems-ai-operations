# Lab W3D1: Profile Inference on a Real GPU

This lab measured VRAM use, mean GPU utilization, and token throughput across
FP16 and INT8 inference runs at three context lengths. It also compared
throughput at batch sizes 1 and 8. The lab passed its machine Green Check.

## Recorded results

| Data type | Context | VRAM (GB) | Mean GPU utilization | Throughput (tokens/s) |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 512 | 3.113 | 62.5% | 31.6 |
| FP16 | 2048 | 3.295 | 66.3% | 26.0 |
| FP16 | 4096 | 3.568 | 91.3% | 27.5 |
| INT8 | 512 | 1.805 | 25.2% | 5.9 |
| INT8 | 2048 | 2.035 | 28.9% | 5.6 |
| INT8 | 4096 | 2.309 | 31.2% | 5.3 |

In these runs, increasing context length increased VRAM use for both data
types, consistent with additional context-dependent state such as the KV
cache. INT8 used less VRAM than FP16 at each tested context length, but it was
also substantially slower in this specific environment. These results should
not be generalized to every INT8 implementation or GPU.

## Batch experiment

| Batch size | Throughput (tokens/s) |
| ---: | ---: |
| 1 | 31.2 |
| 8 | 199.0 |

Batching improved aggregate throughput by approximately **6.38×**
(`199.0 / 31.2`), while utilization increased only modestly. The central
lesson is that utilization indicates that the GPU is busy; it does not directly
measure productive throughput. Workload-level metrics such as tokens per
second are necessary when evaluating useful performance.

## Evidence

- [`profile.json`](profile.json): canonical measurements for the six
  data-type and context-length configurations
- [`batch_check.json`](batch_check.json): canonical batch-throughput check
- [`gpu_samples.csv`](gpu_samples.csv): raw GPU sampler output supplied with
  the completed work

The evidence files are preserved separately from this interpretation.
