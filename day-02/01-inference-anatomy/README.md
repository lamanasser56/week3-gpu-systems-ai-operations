# Lab W3D2: Inference Anatomy, by Hand

## Overview

This lab established a hand-rolled inference baseline before later comparison
with an inference engine. It ran `Qwen/Qwen2.5-1.5B-Instruct` in FP16 using
direct Transformers inference, without a vLLM server, on a fresh Google Colab
NVIDIA Tesla T4.

The experiment separated prefill from decode, measured time to first token
(TTFT) and time per output token (TPOT), compared KV-cache arithmetic with
measured cache storage, distinguished peak allocation from cache size, and
tested static batching with mixed output lengths. [`baselines.json`](baselines.json)
is intentionally preserved for a later A/B comparison with an inference
engine.

## TTFT and TPOT

| Prompt tokens | TTFT (s) | TPOT (s) | Total (s) | Streamed tokens |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0447 | 0.0420 | 5.4266 | 129 |
| 512 | 0.0782 | 0.0370 | 4.8189 | 129 |
| 2048 | 0.4342 | 0.0498 | 6.8050 | 129 |

TTFT rose substantially as prompt length increased, consistent with prefill
processing the input sequence before the first generated token becomes
available. TPOT remained in a relatively similar range compared with the much
larger TTFT change; the timings are not claimed to be perfectly monotonic.

Streaming timestamps were necessary to distinguish first-token latency from
total generation latency. A warm-up generation was discarded before the
measurement so CUDA initialization and kernel warm-up were not included in
TTFT.

The exported baseline contains a 512-token TPOT value of **0.0355 s**. This is
separate from the **0.0370 s** value in the streaming table because it came
from a later measurement call during baseline export.

## KV-cache arithmetic and measurement

For Qwen2.5-1.5B, the per-token FP16 KV-cache calculation uses:

- 28 transformer layers;
- 2 KV heads;
- head dimension 128;
- 2 bytes per FP16 value; and
- a factor of 2 for K and V.

```text
2 × 28 × 2 × 128 × 2 bytes = 28 KB/token
```

| Context | Total tokens | Peak (KB/token) | Measured KV (KB/token) |
| ---: | ---: | ---: | ---: |
| 512 | 768 | 63.4 | 28.0 |
| 2048 | 2304 | 84.0 | 28.0 |
| 4096 | 4352 | 87.6 | 28.0 |

The measured KV cache matched the 28.0 KB/token architectural calculation in
these runs. The larger peak values are **not** KV-cache sizes. They include
other generation-time allocation, such as activations and allocator or
workspace overhead.

These metrics answer different capacity questions: measured KV storage
isolates the cache itself, while peak allocation helps assess whether the
overall request may fit on the GPU.

## Static batching and the straggler tax

The queue contained 24 requests with desired output lengths
`[32, 32, 32, 256]` repeated six times. This deliberately placed short
requests alongside 256-token stragglers.

| Batch size | Wall time (s) | Throughput (tokens/s) | Slot efficiency |
| ---: | ---: | ---: | ---: |
| 1 | 62.14 | 34.0 | 1.000 |
| 4 | 42.24 | 50.0 | 0.344 |
| 8 | 21.80 | 96.9 | 0.344 |

Batch 8 delivered approximately **2.85×** the aggregate throughput of batch
1 (`96.9 / 34.0`). This does not mean that every individual request became
faster.

Slot efficiency fell from 1.000 to 0.344 because a static batch runs until its
longest output completes. Short requests continue to occupy decode slots while
waiting for the 256-token request. This static-batching, or straggler, tax
shows why throughput and slot efficiency describe different aspects of system
performance. The result provides a baseline for later comparison with
continuous batching.

## Machine verification

The verifier produced:

```text
ttft lengths: ['128', '2048', '512'], tpot_s: 0.0355
batch tokens/s 1/4/8: 34.0/50.0/96.9
KV measured 28.0 KB/token vs formula 28.0 KB/token
GREEN CHECK: PASS
```

It checked:

- the expected `baselines.json` schema;
- positive TTFT and TPOT values;
- TTFT at prompt length 2048 greater than TTFT at 128;
- batch-8 throughput greater than batch-1 throughput; and
- measured KV storage within the verifier's allowed range around the 28.0
  KB/token formula.

## Artifacts

- [`baselines.json`](baselines.json) contains the model, data type, TTFT by
  prompt length, separately exported TPOT baseline, and throughput for batch
  sizes 1, 4, and 8.
- [`kv_check.json`](kv_check.json) contains theoretical and measured KV
  KB/token plus peak KB/token from the 4096-context measurement.

Both files are retained as raw evidence, separate from this interpretation.

## Key technical takeaways

- Prefill and decode are distinct phases and should be measured separately.
- TTFT captures first-token latency; total generation time is not TTFT.
- Prompt length has a strong effect on prefill and TTFT.
- KV-cache size can be predicted from model architecture.
- Peak GPU allocation must not automatically be labeled KV cache.
- Static batching improves aggregate throughput but suffers from stragglers.
- Throughput and slot efficiency describe different aspects of performance.
- These measurements form the baseline for later engine-level comparison.
