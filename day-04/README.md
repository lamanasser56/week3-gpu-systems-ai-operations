# Day 4 – Quantisation and Model Lock

## Overview

Day 4 tests an AWQ-quantised serving candidate against the Day 3 FP16 vLLM
result, checks qualitative behaviour and structured function calling, and
locks a model configuration for later bootcamp work.

## Completed Lab: Quantise and lock the model

`Qwen/Qwen2.5-1.5B-Instruct-AWQ` passed the required function-calling gate
10/10 and reached 334.2 tokens/s at concurrency 8, compared with 225.7 tokens/s
for FP16. The AWQ model was therefore locked with the tested vLLM launch
configuration.

In this measured setup, AWQ reduced the model weight footprint, but total GPU
memory reported by `nvidia-smi` did not fall by the same amount. Under vLLM's
configured GPU-memory-utilization target, memory freed by smaller weights
could be reused for additional KV-cache blocks, increasing serving headroom.

See [the full lab review](01-quantise-and-lock-the-model/) for measurements,
interpretation, reproducibility notes, and verification artifacts.
