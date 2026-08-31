# Day 2 – Inference Systems

## Overview

Day 2 begins with a hand-rolled Transformers inference baseline that exposes
the mechanics hidden behind an inference engine. The first lab separates
prefill from decode, measures TTFT and TPOT, validates KV-cache arithmetic,
and examines the throughput and slot-efficiency tradeoff in static batching.
These measurements will support a later A/B comparison with an inference
engine; later Day 2 work is not documented as complete here.

## Completed Part 1: Inference Anatomy, by Hand

The experiment ran `Qwen/Qwen2.5-1.5B-Instruct` in FP16 through direct
Transformers inference on a fresh Google Colab NVIDIA Tesla T4.

Key results:

- TTFT increased from 0.0447 s at prompt length 128 to 0.4342 s at prompt
  length 2048, reflecting the larger prefill workload.
- Measured KV-cache storage matched the architectural calculation of 28.0
  KB/token.
- Static batch-8 throughput reached 96.9 tokens/s versus 34.0 tokens/s for
  batch 1, a 2.85× increase, while mixed output lengths reduced slot
  efficiency to 0.344.
- Machine verification returned `GREEN CHECK: PASS`.

See [the full lab review](01-inference-anatomy/) for the measurements,
interpretation, and verification criteria.

## Artifacts

- [`baselines.json`](01-inference-anatomy/baselines.json) preserves the model,
  data type, TTFT values, exported TPOT baseline, and batch-throughput results
  for later comparison.
- [`kv_check.json`](01-inference-anatomy/kv_check.json) records theoretical and
  measured KV-cache storage plus the 4096-context peak allocation rate.
