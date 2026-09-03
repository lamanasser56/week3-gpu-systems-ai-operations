# Week 3 – GPU Systems & AI Operations

Week 3 builds an LLM serving system one operational layer at a time: first
measure the hardware and model, then understand inference costs, improve the
serving engine, lock a validated model configuration, and finally establish
the load that configuration can promise under a latency objective.

```text
Day 1: measure the hardware and model
  → Day 2: understand inference cost
  → Day 3: improve the serving engine
  → Day 4: optimize and lock the serving configuration
  → Day 5: measure the capacity of that locked build
```

Detailed lab notes and artifacts live on independent day branches rather than
being merged into `main`.

## Week 3 core status

| Day | Core lab | Green Check | Main result | Branch |
| --- | --- | --- | --- | --- |
| 1 | Profile Inference on a Real GPU | `PASS` | Batch 1→8 increased throughput from 31.2 to 199.0 tokens/s while mean GPU utilisation moved from 65.7% to 70.0%. | [`w3d1`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d1) |
| 2 | Inference Anatomy, by Hand | `PASS` | FP16 KV cache measured 28.0 KB/token; static batch 1/4/8 reached 34.0/50.0/96.9 tokens/s. | [`w3d2`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d2) |
| 3 | The Engine Swap | `PASS` | vLLM reached 225.7 tokens/s at concurrency 8 and scaled 3.97× from concurrency 1→8. | [`w3d3`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d3) |
| 4 | Quantise and lock the model | `PASS` | AWQ reached 334.2 tokens/s at concurrency 8 and passed the function-calling gate 10/10. | [`w3d4`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d4) |
| 5 | The benchmark harness | `PASS` | Concurrency 16 remained under the 3.0-second p95 SLO at 712.89 tokens/s; the result is sweep-bounded. | [`w3d5`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d5) |

## The core build progression

### Day 1 — Profile the GPU and model

The week starts with direct Transformers inference on a real GPU. FP16 and
INT8 runs expose how context length affects VRAM and how reduced precision can
save memory without necessarily improving speed in a particular software and
hardware stack. Comparing batch 1 with batch 8 also separates GPU activity
from useful output: the GPU can look only slightly busier while completing far
more token work.

**Takeaway:** “busy” is not the same as productive. Evaluate VRAM, throughput,
latency, and workload goals alongside GPU utilisation.

### Day 2 — Understand inference anatomy

Before changing engines, Day 2 separates prefill from decode and distinguishes
time to first token (TTFT) from time per output token (TPOT). It validates the
KV-cache calculation against measured storage and separates cache size from
overall peak GPU allocation.

The static-batching experiment then demonstrates the straggler tax: short
outputs continue occupying slots while the batch waits for its longest
sequence. Aggregate throughput can rise even while the schedule wastes
capacity.

**Takeaway:** understand where latency and memory go before changing the
engine.

### Day 3 — Swap the serving engine

Day 3 replaces direct serving with vLLM and compares throughput under
concurrency. Continuous batching can retire finished sequences and admit
waiting requests, avoiding part of the slot waste seen in static batching.
The serving path also establishes the OpenAI-compatible `/v1` seam used by the
later smoke test and benchmark harness.

KV-block management—and the associated PagedAttention concept—connects the
engine's scheduler to efficient KV-cache use. The repository's measured Day 3
result is the continuous-batching throughput comparison; it does not claim a
separate PagedAttention benchmark.

**Takeaway:** better scheduling and KV management improve serving efficiency
under load.

### Day 4 — Quantise and lock

Day 4 evaluates AWQ as the production candidate. Smaller quantised weights
create memory headroom, but total GPU allocation need not fall dramatically:
vLLM can use freed memory for additional KV-cache capacity under its configured
memory target.

The choice is validated rather than assumed. A five-prompt quality comparison
found minor degradation but no catastrophic failure, and a structured
function-calling smoke test checked both tool use and restraint. The tested
model and exact serving flags were then locked for subsequent work.

**Takeaway:** optimize memory, but validate behaviour before choosing the
production model.

### Day 5 — Benchmark capacity

Day 5 applies a repeatable benchmark harness to the locked Day 4 build. A
concurrency sweep measures tokens per second, TTFT p95, end-to-end latency p95,
and errors. A 3.0-second p95 service-level objective (SLO) turns those metrics
into an operational capacity decision.

Concurrency 16 was the highest tested point that met the SLO. Throughput was
still rising and p95 had not crossed the target, so 16 is a **sweep-bounded**
capacity point—not proof of the true saturation knee.

**Takeaway:** capacity is the highest load the service can promise while
meeting its SLO, not simply the highest throughput observed.

## Current locked build

The source of truth is the model lock and smoke result on
[`w3d4`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d4/day-04/01-quantise-and-lock-the-model).

- Model: `Qwen/Qwen2.5-1.5B-Instruct-AWQ`
- Quantization: `AWQ`
- Data type: `half`
- Maximum model length: `4096`
- GPU memory utilization target: `0.85`
- Automatic tool choice: enabled
- Tool-call parser: `hermes`
- Function-calling smoke score: `10/10`

```text
--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype half --max-model-len 4096
--gpu-memory-utilization 0.85 --quantization awq
--enable-auto-tool-choice --tool-call-parser hermes
```

## Current measured capacity

The source of truth is the benchmark report, knee result, and capacity note on
[`w3d5`](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d5/day-05/01-the-benchmark-harness).

- End-to-end p95 SLO: `3.0 seconds`
- Highest tested SLO-valid concurrency: `16`
- Status: `sweep-bounded`
- Throughput: `712.89 tokens/s`
- End-to-end latency p95: `2.3426 seconds`
- Measured request rate: approximately `6.84 requests/s`
- Request errors: `0`

## Branches and deeper work

`main` documents the core Week 3 build progression. Each day branch contains
that day's detailed README, measurements, and result artifacts:

- [`w3d1` — GPU profiling and memory behaviour](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d1)
- [`w3d2` — inference anatomy](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d2)
- [`w3d3` — vLLM engine comparison](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d3)
- [`w3d4` — AWQ validation and model lock](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d4)
- [`w3d5` — capacity benchmark](https://github.com/lamanasser56/week3-gpu-systems-ai-operations/tree/w3d5)

Extras, bug labs, and future stretch work stay associated with their individual
day branches. The completed Day 1 Memory Leak Hunter and cleanup-scope bug labs
remain on `w3d1`. The repository does not document the Day 2 Paged KV Allocator
extra or prompt-length bug as complete, and Day 5's concurrency-32 sweep remains
a future stretch for locating the true capacity edge.
