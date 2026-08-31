# Day 1 – GPU Inference and Memory Behavior

## Day 1 overview

Day 1 focused on profiling LLM inference on a real GPU and diagnosing
application-level GPU memory leaks. The work used
`Qwen/Qwen2.5-1.5B-Instruct` with Transformers-based direct inference in
Google Colab on an NVIDIA Tesla T4 with 15360 MiB of GPU memory.

The experiments examined VRAM behavior, FP16 and INT8 execution,
context-length effects, GPU utilization versus useful throughput, and
batching. A second lab established a clean memory control, reproduced a
deliberate leak, measured its growth, and verified a fix. A third lab diagnosed
why deleting a function-local model parameter did not release a caller-owned
model. All three Day 1 labs passed their machine Green Check.

## Profile inference experiment

The same model was profiled at three context lengths for each data type.

| dtype | context | VRAM GB | GPU util % | tokens/s |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 512 | 3.113 | 62.5 | 31.6 |
| fp16 | 2048 | 3.295 | 66.3 | 26.0 |
| fp16 | 4096 | 3.568 | 91.3 | 27.5 |
| int8 | 512 | 1.805 | 25.2 | 5.9 |
| int8 | 2048 | 2.035 | 28.9 | 5.6 |
| int8 | 4096 | 2.309 | 31.2 | 5.3 |

VRAM increased with context length for both data types, consistent with the
cost of additional context-dependent state such as the KV cache. INT8 used
substantially less VRAM than FP16 at every tested context length. It was not
faster, however, in this particular Tesla T4, Transformers, and bitsandbytes
setup. The result demonstrates a memory benefit in this experiment; it does
not establish INT8 quantization as either a general speed optimization or a
general performance penalty.

## Batch utilization experiment

| Batch size | Throughput | Mean GPU utilization |
| ---: | ---: | ---: |
| 1 | 31.2 tokens/s | 65.7% |
| 8 | 199.0 tokens/s | 70.0% |

Batch 8 delivered **6.38×** the aggregate token throughput of batch 1 while
mean GPU utilization increased by only **4.3 percentage points**. In other
words, the GPU completed about 6.38 times as much aggregate token work while
the reported utilization moved only from 65.7% to 70.0%.

This is the central profiling lesson from Day 1: **busy is not the same as
productive**. GPU utilization describes activity, but it does not directly
measure useful work. Throughput, latency, memory use, and workload goals must
also be considered when deciding whether inference is using a GPU efficiently.
Batching can improve aggregate throughput by exposing more parallel work, but
its latency and memory tradeoffs still need to be evaluated.

## Memory Leak Hunter

### Clean reload/unload control

The control performed five model reload/unload cycles. Every cycle measured
3134.0 MB reserved after loading and 0.0 MB after unloading. This repeatable
return to zero established clean reload/unload behavior before a leak was
introduced deliberately.

### Deliberate leak

The deliberately leaky run contained 20 samples. Its fitted slope was 211.248
MB/iteration, well above the 1.0 MB/iteration detector threshold, so the
detector correctly returned `leaking = true`.

| Sample position | Reserved memory |
| --- | ---: |
| Iteration 0 | 3254 MB |
| Iteration 5 | 4294 MB |
| Iteration 10 | 5354 MB |
| Iteration 15 | 6414 MB |
| Final sample | 7266 MB |

The test introduced two causes: inference ran without `torch.no_grad()`, and
output logits were retained in a growing Python list. Live Python references
kept the output tensors reachable; because gradient tracking was enabled,
those references could also retain their associated autograd graphs and
intermediate tensors. The resulting iteration-by-iteration growth was
therefore expected for this deliberately constructed workload.

## Memory leak fix

The fixed run also contained 20 samples. Its slope was 0.286 MB/iteration,
below the same 1.0 MB/iteration threshold, and the detector returned
`leaking = false`. Reserved memory measured 10420 MB in the first sample and
then stabilized around 10440 MB in subsequent samples.

The absolute reserved-memory level remained high. The important detector
signal was that it stopped growing iteration after iteration. A live workload
does not need to return reserved memory to zero to demonstrate stable memory
behavior, particularly when the CUDA caching allocator retains storage for
reuse.

The fix removed retained output references, used `torch.no_grad()`, released
stale references, and applied garbage collection and CUDA cache cleanup where
appropriate. Together, those changes made the measured memory trend
effectively flat under the detector threshold.

## The Leak That Isn't Freed

The third lab investigated a broken unload helper whose `del model` statement
deleted only its local parameter binding. The caller's `model` binding remained
live, so garbage collection and `torch.cuda.empty_cache()` could not release
the model's live allocations.

| Run | FP16 (GB) | INT8 (GB) | INT4 (GB) |
| --- | ---: | ---: | ---: |
| Broken | 3.061 | 4.791 | 2.879 |
| Fixed | 3.061 | 1.742 | 1.150 |

The broken run was not a perfectly monotonic FP16-to-INT8-to-INT4 climb. Its
important signal was contaminated lifetime behavior: INT8 appeared to use
substantially more resident VRAM than FP16 because the prior object was not
reliably released. The broken INT4 value is not treated as evidence of normal
quantization behavior.

The fix deleted `model` in the caller scope that owned the binding, followed
by garbage collection and CUDA cache cleanup. Allocated memory after cleanup
was 0.0 GB for FP16, INT8, and INT4, and the expected `FP16 > INT8 > INT4`
ordering was restored. The lab produced `GREEN CHECK: PASS`.

The key lesson is that `torch.cuda.empty_cache()` is not a substitute for
deleting live tensor or model references. Scope matters: `del` removes a
specific binding, and deleting a callee's local parameter does not remove the
caller's reference to the same object.

## Machine Verification

All three completed labs produced:

```text
GREEN CHECK: PASS
```

The profile-inference verification checked:

- the expected result schema;
- increasing VRAM as context length increased;
- higher FP16 VRAM than INT8 VRAM; and
- higher throughput for batch 8 than batch 1.

The Memory Leak Hunter verification checked:

- the clean reload/unload control;
- an independent slope refit confirming the deliberate leak; and
- a fixed-run slope below the allowed threshold.

The Leak That Isn't Freed verification checked that the fixed measurements
satisfied `int8_gb < fp16_gb` and `int4_gb < int8_gb`.

These automated checks make the experiments repeatable and guard against
drawing conclusions from visual inspection alone.

## Artifacts

### `01-profile-inference/`

- [`profile.json`](01-profile-inference/profile.json) contains the six
  data-type and context-length measurements.
- [`batch_check.json`](01-profile-inference/batch_check.json) contains the
  canonical batch 1 and batch 8 throughput results.
- [`gpu_samples.csv`](01-profile-inference/gpu_samples.csv) contains the raw
  timestamped GPU utilization and memory samples supplied with the lab.

### `02-memory-leak-hunter/`

- [`leak_report.json`](02-memory-leak-hunter/leak_report.json) contains the
  reload/unload control, detector settings and classifications, fitted slopes,
  and raw leaky and fixed sample series.

### `03-leak-that-isnt-freed/`

- [`bug_leak_not_freed_report.json`](03-leak-that-isnt-freed/bug_leak_not_freed_report.json)
  contains the broken and fixed measurements, diagnosis, post-cleanup
  allocations, and Green Check result.

The raw experimental evidence is kept separate from explanatory
documentation, and its recorded measurements are not altered here.

## Key Day 1 takeaways

- Context length has a measurable VRAM cost.
- Lower precision reduces memory, but not necessarily latency or throughput.
- GPU utilization is not a direct throughput metric.
- Batching can dramatically improve aggregate GPU productivity.
- Live tensor references can create application-level GPU memory leaks.
- `torch.cuda.empty_cache()` cannot free allocations that remain live through
  Python references.
- `del` must remove the binding in the scope that owns the model reference.
- Memory-leak detection should use repeatable measurements and trends rather
  than visual inspection alone.
- Automated verification makes performance experiments reproducible.

## Completed labs

1. [Profile Inference on a Real GPU](01-profile-inference/)
2. [The Memory Leak Hunter](02-memory-leak-hunter/)
3. [The Leak That Isn't Freed](03-leak-that-isnt-freed/)
