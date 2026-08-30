# Day 1 – GPU Inference and Memory Behavior

Day 1 combined inference profiling with GPU-memory leak detection. Both
completed labs passed their machine Green Check. The unfinished "Leak That
Isn't Freed" bug lab is intentionally excluded.

## Concepts learned

### Inference profiling

- **VRAM behavior:** Model weights, runtime state, inputs, outputs, and caches
  all contribute to memory use. Measurements must be interpreted in the
  context of the workload that produced them.
- **Context length and KV cache:** Longer contexts can require more KV-cache
  storage. In the recorded runs, VRAM increased with context length for both
  data types.
- **FP16 and INT8:** Quantization reduced measured VRAM in this experiment,
  but the INT8 configuration also delivered lower throughput. Lower precision
  does not guarantee faster inference on every hardware and software path.
- **Utilization and throughput:** GPU utilization describes how busy the GPU
  was, not how much useful work it completed. Tokens per second is the direct
  productivity measure used here.
- **Batching:** Combining requests can improve aggregate throughput by giving
  the GPU more useful parallel work. Batch size must still be balanced against
  latency and memory constraints.

### Memory leak detection

- **Tensor and reference lifetime:** PyTorch cannot release storage while a
  live Python reference still retains the associated tensor.
- **`torch.no_grad()`:** Disabling gradient tracking for inference prevents
  unnecessary autograd graph construction and retention.
- **CUDA caching allocator:** Released tensor storage may remain reserved for
  reuse. Reserved memory and live tensor memory therefore need careful
  interpretation; cache cleanup can be useful during controlled verification.
- **Automated verification:** A measured memory-growth slope and an explicit
  threshold turn visual suspicion into a repeatable pass/fail Green Check.

## Completed labs

1. [Profile Inference on a Real GPU](01-profile-inference/)
2. [The Memory Leak Hunter](02-memory-leak-hunter/)
