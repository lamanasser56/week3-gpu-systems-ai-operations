# Lab W3D1 Extra: The Memory Leak Hunter

This lab established a clean reload/unload baseline, measured a deliberately
leaky inference loop, and verified a corrected loop using an automated
memory-growth threshold. The lab passed its machine Green Check.

## Recorded results

| Run | Slope (MB/iteration) | Threshold (MB/iteration) | Detector result |
| --- | ---: | ---: | --- |
| Deliberate leak | 211.248 | 1.0 | Leaking |
| Fixed | 0.286 | 1.0 | Not leaking |

Across five clean baseline cycles, reserved memory reached 3134.0 MB after
each load and returned to **0.0 MB after every unload**. The deliberate run's
211.248 MB/iteration slope exceeded the 1.0 MB/iteration threshold and was
correctly detected as leaking. After the fix, the slope fell to 0.286
MB/iteration, below the threshold, and was correctly classified as not
leaking. Under this test, the corrected run was effectively flat after its
initial allocator plateau.

## Cause and correction

Retaining output tensors keeps their GPU storage alive. When inference runs
without `torch.no_grad()`, retained outputs can also keep their autograd graphs
and associated tensors reachable. Together, those reference chains can cause
iteration-by-iteration memory growth.

The correction addressed both causes: inference avoided gradient tracking,
and output references were released. Garbage collection and CUDA cache cleanup
were then used during controlled cleanup so that the verification measured a
stable workload state. The result supports the narrow conclusion that the
tested loop no longer exhibited growth above the detector threshold.

## Evidence

[`leak_report.json`](leak_report.json) contains the baseline cycles, detector
threshold, fitted slopes, classifications, and raw sample series. It is kept
separate from this explanatory documentation.
