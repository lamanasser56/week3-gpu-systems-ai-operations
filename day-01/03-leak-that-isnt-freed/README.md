# Lab W3D1 Bug: The Leak That Isn't Freed

This lab diagnosed a GPU cleanup bug caused by Python reference ownership and
scope. The bug made sequential quantization measurements unreliable even
though garbage collection and CUDA cache cleanup were called.

## Bug

The broken helper was conceptually:

```python
def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
```

`del model` removes only the helper's local parameter binding. It does not
delete the caller's `model` variable, so the caller can continue to hold a
live reference to the model. Garbage collection cannot reclaim a reachable
object, and `torch.cuda.empty_cache()` cannot release memory belonging to live
tensors.

## Broken-run evidence

| Data type | Resident VRAM (GB) |
| --- | ---: |
| FP16 | 3.061 |
| INT8 | 4.791 |
| INT4 | 2.879 |

This run did not produce a perfectly monotonic FP16-to-INT8-to-INT4 climb.
It did demonstrate incorrect cleanup behavior: INT8 appeared to consume
substantially more resident VRAM than FP16 because the measurements were
contaminated by incorrect object lifetime and cleanup. The broken INT4 result
does not establish normal INT4 quantization behavior.

Memory measurements taken immediately around the broken `unload(model)` call
showed that deleting the local parameter was not a reliable way to release a
caller-owned model. The key issue was the lifetime of the caller's reference,
not whether `empty_cache()` appeared in the helper.

## Fix

The model was deleted in the same scope that owned its binding:

```python
model = load(dtype)
# Run the measurement.
del model
gc.collect()
torch.cuda.empty_cache()
```

Removing the owning reference made the model unreachable, after which garbage
collection and CUDA cache cleanup could perform their intended roles.

## Fixed-run evidence

| Data type | Resident VRAM (GB) | Allocated after cleanup (GB) |
| --- | ---: | ---: |
| FP16 | 3.061 | 0.0 |
| INT8 | 1.742 | 0.0 |
| INT4 | 1.150 | 0.0 |

Correct reference cleanup allowed each measurement to begin without the
previous model remaining live. The expected ordering was restored:

```text
FP16 > INT8 > INT4
```

In this experiment, INT8 required less GPU memory than FP16, and INT4 required
less than INT8. These exact VRAM values apply to this measured GPU, model, and
software setup; they should not be generalized to other environments.

## Verification

```text
GREEN CHECK: PASS
```

The machine verification checked:

```text
int8_gb < fp16_gb
int4_gb < int8_gb
```

Both conditions held for the fixed run.

## Key lesson

`torch.cuda.empty_cache()` is not a substitute for deleting live tensor or
model references. It releases unused cached blocks; it cannot free an
allocation while Python still holds a reference to the object that owns it.

Scope determines which binding `del` removes. Deleting a function's local
parameter does not delete a separate binding in the caller, even when both
names refer to the same object. Cleanup must release the reference in its
owning scope before allocator cleanup can reclaim or reuse the memory.

## Evidence

[`bug_leak_not_freed_report.json`](bug_leak_not_freed_report.json) contains the
broken measurements, recorded diagnosis, corrected measurements, post-cleanup
allocations, and Green Check result.
