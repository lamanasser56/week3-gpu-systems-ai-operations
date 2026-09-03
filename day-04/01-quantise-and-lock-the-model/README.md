# Lab W3D4: Quantise and lock the model

## Objective

This lab evaluated `Qwen/Qwen2.5-1.5B-Instruct-AWQ` as a quantised replacement
for the Day 3 FP16 service. The decision combined throughput, a five-prompt
quality spot check, and a function-calling gate before locking the serving
model and launch configuration.

AWQ reduces model-weight storage. Under vLLM's configured
`--gpu-memory-utilization 0.85`, however, total GPU usage need not fall by the
same amount: vLLM can use the freed memory headroom for more KV-cache blocks.
Quantisation can therefore appear as greater serving capacity rather than a
dramatic reduction in the `nvidia-smi` total.

## Environment and launch configuration

The run used Google Colab with an NVIDIA T4 and a Python 3.10 virtual
environment at `/content/venv`.

| Component | Version |
| --- | --- |
| vLLM | 0.6.6.post1 |
| transformers | 4.46.3 |
| accelerate | 1.1.1 |
| AutoAWQ | 0.2.9 |
| httpx | 0.27.2 |
| openai | 1.54.5 |

The locked launch flags are recorded exactly in [`model-lock.md`](model-lock.md).
The Python 3.10 virtual environment was used because Colab's Python version had
changed while the pinned serving stack required a compatible environment. The
environment itself is not stored in this repository.

## Measured results

The AWQ server used 11,723 MiB of GPU memory and reported 22,955 GPU KV-cache
blocks and 9,362 CPU blocks.

| Build | Concurrency | Requests | Throughput (tokens/s) | Wall time (s) |
| --- | ---: | ---: | ---: | ---: |
| FP16 (Day 3) | 8 | — | 225.7 | — |
| AWQ | 8 | 24 | 334.2 | 4.094 |

The measured AWQ/FP16 throughput ratio was approximately **1.48×**.

## Quality and function-calling gate

Both FP16 and AWQ preserved general task understanding. FP16 was slightly more
consistent and professional in the five-prompt comparison. AWQ showed minor
degradation on some prompts—particularly an odd quantisation analogy and
weaker handling of the informal tool-related prompt—but no catastrophic
degradation.

| Build | Score | Distractor call-free | Majority clean | Passed |
| --- | ---: | ---: | :---: | :---: |
| FP16 | 10/10 | 2/2 | yes | yes |
| AWQ | 10/10 | 2/2 | yes | yes |

AWQ was locked because it passed the required function-calling gate 10/10,
preserved the required structured tool-calling behaviour, and delivered higher
measured throughput. The minor qualitative degradation did not affect the
required smoke-test behaviour.

## What I learned

Day 3 connected vLLM continuous batching and KV-cache management to throughput
under concurrency. Day 4 extended that path: quantisation made the weights
smaller, created more memory headroom for KV cache, and required quality and
tool-calling validation before the model could be locked.

## Verification and artifacts

The final verifier output was:

```text
smoke score: 10/10, distractor clean: True
model-lock.md: all fields filled
GREEN CHECK: PASS
```

- [`model-lock.md`](model-lock.md) records the locked AWQ model and exact flags.
- [`smoke_result.json`](smoke_result.json) records the locked AWQ smoke result.
- [`smoke_test.py`](smoke_test.py) is the supplied function-calling test.
- [`verify_cell.py`](verify_cell.py) is the supplied green-check verifier.
