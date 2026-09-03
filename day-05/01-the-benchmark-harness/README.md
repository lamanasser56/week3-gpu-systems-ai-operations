# Lab W3D5: The benchmark harness

## Objective

This lab ran a repeatable benchmark harness against the locked Day 4 model,
`Qwen/Qwen2.5-1.5B-Instruct-AWQ`. It swept concurrency to measure throughput
and tail latency, applied a p95 end-to-end latency SLO of 3.0 seconds, and
identified the highest tested SLO-valid capacity point. That capacity point is
distinct from whichever tested point has the greatest throughput.

The vLLM server used this configuration:

```text
--model Qwen/Qwen2.5-1.5B-Instruct-AWQ
--dtype half
--max-model-len 4096
--gpu-memory-utilization 0.85
--port 8000
--quantization awq
--enable-auto-tool-choice
--tool-call-parser hermes
```

The harness targeted `http://localhost:8000`, used [`prompts.txt`](prompts.txt),
sent 20 requests at each concurrency level, allowed up to 128 output tokens,
and wrote [`bench_report.json`](bench_report.json).

## Results

| Concurrency | Tokens/s | TTFT p50 (s) | TTFT p95 (s) | Latency p95 (s) | Errors | OK | Wall (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94.75 | 0.0521 | 0.0883 | 1.3708 | 0 | 20 | 21.139 |
| 2 | 178.31 | 0.0597 | 0.1141 | 1.4348 | 0 | 20 | 11.233 |
| 4 | 296.45 | 0.0619 | 0.1091 | 1.6486 | 0 | 20 | 6.757 |
| 8 | 490.22 | 0.1327 | 0.1562 | 1.8535 | 0 | 20 | 4.235 |
| 16 | 712.89 | 0.231 | 0.233 | 2.3426 | 0 | 20 | 2.926 |

At concurrency 16, the measured request rate was `20 / 2.926`, approximately
**6.84 requests/s**. All five levels completed with zero request errors.

## Reading the capacity result

- **Concurrency** is the number of requests allowed in flight at once.
- **Tokens/s** measures aggregate output-token throughput across the system.
- **TTFT p95** is the time-to-first-token value that 95% of requests meet or
  beat, capturing tail responsiveness before generation begins.
- **End-to-end latency p95** is the full-response latency that 95% of requests
  meet or beat.
- An **SLO** is the service promise used here to bound acceptable p95
  end-to-end latency at 3.0 seconds.
- A **knee** is the useful operating edge where increasing load stops producing
  worthwhile throughput gains or violates the latency SLO.
- A **sweep-bounded knee** means the reported point is only the best supported
  point within the tested range; the sweep ended before the true edge appeared.

The reported capacity point is therefore **at least concurrency 16 under the
chosen SLO**, with 712.89 tokens/s and approximately 6.84 requests/s. It must
not be described as the proven saturation point: throughput was still rising
at concurrency 16 and its 2.3426-second p95 latency remained below the
3.0-second SLO. A stretch sweep including concurrency 32 would help locate the
true edge.

This decode-heavy workload is likely memory-sensitive, but the core sweep did
not reach saturation, so it does not establish a definitive limiting family.

## What I learned

- Day 3: vLLM → continuous batching → concurrency → throughput.
- Day 4: quantisation → smaller weights → more KV-cache headroom → validate
  tool calling → lock model.
- Day 5: locked model → concurrency sweep → tokens/s plus p95 latency → SLO →
  capacity knee.

## Verification and artifacts

The final verifier output was:

```text
levels: 5, concurrencies: [1, 2, 4, 8, 16], total errors: 0
capacity-note.md: all fields filled
GREEN CHECK: PASS
```

- [`bench.py`](bench.py) is the benchmark harness.
- [`prompts.txt`](prompts.txt) is the benchmark workload.
- [`bench_report.json`](bench_report.json) is the unaltered harness output.
- [`capacity-note.md`](capacity-note.md) records the SLO-valid capacity result.
- [`knee.json`](knee.json) records the target and selected concurrency.
- [`verify_cell.py`](verify_cell.py) is the supplied green-check verifier.
