# Capacity note (team, one page)

## The numbers

- Locked model: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Target p95 end-to-end latency (your SLO today): 3.0 seconds
- Knee concurrency (highest concurrency whose p95 is still under target): 16 (sweep-bounded)
- Tokens per second at the knee: 712.89
- Max sustainable request rate at the target p95: 6.84 req/s

## The limiting family

One sentence, using this morning's triage lens (compute vs memory vs overhead):
which family limits this stack at the knee, and the tell that points to it.

- Memory is the likely limiting family for this decode-heavy serving workload, but the true saturation point was not reached in the 1–16 sweep: throughput was still rising at concurrency 16 while p95 remained under the 3.0-second SLO.

## Why the knee, not the peak

One sentence in your own words on why you report the knee at the SLO rather than
the peak throughput.

- Capacity should be reported at the highest load that still satisfies the latency SLO, because a higher throughput point is not useful if requests become too slow to meet the service promise.
