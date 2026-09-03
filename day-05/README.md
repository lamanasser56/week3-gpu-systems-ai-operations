# Day 5 – Capacity Benchmarking

## Overview

Day 5 applies a repeatable benchmark harness to the AWQ model locked on Day 4.
The lab sweeps concurrency, measures throughput and tail latency, and uses a
latency service-level objective to report a defensible capacity point.

## Completed Lab: The benchmark harness

The core sweep tested concurrency levels 1, 2, 4, 8, and 16 with 20 requests
per level and no request errors. Concurrency 16 remained under the 3.0-second
p95 latency SLO at 712.89 tokens/s. Because throughput was still rising and the
SLO had not been crossed when the sweep ended, 16 is a sweep-bounded result—not
a proven saturation point.

See [the full lab review](01-the-benchmark-harness/) for the exact results,
capacity interpretation, harness, prompts, and verification artifacts.
