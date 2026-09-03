# Model lock (team record)

## The locked model

- Model id: Qwen/Qwen2.5-1.5B-Instruct-AWQ
- Quantisation: awq
- Why this one: Passed the function-calling smoke test 10/10, preserved required tool-calling behaviour, and delivered higher measured throughput while using quantised weights.

## The launch flags

The exact vLLM flags our team runs:

--model Qwen/Qwen2.5-1.5B-Instruct-AWQ --dtype half --max-model-len 4096
--gpu-memory-utilization 0.85 --quantization awq
--enable-auto-tool-choice --tool-call-parser hermes

- Tool-call parser: hermes

## The smoke score

- Score (valid behaviours out of 10): 10
- Distractor stayed call-free in the majority: yes
- Passed the gate (>= 8/10 and distractor majority clean): yes
- Measured against: both — AWQ 10/10, fp16 10/10

## Quality spot check note

- Both builds preserved general task understanding. FP16 was slightly more consistent in the five-prompt side-by-side, while AWQ showed minor quality degradation on some prompts, but no catastrophic degradation and it still passed the function-calling smoke test 10/10.
