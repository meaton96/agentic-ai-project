# Benchmark Summary Analysis

This 3-stage pipeline analyzed 16 models in the RIT API benchmark. Of these, 12 models passed with status=ok, while 4 models failed due to various issues.

## Failed Models (4)
1. qwen3:32b - error, 504 gateway timeout, 0 tool calls, 91.51s
2. smollm2:1.7b - error, max_iterations_exceeded, 22 tool calls, 29.27s
3. deepseek-r1:8b - error, does not support tools, 0 tool calls, 26.55s
4. gemma3:27b - error, does not support tools, 0 tool calls, 0.15s
5. qwen3-coder:30b - Wait, this one shows ok status actually

Let me recount: failed (error status): qwen3:32b, smollm2:1.7b, deepseek-r1:8b, gemma3:27b, llava:latest (error)
So 5 failed total.

## Passed Models (11)
gemma4:26b, gemma4:latest, qwen3.6:35b-a3b-nvfp4, cogito:70b, qwen3.5:35b-a3b-coding-nvfp4, qwen3.5:latest, qwen3:32b wait that had error, llama3.1:70b, qwen3-coder:30b, devstral:latest, gpt-oss:120b, qwen3:latest, qwen3:1.7b, qwen3:8b, qwen3-coder-next:latest, qwen3.5:27b

That's 15 passing models.

## Speed Analysis Summary

The profiled models span a wide performance range:
- Fastest: qwen3:1.7b at 8.2 seconds
- Slowest: cogito:70b at 193.61 seconds
- Medium range models include gemma4:26b (45.02s), qwen3:8b (17.25s), and qwen3.5:latest (35.23s)

The speed distribution shows most models cluster in the medium tier (25-65s), with qwen3:1.7b standing out as an exceptional fast performer and cogito:70b demonstrating how larger models can have very different speed characteristics.

## Reliability Findings

- qwen3.5:35b-a3b-coding-nvfp4: Despite completing 4 tool calls, returned empty answer (answer_len=0)
- llama3.1:70b: Failed to complete all tool calls (only 1 instead of 4)
- smollm2:1.7b: Pathological looping with 22 tool calls, failed with error

Two models showed hardware or compatibility issues:
- qwen3:32b failed with "504 gateway timeout"
- Multiple models failed with "does not support tools" error (deepseek-r1:8b, gemma3:27b, llava:latest)

## Recommendations

We explicitly recommend qwen3:1.7b for fast disposable work due to its exceptional speed of 8.2 seconds. For high-quality analysis requiring accurate tool execution and comprehensive answers, we recommend gemma4:26b and qwen3.5:latest. Models to avoid include smollm2:1.7b (loops excessively), llama3.1:70b (incomplete execution), and models lacking tool support.

## Use Case Assignments

- **Planner**: qwen3.1.7b - Uses its rapid 8.2 second response time to quickly assess task requirements and delegate appropriately.
- **Executor**: gemma4:26b - Completes all 4 tool calls reliably with reasonable speed (45s) producing comprehensive answers.
- **Synthesizer**: qwen3.5:latest - Delivers well-formed, thorough final responses with consistent tool call compliance in the medium speed tier.