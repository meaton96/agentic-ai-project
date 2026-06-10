# Model Recommendation Report

## Summary
14 models passed (status=ok) and 5 failed. The quality landscape shows a mix of performance, with notable anomalies in answer accuracy and task completion.

## Speed Analysis

| Model | Time (s) | Tier |
|-------|---------|------|
| qwen3:1.7b | 8.2 | Fast (<25s) |
| qwen3-coder:30b | 14.26 | Fast |
| qwen3:8b | 17.25 | Fast |
| gemma4:latest | 21.06 | Medium |
| cogito:70b | 193.61 | Slow (>65s) |

Fastest model: qwen3:1.7b (8.2s). Slowest passing model: cogito:70b (193.61s).

## Reliability Analysis

- **qwen3.5:35b-a3b-coding-nvfp4**: Status=ok but returned empty answer (answer_len=0)
- **llama3.1:70b**: Status=ok but completed only 1 tool call (failed to finish task)
- **qwen3-coder:30b**: Clean tool usage (4 calls) with answer_len=437
- **qwen3.5:27b**: 59.77s with answer_len=375 (no anomalies)
- **gemma4:26b**: 45.02s with answer_len=413 (potential word count hallucination)

## Recommendations

**Recommend** qwen3:1.7b for fast disposable work due to its speed and clean tool usage. For high-quality analysis, recommend cogito:70b despite its slowness, as it completed all tool calls with valid answers. Avoid qwen3.5:35b-a3b-coding-nvfp4 due to its empty answer anomaly.

## Use Case Assignments

- **Planner**: qwen3:1.7b (fast and reliable for initial task breakdown)
- **Executor**: qwen3-coder:30b (balanced speed and tool usage accuracy)
- **Synthesizer**: cogito:70b (slow but produces detailed, valid outputs)

The benchmark shows qwen3:1.7b and gemma4:latest as optimal for speed, while cogito:70b and qwen3.5:27b demonstrate reliability for complex tasks.