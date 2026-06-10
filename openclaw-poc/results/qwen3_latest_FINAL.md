## Model Recommendation Report

### Summary
Out of 12 models analyzed, 8 passed (status=ok) with correct tool usage and answers, while 4 failed due to errors. The passing models show a mix of speed tiers, with qwen3:1.7b being the fastest at 8.2s and llama3.1:70b being the slowest at 106.36s. Anomalies were observed in qwen3:1.7b (hallucination) and smollm2:1.7b (excessive iterations).

### Speed Analysis
Top 5 speed tiers (s):\n- qwen3:1.7b: 8.2s (fast)\n- qwen3:8b: 17.25s (fast)\n- qwen3-coder:30b: 14.26s (fast)\n- qwen3.5:latest: 35.23s (medium)\n- devstral:latest: 47.29s (medium)\n
### Reliability Analysis\nAll 5 profiled models made exactly 4 tool calls. Anomalies included: qwen3:1.7b reported 5 words instead of 4, and smollm2:1.7b made 22 tool calls without completing.\n
### Recommendations\n**Use qwen3:1.7b** for fast disposable work due to its speed. **Recommend qwen3-coder:30b** for high-quality analysis. Avoid **llama3.1:70b** due to its slowness and partial execution.\n
### Use Case Assignments\n- **Planner**: qwen3.5:latest (balanced speed and reliability)\n- **Executor**: qwen3-coder:30b (high-quality output)\n- **Synthesizer**: devstral:latest (good middle-ground performance)