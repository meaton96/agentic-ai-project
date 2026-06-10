# Model Recommendation Report

## Summary
The recent benchmark analysis of the RIT API model fleet reveals a diverse landscape of performance, ranging from highly efficient and accurate models to those struggling with basic tool-calling logic. Out of the 20 models tested, 13 models achieved a status of 'ok', representing a 65% success rate for the task. However, "success" is a broad term in this context; while many models completed the necessary 4 tool calls, several demonstrated significant reliability issues. Specifically, we observed failures ranging from total lack of tool support (e.g., deepseek-r1:8b, gemma3:27b) to complete task abandonment (llama3.1:7lyb) and logic loops (smollm2:1.7b). The failures observed are particularly critical in a production pipeline, as they can lead to infinite loops or unrecoverable error states. The remaining 35% of the fleet, however, showed much higher levels of stability, providing the necessary foundation for dependable agentic workflows.

## Speed Analysis
The profiled models demonstrate a wide spectrum of latency. The fastest model in our deep-dive is **qwen3:1.7b** with an impressive time of **8.2s**, making it ideal for lightweight, high-frequency tasks. Conversely, the slowest performing model in our selection was **cogito:70b**, which took **193.61s** to complete the task. The speed distribution across the selected five is as follows:
- qwen3:1.7b: 8.2s (Fast)
- qwen3-coder:30b: 14.26s (Fast)
- qwen3:latest: 20.56s (Fast)
- qwen3.5:35b-a3b-coding-nvfp4: 51.4s (Medium)
- cogito:70b: 193.61s (Slow)

This variance indicates that while small-scale models can react near-instantaneously, larger-scale models (like the 70b class) introduce significant latency overhead, likely due to more complex internal reasoning or higher token processing requirements.

## Reliability Analysis
Reliability remains the primary differentiator between models. While models like **qwen3-coder:30b** and **qwen3:1.7b** performed flawlessly with exactly 4 tool calls and no anomalies, other models exhibited concerning behaviors. The **qwen3.5:35b-a3b-coding-nvfp4** model, despite its status being 'ok', returned an `answer_len` of 0, indicating a failure to synthesize the tool outputs into a final response. Even more critical was the observation of **llama3.1:70b**, which failed to utilize the full toolset, stopping after just 1 tool call and leaving the task unfinished. While **smollm2:1.7b** was categorized as an error due to iteration limits, it is worth noting its attempt to use 22 tool calls, illustrating a failure in loop termination logic. The 35b-a3b-nvfp4 variant of Qwen shows promise in speed but fails in content delivery, suggesting that tool-calling accuracy does not always correlate with final output quality.

## Recommendations
Based on the data, I **recommend** using **qwen3:1.7b** or **qwen3-coder:30b** for fast, disposable work where latency is the primary constraint and the task complexity is low. For tasks requiring deep reasoning and high-quality analysis where time is not a constraint, I **recommend** utilizing **cogito:70b**, provided the pipeline can handle the 193s latency. For general-purpose agentic tasks that require a balance of speed and accuracy, I **recommend** the **qwen3:latest** model as it demonstrated stable tool usage and respectable speed. We must avoid using **llama3.1:70b** or **smollm2:1.7b** in any automated pipeline due to their high risk of task abandonment and infinite loops, respectively.

## Use Case Assignments
- **Planner**: **cogito:70b** is assigned here because its high latency is acceptable for high-level strategic planning where reasoning depth is prioritized over speed.
- **Executor**: **qwen3-coder:30b** is assigned to execution because it demonstrates the necessary tool-calling precision and high speed required for rapid action.
- **Synthesizer**: **qwen3:latest** is assigned for synthesis as it shows the stability needed to aggregate various tool outputs into a coherent final response without losing data.
