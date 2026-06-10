# Model Recommendation Report

## Summary
The benchmark evaluated **14** models with a status of `ok`. Of these, **11** met the health criteria (4 tool calls, non‑empty answer, correct answer preview) and are considered **passing**. The remaining **3** `ok` models exhibited critical anomalies that caused them to fail the functional check, and **6** models failed outright due to errors such as time‑outs or lack of tool support. In short, the landscape is split roughly two‑thirds viable and one‑third problematic, with a wide spread in execution speed.

## Speed Analysis
The five models selected for in‑depth profiling span the full speed spectrum:

| Model | Time (s) | Speed Tier |
|-------|----------|------------|
| **qwen3:1.7b** | **8.2** | Fast (<25 s) |
| qwen3‑coder:30b | 14.26 | Fast |
| qwen3.5:35b‑a3b‑coding‑nvfp4 | 51.4 | Medium (25‑65 s) |
| smollm2:1.7b | 29.27 | Medium |
| **cogito:70b** | **193.61** | Slow (>65 s) |

The **fastest** model is **qwen3:1.7b** at **8.2 seconds**, while the **slowest** passing model is **cogito:70b** at **193.61 seconds**. The medium tier contains both well‑behaved and anomalous entries, illustrating that speed alone does not guarantee reliability.

## Reliability Analysis
### Profiling Details
1. **qwen3:1.7b** – *Fast* (8.2 s) – Completed **4 tool calls**, produced a full answer (answer_len = 572). **Anomalies:** none observed.
2. **cogito:70b** – *Slow* (193.61 s) – Completed **4 tool calls**, answer length = 384. **Anomalies:** none observed, though the long runtime may affect throughput.
3. **qwen3.5:35b‑a3b‑coding‑nvfp4** – *Medium* (51.4 s) – Executed the required **4 tool calls** but returned an **empty final answer** (answer_len = 0). This is a critical reliability issue despite correct tool usage.
4. **llama3.1:70b** – *Slow* (106.36 s) – Performed **only 1 tool call** before halting, violating the 4‑call contract. The partial execution resulted in a valid answer length (112) but the workflow was incomplete.
5. **smollm2:1.7b** – *Medium* (29.27 s) – Entered a **loop of 22 tool calls** and never produced a proper answer, finally aborting with *max_iterations_exceeded*. This demonstrates an uncontrolled iteration bug.

Overall, the fast and slow passing models behaved reliably, while the medium‑tier selections exposed the most diverse failure modes: empty answers, premature termination, and runaway loops.

## Recommendations
*For *fast disposable work* where latency is paramount, we **recommend** the **qwen3:1.7b** model. It consistently finishes in under ten seconds with correct tool usage and no observable anomalies.*

*For *high‑quality, thorough analysis* where runtime is acceptable, the **cogito:70b** model is the clear choice. Despite its slower pace, it delivers complete tool interaction and a robust answer.*

*Models to **avoid** in production pipelines are **qwen3.5:35b‑a3b‑coding‑nvfp4**, **llama3.1:70b**, and **smollm2:1.7b**. Each exhibits a distinct reliability flaw—empty output, incomplete tool sequence, and uncontrolled iteration, respectively—that could compromise downstream tasks.*

## Use Case Assignments
* **Planner:** **qwen3:1.7b** – Its rapid execution makes it ideal for generating task plans without bottlenecking the pipeline.
* **Executor:** **cogito:70b** – The thorough, error‑free execution is suited for the heavy‑lifting stage where correctness outweighs speed.
* **Synthesizer:** **gemma4:latest** (fast‑medium with 21.06 s runtime and clean tool usage) – Provides a balanced blend of speed and reliability for aggregating results into a final narrative.

These assignments leverage each model’s strengths while mitigating the risks highlighted in the benchmark.

---
*Prepared by the automated pipeline agent on 2026‑06‑10.*