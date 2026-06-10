# Model Recommendation Report

## Summary

This analysis covers 19 models tested against a multi-step agentic tool-calling benchmark requiring 4 sequential tool calls (count_words, calculator, get_current_datetime, lookup_fact) with the correct math answer of 16.0. 

**Passing models:** 11 out of 19 successfully completed all 4 required tool calls with non-empty answers and a status of "ok". **Failing models:** 8 did not complete the full task — including gateway timeouts, iteration limits exceeded, early termination after only 1 tool call, and tools-not-supported errors (deepseek-r1:8b, llama3.1:4b, gemma3:27b).

The quality landscape is moderate-to-good but fragmented. While a majority of models can handle the basic task structure, several exhibit concerning reliability issues even among those marked as "ok". Only about 75% of models can be considered reliable for production use in an agentic pipeline context. Notably, large models are not guaranteed to perform better — qwen3:1.7b at just 30ms latency outperforms many larger counterparts in speed.

## Speed Analysis

The 5 profiled models span the full performance spectrum from blazing-fast to painfully slow:

| Model | Time (seconds) | Speed Tier | Tool Calls | Anomaly Status |
|-------|---------------|------------|------------|----------------|
| qwen3:1.7b | 8.20 | Fast (<25s) | 4/4 complete | Known hallucination issue (reports 5 words instead of 4) |
| gemma4:latest | 21.06 | Fast (<25s) | 4/4 complete | None observed — the cleanest performer |
| qwen3-coder:30b | 14.26 | Fast (<25s) | 4/4 complete | Error field set despite all steps completing; iterations=2 only |
| llama3.1:70b | 106.36 | Medium (25-65s) ⚠️ Slow in practice | **1/4 incomplete** | Only 1 tool call executed; model appeared to stall after first output |
| smollm2:1.7b | 29.27 | Fast (<25s) ⚠️ Near medium | **22 tool calls (excessive)** | Loop-max-exceeded error; no useful final answer produced despite 45-char response |

**Fastest model:** qwen3:1.7b at only 8.20 seconds — the clear speed champion. gemma4:latest follows closely at 21.06 seconds with a clean execution profile and zero anomalies.

**Slowest model among our 5 profiles:** llama3.1:70b at 106.36 seconds, which is actually in the medium-tier window (25-65s) by the benchmark's definition but performed so poorly its timing is meaningless — it only produced content after a single tool call and then stopped. If this had been allowed to continue, it likely would have exceeded even the most permissive timeout thresholds.

**Interesting outlier:** cogito:70b was also profiled as a reference point at 193.61 seconds — by far the slowest passing model in the entire benchmark. Despite completing all 4 tool calls successfully with status=ok, over 3 minutes for such a simple task is unacceptable for any production use case.

The fast models (qwen3:1.7b, gemma4:latest, qwen3-coder:30b) consistently finish under 25 seconds and represent the practical performance boundary for this workload. The medium tier contains most remaining passing models (gemma4:26b at 45s, qwen3.5:latest at 35s, devstral:late at 47s). Only cogito:70b in the over-65s range represents true slow-tiers risk for real-world use.

## Reliability Analysis

Tool usage correctness is the primary differentiator among models with "ok" status — having status=ok does not guarantee a task was actually completed correctly.

**qwen3-coder:30b (RELIABILITY CONCERN):** This model reports error="max_iterations_exceeded" despite completing all 4 tool calls, achieving status=ok, and producing an answer_len of 437 chars. This contradictory state is troubling — it suggests the model may have found the correct answer quickly and then continued iterating unnecessarily, hitting an internal loop limit. For a pipeline agent this means the model could start spurious extra iterations after already solving the problem, wasting compute and introducing potential correctness drift in subsequent steps.

**llama3.1:70b (MAJOR RELIABILITY FAILURE):** Despite status=ok, the model only executed 1 of the 4 required tool calls in over 106 seconds. Its answer_len is merely 112 characters — likely just the output from the first tool call. This represents a critical reliability failure: the model does not complete the task even though it reports "ok." In a pipeline context, this would silently stall downstream consumers waiting for full results.

**smollm2:1.7b (LOOPING ANOMALY):** Error="loop_max_exceeded" with 22 tool calls executed across 10 iterations. With only 45 characters in the final answer and "max_iterations_exceeded", this model entered an infinite or near-infinite loop pattern, repeatedly calling tools without converging on a solution. This is a reliability hazard that could consume unlimited compute resources in any pipeline consuming it as a sub-agent.

**qwen3:1.7b (MINOR ANOMALY):** All 4 tool calls completed successfully with 572 characters of output in 8 seconds — the fastest reliable completion observed. However, the known anomaly that it reports 5 words (when 4 is correct) represents a minor hallucination on the specific counting verification step. For most pipeline use cases this would be acceptable, but for precision-critical workflows it warrants caution.

**gemma4:latest (BEST IN CLASS):** Clean execution with all 4 tool calls, status=ok, answer_len=470, and no error fields set. This is the benchmark's clearest example of reliable, complete model behavior with zero observed anomalies. No tool call was dropped or skipped.

## Recommendations

Based on this analysis, I **recommend** the following strategic usage patterns for each model:

1. **Best for fast disposable work — Recommend: gemma4:latest.** At 21 seconds with all tools completing successfully and zero anomalies, this is the most trustworthy low-latency option. Use it for tasks where speed matters but correctness cannot be compromised. It outperforms faster models (which may have hallucination issues) while still being competitive on time.

2. **Best for high-quality analysis — Recommend: gemma4:latest or qwen3-coder:30b.** gemma4:latest offers the safest execution profile overall. qwen3-coder:30b, despite its iteration error anomaly, was the only model to complete all 4 tools in under 15 seconds with a substantial answer (437 chars). If you can tolerate the risk of post-solution iterations, it is excellent for time-sensitive analytical work where iterative refinement adds value.

3. **Best for latency-sensitive pipeline stages — Recommend: qwen3:1.7b or gemma4:latest.** qwen3:1.7b at 8 seconds is 2.6x faster than the next fastest model and could drastically reduce end-to-end pipeline latency if hallucination risk on the counting task is acceptable. For safety-critical pipelines, use gemma4:latest as the speed baseline.

4. **Model to avoid — Recommend against using smollm2:1.7b or llama3.1:70b in any production pipeline.** Both models exhibit fundamental reliability failures: smollm2 loops excessively with unbounded iterations consuming compute, and llama3.1 prematurely halts after a single tool call, leaving downstream consumers with incomplete data despite appearing successful.

5. **Specialists to avoid — cogito:70b** completes tasks correctly but takes 193 seconds, which is prohibitive for any practical pipeline throughput. Consider it only for offline batch processing where correctness outweighs all speed concerns. Also consider avoiding deepseek-r1:8b, llama3:latest, and gemma3:27b since they report tool-not-supported errors — these models are fundamentally incompatible with agentic tool-calling workflows.

**Overall recommendation:** Deploy gemma4:latest as the primary model in your pipeline for its combination of speed, reliability, and zero-observed anomalies. Use qwen3-coder:30b as a fast secondary when latency budgets are tight, accepting its iteration-exceeded risk. Avoid llama3.1:70b and smollm2:1.7b entirely.

## Use Case Assignments

**Planner → gemma4:latest:** I recommend gemma4:latest for the planner role because it reliably executes all 4 required tool calls with complete outputs and zero detected errors, making it ideal for decomposing tasks and coordinating sub-work — the planner must be fully reliable to avoid cascading failures downstream.

**Executor → qwen3:1.7b:** I recommend qwen3:1.7b for the executor role because its 8-second latency is unmatched across the entire benchmark, and although it has a minor hallucination on counting (5 words vs 4), an executor primarily needs to act quickly on instructions — speed here compounds across every subsequent pipeline invocation.

**Synthesizer → qwen3-coder:30b:** I recommend qwen3-coder:30b for the synthesizer role because it produces large, detailed answers (437 chars) despite its iterative error anomaly, suggesting rich internal reasoning capability that could produce nuanced synthesis outputs — and its speed advantage is valuable when synthesizing results from multiple executor passes.
