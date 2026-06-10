# Model Profiling Report

## Profile 1: qwen3-coder:30b

**Speed Tier:** Fast (<25s)
**Actual Time:** 14.26 seconds

**Tool Usage:** Perfect - made exactly 4 tool calls and finished cleanly with no errors.

**Anomalies:** None observed. Completed all 4 iterations with the correct task outcome. Longest answer (437 chars) suggesting thorough analysis.

---

## Profile 2: qwen3:1.7b

**Speed Tier:** Fast (<25s)
**Actual Time:** 8.2 seconds (FASTEST OF ALL MODELS TESTED)

**Tool Usage:** Perfect - made exactly 4 tool calls and finished cleanly.

**Anomalies:** Slightly elevated answer length (572 chars vs ~350-400 range for others), but all other metrics indicate correct performance. This may indicate more verbose reasoning or additional context in the final answer.

---

## Profile 3: cogito:70b

**Speed Tier:** Slow (>65s)
**Actual Time:** 193.61 seconds (SLOWEST PASSING MODEL)

**Tool Usage:** Perfect - made exactly 4 tool calls and finished cleanly.

**Anomalies:** None observed. Despite being over 4x slower than other models, it completed all required tool calls correctly. May indicate thorough verification or complex internal reasoning processes.

---

## Profile 4: qwen3.5:35b-a3b-coding-nvfp4

**Speed Tier:** Medium (25-65s)
**Actual Time:** 51.4 seconds

**Tool Usage:** Made 4 tool calls but FAILED to produce output.

**Anomalies:** **CRITICAL ANOMALY** - The answer_len is 0, meaning the model made all 4 required tool calls but returned an empty final answer. This is a complete task failure despite having correct intermediate steps. The model likely failed during the final synthesis stage.

---

## Profile 5: smollm2:1.7b

**Speed Tier:** Medium (25-65s)
**Actual Time:** 29.27 seconds

**Tool Usage:** FAILED - made 22 tool calls (far exceeding the expected 4).

**Anomalies:** **CRITICAL ANOMALY** - Excessive iteration loop. Expected 4 tool calls, but the model made 22 iterations without producing a valid answer. This indicates a runaway loop in the agent's decision-making process. Despite the relatively fast individual tool call speed, the excessive calls make it unreliable.
