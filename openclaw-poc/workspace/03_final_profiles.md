# Model Profiles

## Profile 1: qwen3:1.7b
- **Speed tier:** fast (8.2s)
- **Tool usage:** 4 tool calls, completed cleanly
- **Anomalies:** high word count (572) but otherwise clean execution

## Profile 2: cogito:70b
- **Speed tier:** slow (193.61s)
- **Tool usage:** 4 tool calls, completed cleanly
- **Anomalies:** none observed despite lengthy task completion

## Profile 3: qwen3.5:35b-a3b-coding-nvfp4
- **Speed tier:** medium (51.4s)
- **Tool usage:** 4 tool calls made
- **Anomalies:** answer length is 0 bytes (empty final answer) despite making all tool calls

## Profile 4: qwen3:8b
- **Speed tier:** fast (17.25s)
- **Tool usage:** 4 tool calls, completed cleanly
- **Anomalies:** none observed

## Profile 5: qwen3.6:35b-a3b-nvfp4
- **Speed tier:** medium (24.04s)
- **Tool usage:** 4 tool calls, completed cleanly
- **Anomalies:** none observed, produced large answer (361 words)