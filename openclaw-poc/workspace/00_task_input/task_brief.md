# Task Brief: Benchmark Analysis

You are analyzing the results of an agentic tool-calling benchmark run against
the RIT API model fleet. Every model was given the same multi-step task
requiring 4 tool calls (count_words, calculator, get_current_datetime,
lookup_fact). The correct math answer was 16.0.

A model row is HEALTHY when: status=ok, tool_calls=4, answer_len > 0, and the
answer preview shows the correct values (word count 4, sum 16).

Known anomalies to look for:
- qwen3:1.7b reported 5 words (hallucination) despite status=ok
- qwen3.5:35b-a3b-coding-nvfp4 returned an empty final answer (answer_len=0)
- llama3.1:70b stopped after 1 tool call and never finished the task
- smollm2:1.7b looped 22 tool calls without producing an answer

## Benchmark Data (CSV)

```csv
model,status,error,tool_calls,iterations,time_s,answer_len
gemma4:26b,ok,,4,5,45.02,413
gemma4:latest,ok,,4,5,21.06,470
qwen3.6:35b-a3b-nvfp4,ok,,4,4,24.04,361
cogito:70b,ok,,4,5,193.61,384
qwen3.5:35b-a3b-coding-nvfp4,ok,,4,5,51.4,0
qwen3.5:latest,ok,,4,4,35.23,387
qwen3:32b,error,504 gateway timeout,0,1,91.51,0
llama3.1:70b,ok,,1,2,106.36,112
smollm2:1.7b,error,max_iterations_exceeded,22,10,29.27,45
qwen3-coder:30b,ok,,4,2,14.26,437
deepseek-r1:8b,error,does not support tools,0,1,26.55,0
devstral:latest,ok,,4,5,47.29,324
gpt-oss:120b,ok,,4,5,62.21,348
qwen3:latest,ok,,4,2,20.56,379
qwen3:1.7b,ok,,4,2,8.2,572
qwen3:8b,ok,,4,2,17.25,347
llava:latest,error,does not support tools,0,1,0.06,0
qwen3-coder-next:latest,ok,,4,2,31.54,399
qwen3.5:27b,ok,,4,5,59.77,375
gemma3:27b,error,does not support tools,0,1,0.15,0
```

Follow the instructions in your pipeline-stage prompt. This file is the only
input you need.
