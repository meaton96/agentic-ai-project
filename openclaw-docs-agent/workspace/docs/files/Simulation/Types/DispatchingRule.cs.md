# `Simulation/Types/DispatchingRule.cs`
**Language:** C# | **Lines:** 52 | **Last Scanned:** 2026-06-04

## Purpose
This file defines an enumeration of dispatching rules used to prioritize jobs in the dispatching engine. The rules determine the sorting criterion for candidate jobs when an idle machine becomes available. Each rule is designed to optimize specific aspects of job scheduling, such as processing time, remaining work, or due time.

## Enums
The `DispatchingRule` enum defines the following rules:
| Rule | Description |
|------|-------------|
| SPT_SMPT | Shortest Processing Time with Smallest Most Urgent Remaining Time secondary metric |
| SPT_SRWT | Shortest Processing Time with Shortest Remaining Work Time secondary metric |
| LPT_MMUR | Longest Processing Time with Smallest Most Urgent Remaining Time secondary metric |
| LPT_SMPT | Longest Processing Time with Smallest Most Urgent Remaining Time secondary metric |
| SRT_SRWT | Shortest Remaining Time with Shortest Remaining Work Time secondary metric |
| SRT_SMPT | Shortest Remaining Time with Smallest Most Urgent Remaining Time secondary metric |
| LRT_MMUR | Longest Remaining Time with Smallest Most Urgent Remaining Time secondary metric |
| SDT_SRWT | Smallest Due Time with Shortest Remaining Work Time secondary metric |
| Random | Unweighted random selection of queued jobs |

## Notes
The dispatching rules are designed to be used by the `DispatchingEngine` to order candidate jobs. Each rule has a primary sort key and may have a secondary tiebreaker. The rules can be used to optimize different aspects of job scheduling, such as reducing work-in-progress (WIP), increasing throughput, or meeting due dates.