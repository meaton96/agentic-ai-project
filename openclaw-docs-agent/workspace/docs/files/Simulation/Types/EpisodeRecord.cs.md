# `Simulation/Types/EpisodeRecord.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
This file contains classes for recording and analyzing simulation episodes, providing a snapshot of episode statistics and metrics. It serves as the single source of truth for post-episode analytics and CSV export. The classes in this file are used to track various aspects of simulation episodes, including machine and AGV performance, congestion, and disruption metrics.

## Dependencies
* `System.Collections.Generic`: for using generic collections such as List<T>
* `Assets.Scripts.Simulation.Types`: for using custom types such as StochasticConfig

## Classes
### EpisodeRecord
Role: Immutable snapshot of a completed episode's statistics.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| OptimalityGap | `double OptimalityGap { get; }` | Calculates the optimality gap as a percentage |

### MachineRecord
Role: Per-machine statistics for one episode.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| UtilizationRate | `double UtilizationRate { get; }` | Calculates the utilization rate of the machine |
| IdleTime | `double IdleTime { get; }` | Calculates the idle time of the machine |
| IdleRate | `double IdleRate { get; }` | Calculates the idle rate of the machine |

### AGVRecord
Role: Per-AGV time budget and throughput for one episode.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| ProductiveTime | `double ProductiveTime { get; }` | Calculates the productive time of the AGV |
| TotalAccountedTime | `double TotalAccountedTime { get; }` | Calculates the total accounted time of the AGV |
| CongestionFraction | `double CongestionFraction { get; }` | Calculates the congestion fraction of the AGV |

### SegmentRecord
Role: Per-zone congestion metrics for one episode.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| MeanBlockTime | `float MeanBlockTime { get; }` | Calculates the mean block time of the zone |
| BlockRate | `float BlockRate { get; }` | Calculates the block rate of the zone |

## Notes
The classes in this file are designed to be immutable, with properties and methods that calculate derived metrics on access. The EpisodeRecord class serves as the central hub for episode statistics, with references to lists of MachineRecord, AGVRecord, and SegmentRecord objects. The use of derived properties and methods allows for efficient calculation of complex metrics without storing redundant data.