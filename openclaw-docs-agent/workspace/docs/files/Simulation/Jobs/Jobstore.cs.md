# `Simulation/Jobs/Jobstore.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `JobStore` class, which serves as a central repository for managing job data in the simulation. It provides methods for initializing, adding, and retrieving job data, as well as querying the state of jobs. The `JobStore` class plays a crucial role in the simulation by acting as a passive data store that translates static job definitions into live runtime trackers.

## Dependencies
* `System.Collections.Generic`: for using generic collections such as `List<T>` and `HashSet<T>`.
* `System.Linq`: for using LINQ queries to filter and manipulate job data.
* `UnityEngine`: for using Unity-specific features such as `MonoBehaviour` and `Transform`.

## Classes
### JobStore
The `JobStore` class is responsible for managing job data in the simulation. It inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Initialize | `void Initialize(IEnumerable<FJSSPJobDefinition> definitions, bool spawnVisuals)` | Converts static job definitions into runtime state trackers. |
| AddDynamicJob | `JobData AddDynamicJob(FJSSPJobDefinition def, bool spawnVisuals)` | Inserts a single dynamically-arrived job into the store mid-episode. |
| Cleanup | `void Cleanup()` | Destroys all associated `JobVisual` objects and clears the internal list. |
| Get | `JobData Get(int jobId)` | Retrieves a specific `JobData` instance by its unique identifier. |
| GetNextNeedsRouting | `JobData GetNextNeedsRouting()` | Finds the first job that requires a routing decision. |
| GetNextUnassignedPickup | `JobData GetNextUnassignedPickup()` | Finds the first job waiting for pickup that has not yet been assigned an AGV. |
| GetDispatchableJobs | `List<int> GetDispatchableJobs(int machineId)` | Returns a list of job IDs currently queued and ready for processing at a specific machine. |
| HasDispatchableJob | `bool HasDispatchableJob(int machineId)` | Checks if any jobs are currently queued at the specified machine. |
| AreAllExited | `bool AreAllExited()` | Determines if all jobs in the simulation have reached the `Exited` state. |
| CountInState | `int CountInState(JobState state)` | Returns the total count of jobs currently in the specified `JobState`. |
| GetMachineLoad | `float GetMachineLoad(int machineId)` | Calculates the total remaining processing load for a specific machine. |
| GetProcessingTime | `float GetProcessingTime(int jobId, int machineId)` | Retrieves the estimated processing time for a specific job on a specific machine. |

## Notes
The `JobStore` class uses a `List<JobData>` to store job data, which is initialized and populated through the `Initialize` method. The class also uses a `HashSet<int>` to keep track of deferred job IDs. The `GetMachineLoad` method uses LINQ queries to calculate the total remaining processing load for a specific machine.