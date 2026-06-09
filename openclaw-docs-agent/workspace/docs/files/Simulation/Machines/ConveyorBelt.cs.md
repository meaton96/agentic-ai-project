# `Simulation/Machines/ConveyorBelt.cs`
**Language:** C# | **Lines:** 734 | **Last Scanned:** 2026-06-04

## Purpose
This file implements a linear conveyor belt simulation, allowing jobs to be smoothly moved between an input end and an output end. The conveyor belt manages a fixed-capacity list of belt entries, each representing a job visual positioned along the belt spine. The component is designed to work within a Unity environment, utilizing Unity's MonoBehaviour and Update loop.

## Dependencies
* `UnityEngine`: for Unity-specific functionality, such as MonoBehaviour and Update loop.
* `Assets.Scripts.Simulation.Jobs`: for job-related classes and functionality, including JobVisual.

## Classes
### ConveyorBelt
Role: Manages a linear conveyor belt simulation, handling job visuals and their movement.
Inheritance: Inherits from MonoBehaviour.
| Method | Signature | Description |
|--------|-----------|-------------|
| DumpBeltJobs | `string DumpBeltJobs()` | Returns a string representation of the current belt jobs. |
| GetSlotWorldPosition | `Vector3 GetSlotWorldPosition(int slotIndex)` | Calculates the world-space coordinate for a specific belt slot. |
| GetTargetForEntry | `Vector3 GetTargetForEntry(int entryIndex)` | Maps a list entry index to its target world slot position. |
| TryEnqueue | `bool TryEnqueue(int jobId, JobVisual visual)` | Attempts to place a job at the input end of the belt. |
| PeekFront | `int PeekFront()` | Retrieves the ID of the job at the output end without removing it. |
| PeekFrontVisual | `JobVisual PeekFrontVisual()` | Retrieves the visual of the job at the output end without removing it. |
| DequeueFront | `(int jobId, JobVisual visual) DequeueFront()` | Removes and returns the job at the output end of the belt. |
| RemoveJob | `JobVisual RemoveJob(int jobId)` | Removes a specific job ID from any position on the belt. |
| Contains | `bool Contains(int jobId)` | Checks if a specific job ID is currently managed by this belt. |
| GetJobIds | `List<int> GetJobIds()` | Generates an ordered list of all job IDs currently on the belt. |
| Clear | `void Clear()` | Forcefully clears all jobs from the belt. |
| RecalculateTargets | `void RecalculateTargets()` | Updates target positions for all entries based on their current list index. |
| Update | `void Update()` | Unity Update loop driving the smooth sliding of items along the belt. |
| OnDrawGizmos | `void OnDrawGizmos()` | Renders debug gizmos for the belt spine, slots, and flow direction in the Unity Editor. |

## Notes
The conveyor belt's flow direction is determined by the `reverseFlow` flag, which toggles the input and output ends. The `beltSpeed` variable controls the speed at which jobs move along the belt. The `Update` method is responsible for smoothly moving jobs toward their target positions, while the `OnDrawGizmos` method provides visual debugging information in the Unity Editor.