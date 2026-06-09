# `Simulation/Machines/MachineVisual.cs`
**Language:** C# | **Lines:** 5414 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `MachineVisual` class, responsible for managing the visual representation of a machine in a factory simulation. It handles the machine's appearance, including body color, indicator light, and overhead UI labels. The class is paired with `PhysicalMachine`, which handles the logical state machine.

## Dependencies
* `UnityEngine`: for Unity-specific functionality, such as rendering and UI management.
* `Assets.Scripts.Simulation.Types`: for simulation-specific types, including `MachineType` and `MachineState`.
* `Assets.Scripts.Simulation.Logging`: for logging functionality.

## Classes
### MachineVisual
Manages the visual representation of a machine in the factory simulation. Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Awake | `void Awake()` | Initializes the machine's visual state. |
| OnDestroy | `void OnDestroy()` | Cleans up instance materials when the component is destroyed. |
| Initialise | `void Initialise(int id, MachineType type)` | Initializes the machine with its identity and type color. |
| SetState | `void SetState(MachineState newState)` | Transitions the machine to a new operational state. |
| BeginOperation | `void BeginOperation(int jobId, float simStartTime, float duration)` | Notifies the machine that it has begun processing a job. |
| CompleteOperation | `void CompleteOperation(int jobId)` | Notifies the machine that it has completed processing a job. |
| SetBlockedAfterProcessing | `void SetBlockedAfterProcessing(int jobId)` | Notifies the machine that it is blocked after processing. |
| UpdateProgress | `void UpdateProgress(float normalizedProgress)` | Updates the progress bar to the specified normalized value. |
| UpdateIncomingQueueLabel | `void UpdateIncomingQueueLabel(int count)` | Updates the incoming queue count label. |
| UpdateOutgoingQueueLabel | `void UpdateOutgoingQueueLabel(int count)` | Updates the outgoing queue count label. |
| RecordDecisionPoint | `void RecordDecisionPoint(float simTime, int[] queuedJobIds, int chosenJobId, string ruleName, bool flash = true)` | Records a scheduling decision point and optionally flashes the indicator. |

## Notes
The `MachineVisual` class uses a dictionary to map machine types to their unique identity colors. The `Initialise` method sets the machine's label and applies the type-specific body color. The `SetState` method updates the indicator light material and status text label based on the new operational state. The `RecordDecisionPoint` method logs the scheduling decision and triggers a brief white flash of the indicator light if specified.