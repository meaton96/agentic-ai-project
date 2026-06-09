# `Simulation/Jobs/JobVisual.cs`
**Language:** C# | **Lines:** 276 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `JobVisual` class, which represents a visual token for a single job on the factory floor. It handles the job's lifecycle state, movement, and rendering. The class is responsible for smoothly interpolating the job's position and updating its color to reflect its current state.

## Dependencies
* `UnityEngine`: provides core Unity functionality, including rendering, transforms, and physics.

## Classes
### JobVisual
Role: Visual representation of a job on the factory floor. 
Inherits from: `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Initialize | `void Initialize(int id, int opCount)` | Initializes the visual token and caches rendering components. |
| SetState | `void SetState(JobState state)` | Updates the token's tint to reflect its current lifecycle state. |
| SetTargetPosition | `void SetTargetPosition(Vector3 worldPos)` | Defines a destination for the smooth interpolation logic. |
| SnapToPosition | `void SnapToPosition(Vector3 worldPos)` | Instantly teleports the token to a position with no interpolation. |
| SetOnConveyor | `void SetOnConveyor(bool on)` | Toggles external movement control by a conveyor belt. |
| AttachToCarrier | `void AttachToCarrier(Transform carrier)` | Parents the token to an AGV carrier for physical transport. |
| DetachFromCarrier | `void DetachFromCarrier(Vector3 worldSnapPos)` | Detaches the token from an AGV and returns it to world space. |

## Notes
The `JobVisual` class uses a `MaterialPropertyBlock` to update the token's color without creating material instances, preserving GPU instancing. The class also handles movement ownership, switching between self-driven, carried, or conveyor-driven movement. The `Update` method drives the self-driven position interpolation, skipping processing if the job is carried or on a conveyor.