# `Camera/AGVFollowCamera.cs`
**Language:** C# | **Lines:** 114 | **Last Scanned:** 2024-07-26

## Purpose
This file contains the `AGVFollowCamera` class, which attaches the camera to AGVs within the `AGVPool` and provides a smooth tracking experience. The class toggles control away from `CameraController` and `OrbitCamera` when active, allowing for automated AGV tracking. It plays a crucial role in the project by enabling camera control and tracking of AGVs.

## Dependencies
* `UnityEngine`: provides core Unity functionality, including camera and transform management.
* `Assets.Scripts.Simulation.AGV`: provides access to the `AGVPool` and `AGVController` classes, which manage the AGV fleet.
* `Assets.Scripts.Simulation.Logging`: provides logging functionality through the `SimLogger` class.

## Classes
The `AGVFollowCamera` class:
Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Awake | `void Awake()` | Initializes references to the sibling camera control components. |
| Update | `void Update()` | Polls for user input to toggle camera modes or swap AGV targets. |
| LateUpdate | `void LateUpdate()` | Performs the camera transformation updates after all movement logic is processed. |
| ToggleFollowMode | `void ToggleFollowMode()` | Switches the camera between manual navigation and automated AGV tracking. |
| ChangeAGV | `void ChangeAGV(int direction)` | Increments or decrements the current AGV target index. |

## Notes
The `AGVFollowCamera` class uses a smooth speed value to control the camera's movement, and it logs important events, such as mode changes and AGV target switches, using the `SimLogger` class. The class also handles edge cases, such as an empty `AGVPool` or invalid `currentAgvIndex` values.