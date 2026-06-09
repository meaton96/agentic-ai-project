# `UI/SimulationHUD.cs`
**Language:** C# | **Lines:** 124 | **Last Scanned:** 2024-07-26

## Purpose
This file implements the SimulationHUD class, which serves as a heads-up display for the simulation, providing key episode metrics to the player. It displays simulation time, last applied rule, decision count, and completed jobs, and allows for time scale control. The class plays a crucial role in the project by providing a user interface for simulation monitoring and control.

## Dependencies
* `UnityEngine`: for Unity engine functionality
* `TMPro`: for text rendering
* `Assets.Scripts.Simulation`: for simulation-related functionality
* `Assets.Scripts.Simulation.Jobs`: for job-related functionality
* `Assets.Scripts.Simulation.Logging`: for logging functionality

## Classes
The `SimulationHUD` class is a `MonoBehaviour` that manages the simulation heads-up display.
| Method | Signature | Description |
|--------|-----------|-------------|
| Start | `void Start()` | Initializes the time scale slider and registers event listeners. |
| OnDestroy | `void OnDestroy()` | Removes event listeners to prevent stale callbacks. |
| OnStopClicked | `void OnStopClicked()` | Stops the active episode and returns to the instance select menu. |
| Update | `void Update()` | Polls the simulation bridge and updates the HUD labels. |
| OnTimeScaleChanged | `void OnTimeScaleChanged(float newScale)` | Applies the new time scale to the simulation. |
| UpdateScaleText | `void UpdateScaleText(float scale)` | Updates the time scale label. |

## Notes
The `SimulationHUD` class relies on the `FactoryOrchestrator` instance to access simulation data and control the simulation. The time scale slider directly modifies `Time.timeScale`, which affects the simulation speed. The class uses Unity's event system to handle user input and updates the HUD labels once per frame.