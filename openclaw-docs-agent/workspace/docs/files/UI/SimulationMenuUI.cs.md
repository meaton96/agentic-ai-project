# `UI/SimulationMenuUI.cs`
**Language:** C# | **Lines:** 1200+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the UI logic for the simulation menu in a factory simulation project. It handles user input, configures the simulation, and updates the UI accordingly. The class `SimulationMenuUI` is responsible for managing the simulation menu and interacting with other components.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and UI components
* `Assets.Scripts.Simulation.Machines`: for machine-related data and logic
* `Assets.Scripts.Simulation.Types`: for simulation-related types and enumerations
* `Assets.Scripts.Simulation`: for simulation-related logic and components

## Classes
### `MachineTypeProcRow`
One-line role description: Represents a row in the processing time table for a machine type.
Inheritance note: No inheritance.
| Method | Signature | Description |
|--------|-----------|-------------|
| (none) |  |  |

### `SimulationMenuUI`
One-line role description: Manages the simulation menu and interacts with other components.
Inheritance note: Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `Awake` | `void` | Initializes the simulation menu |
| `Start` | `void` | Configures the simulation and updates the UI |
| `OnEnable` | `void` | Adds event listeners for factory events |
| `OnDisable` | `void` | Removes event listeners for factory events |
| `Update` | `void` | Updates the UI and checks for user input |
| `PopulateDefaults` | `void` | Fills input fields with default values |
| `WireCallbacks` | `void` | Sets up event listeners for UI components |
| `OnSpawnClicked` | `void` | Spawns the factory when the spawn button is clicked |
| `OnStartClicked` | `void` | Starts the simulation when the start button is clicked |
| `RefreshButtonStates` | `void` | Updates the state of UI buttons based on the simulation state |
| `OnFactorySpawned` | `void` | Updates the UI when the factory is spawned |
| `OnEpisodeFinished` | `void` | Updates the UI when the simulation episode is finished |
| `BuildConfig` | `FJSSPConfig` | Builds a simulation configuration based on user input |
| `BuildStochasticConfig` | `StochasticConfig` | Builds a stochastic configuration for the simulation |
| `ShowPanel` | `void` | Shows the simulation menu panel |
| `HidePanel` | `void` | Hides the simulation menu panel |
| `SetStatus` | `void` | Updates the status text in the UI |
| `UpdateTotalMachinesLabel` | `void` | Updates the total machines label in the UI |

## Notes
The `SimulationMenuUI` class uses a combination of Unity-specific components and custom logic to manage the simulation menu. The `BuildConfig` method is used to create a simulation configuration based on user input, and the `BuildStochasticConfig` method is used to create a stochastic configuration for the simulation. The class also uses event listeners to respond to factory events and update the UI accordingly.