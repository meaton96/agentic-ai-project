# `Simulation/FactoryLayout/FactoryLayoutManager.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `FactoryLayoutManager` class, responsible for generating and managing the factory floor layout in a simulation environment. It handles machine placement, aisle creation, and navigation boundary setup. The class plays a crucial role in the project by providing a structured and efficient way to design and optimize factory layouts.

## Dependencies
* `UnityEngine`: for Unity-specific functionality, such as instantiating game objects and manipulating transforms.
* `Unity.AI.Navigation`: for NavMesh-related functionality, including building and updating navigation boundaries.
* `Assets.Scripts.Simulation.Machines`: for machine-related data and functionality, including machine prefabs and capabilities.
* `Assets.Scripts.Simulation.Jobs`: for job-related data and functionality, including job scheduling and routing.

## Classes
### `FactoryLayoutManager`
One-line role description: Manages the factory floor layout, including machine placement, aisle creation, and navigation boundary setup.
Inheritance note: Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `BuildFloor` | `Dictionary<MachineType, List<int>> BuildFloor(FJSSPConfig config)` | Generates the factory floor layout based on the provided configuration. |
| `SampleCapabilities` | `HashSet<MachineType> SampleCapabilities(MachineType primary, FJSSPConfig config, MachineType[] allTypes)` | Samples a set of machine capabilities for a given machine type and configuration. |
| `BuildInfrastructure` | `void BuildInfrastructure(Vector3 floorCentre)` | Instantiates I/O conveyors and the AGV parking zone. |
| `ComputeDistanceMatrix` | (not shown) | Computes the spatial distance matrix for the factory floor layout. |

## Notes
The `FactoryLayoutManager` class uses a combination of Unity-specific functionality and custom simulation logic to generate and manage the factory floor layout. The `BuildFloor` method is the main entry point for generating the layout, and it relies on various other methods and dependencies to perform its tasks. The class also uses a number of configuration parameters and prefabs to customize the layout and behavior of the simulation.