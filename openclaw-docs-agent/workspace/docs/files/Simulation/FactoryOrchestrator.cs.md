# `Simulation/FactoryOrchestrator.cs`
**Language:** C# | **Lines:** 12000+ | **Last Scanned:** 2026-06-04

## Purpose
The `FactoryOrchestrator` class is the central orchestrator for the factory simulation, managing episode lifecycle, coordinating machine/AGV/job systems, and interfacing with the learning agent for dispatch decision-making. It plays a crucial role in the project by integrating various components and enabling the simulation to run. The class is responsible for initializing and controlling the simulation environment.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and scene management
* `Unity.MLAgents`: for integration with the machine learning agent
* `Assets.Scripts.Simulation`: for simulation-specific classes and utilities, such as `FactoryLayoutManager`, `TrafficZoneManager`, and `AGVPool`

## Classes
### FactoryOrchestrator
One-line role description: Manages the factory simulation environment and coordinates episode lifecycle.
Inheritance note: Inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `LoadConfig` | `public void LoadConfig(FJSSPConfig config)` | Loads a simulation configuration and initializes the stochastic event manager. |
| `LoadPrebuiltJobs` | `public void LoadPrebuiltJobs(FJSSPJobDefinition[] jobs)` | Stores pre-built job definitions for use during the next episode start. |
| `SpawnFactory` | `public void SpawnFactory()` | Spawns the factory layout by building the floor plan, constructing the traffic zone graph, and initializing the AGV fleet. |
| `StartEpisode` | `public void StartEpisode()` | Starts a new simulation episode, initializing jobs, stochastic events, failure coordination, and decision systems. |
| `GetRuleIndex` | `public int GetRuleIndex(DispatchingRule rule)` | Converts a dispatching rule enum value to its corresponding action index. |

## Notes
The `FactoryOrchestrator` class uses a singleton pattern to ensure only one instance exists throughout the simulation. It also employs various events, such as `OnDecisionRequired`, `OnStepCompleted`, and `OnEpisodeFinished`, to notify other components of significant events during the simulation. The class relies heavily on other simulation-specific classes, such as `FactoryLayoutManager`, `TrafficZoneManager`, and `AGVPool`, to manage the simulation environment.