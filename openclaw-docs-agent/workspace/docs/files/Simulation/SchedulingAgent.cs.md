# `Simulation/SchedulingAgent.cs`
**Language:** C# | **Lines:** 550 | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `SchedulingAgent` class, a subclass of ML-Agents' `Agent`, which drives job-shop scheduling decisions in a simulation environment. It collects observations, maps discrete action indices to dispatching rules, and handles simulation events. The agent plays a crucial role in the project by enabling the simulation to make informed decisions based on the current state of the environment.

## Dependencies
* `Unity.MLAgents`: provides the foundation for the agent's decision-making and learning capabilities
* `Unity.MLAgents.Actuators` and `Unity.MLAgents.Sensors`: enable the agent to interact with the environment and collect observations
* `Assets.Scripts.Simulation.Logging`: provides logging functionality for debugging and monitoring purposes
* `Assets.Scripts.Simulation.Types`: defines custom types used in the simulation environment

## Classes
The `SchedulingAgent` class:
Inherits from `Agent`
| Method | Signature | Description |
|--------|-----------|-------------|
| ArmAndStart | `void` | Forces the agent into an active state and ends any current episode to trigger a reset |
| SetHeuristicRule | `void (DispatchingRule rule)` | Sets the rule used when the agent is running in Heuristic mode |
| Heuristic | `void (ActionBuffers actionsOut)` | Provides a baseline action based on a hardcoded dispatching rule |
| OnEnable | `void` | Subscribes to simulation events when the component is enabled |
| OnDisable | `void` | Unsubscribes from simulation events when the component is disabled |
| Initialize | `void` | Initializes the agent's internal state |
| OnEpisodeBegin | `void` | Prepares the simulation and internal state for a new episode |
| HandleDecisionRequired | `void (DecisionRequest req)` | Relays the decision requirement from the bridge to ML-Agents |
| HandleEpisodeFinished | `void (EpisodeRecord record)` | Handles the termination of a simulation run |
| CollectObservations | `void (VectorSensor sensor)` | Populates the ML-Agents observation vector with environment state data |
| PadZeros | `void (VectorSensor sensor)` | Fills the observation vector with zeros if the agent is inactive |
| OnActionReceived | `void (ActionBuffers actions)` | Processes the discrete action index returned by the neural network |

## Notes
The `SchedulingAgent` class relies heavily on the `FactoryOrchestrator` instance, which provides access to the simulation environment and its state. The agent's decision-making process is influenced by the `DispatchingRule` used, which can be set using the `SetHeuristicRule` method. The agent's logging functionality is controlled by the `logDecisions` flag.