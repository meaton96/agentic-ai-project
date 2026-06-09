# `Simulation/Channels/EpisodeConfigChannel.cs`
**Language:** C# | **Lines:** 147 | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `EpisodeConfigChannel` class, which receives and processes configuration data from Python for each episode reset in a simulation. It plays a crucial role in integrating Python and C# components of the project. The class enables the consumption of configuration data, allowing for dynamic adjustments to the simulation environment.

## Dependencies
* `Unity.MLAgents.SideChannels`: for side channel communication with Python
* `Newtonsoft.Json`: for JSON deserialization of configuration data
* `Assets.Scripts.Simulation.Types`: for simulation-specific data types (e.g., `FJSSPConfig`)

## Classes
The `EpisodeConfigChannel` class is responsible for receiving and processing configuration data. It inherits from `SideChannel`.
| Method | Signature | Description |
|--------|-----------|-------------|
| `OnMessageReceived` | `void (IncomingMessage msg)` | Called by ML-Agents when a message is received from Python; deserializes the JSON payload into a `FJSSPConfig` |
| `ConsumeConfig` | `FJSSPConfig ()` | Returns and clears the pending configuration; returns null if no configuration has been received |
| `DeserialiseConfig` | `FJSSPConfig (string json)` | Deserializes a JSON string into a `FJSSPConfig` object |

## Notes
The `EpisodeConfigChannel` class uses a lock (`_lock`) to ensure thread safety when accessing the pending configuration. The `DeserialiseConfig` method performs extensive JSON deserialization, which may be a performance bottleneck for large configuration data. The class also logs important events, such as receiving a new configuration, using the `SimLogger` class.