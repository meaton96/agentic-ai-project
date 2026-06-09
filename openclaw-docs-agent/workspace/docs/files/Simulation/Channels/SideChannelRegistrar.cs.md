# `Simulation/Channels/SideChannelRegistrar.cs`
**Language:** C# | **Lines:** 27 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `SideChannelRegistrar` class, which is responsible for registering and unregistering side channels in the Unity ML-Agents framework. It plays a crucial role in managing the communication between the simulation environment and the ML-Agents. The registrar ensures that the side channels are properly set up and torn down during the simulation lifecycle.

## Dependencies
* `Unity.MLAgents`: provides the ML-Agents framework and its components, including side channels.
* `Unity.MLAgents.SideChannels`: offers specific side channel implementations, such as `EpisodeConfigChannel` and `EpisodeTelemetryChannel`.
* `UnityEngine`: provides the `MonoBehaviour` class, which the `SideChannelRegistrar` inherits from.

## Classes
The `SideChannelRegistrar` class is a `MonoBehaviour` that manages the registration and unregistration of side channels. It inherits from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Awake | `void Awake()` | Initializes the side channels and registers them with the `SideChannelManager`. |
| OnDestroy | `void OnDestroy()` | Unregisters the side channels from the `SideChannelManager` when the simulation is destroyed. |

## Notes
The `SideChannelRegistrar` uses the `SideChannelManager` to manage the registration and unregistration of side channels. This ensures that the side channels are properly set up and torn down during the simulation lifecycle, which is essential for maintaining a stable and efficient communication between the simulation environment and the ML-Agents.