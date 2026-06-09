# `Simulation/FactoryLayout/TrafficZoneManager.cs`
**Language:** C# | **Lines:** 12359 | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `TrafficZoneManager` class, which manages the zone-based traffic control network for the factory floor. It divides the floor into reservable segments to manage one-way flow and prevent deadlocks. The class is responsible for constructing the topological traffic network from the current factory layout.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and types
* `Assets.Scripts.Simulation.Logging`: for logging and error handling
* `Assets.Scripts.Simulation.FactoryLayout`: for factory layout management and related types

## Classes
### TrafficZone
Role: Represents a single reservable zone within the traffic network.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| (none) |  |  |

### DockPoint
Role: Describes positioning for AGV-conveyor interaction.
Inheritance: None
| Method | Signature | Description |
|--------|-----------|-------------|
| (none) |  |  |

### TrafficZoneManager
Role: Manages the zone-based traffic control network for the factory floor.
Inheritance: `MonoBehaviour`
| Method | Signature | Description |
|--------|-----------|-------------|
| `BuildZoneGraph` | `public void BuildZoneGraph()` | Constructs the topological traffic network from the current factory layout |
| `GetZone` | `public TrafficZone GetZone(int zoneId)` | Retrieves a zone by its unique ID |
| `RegisterZone` | `private void RegisterZone(TrafficZone zone)` | Registers a new zone in the traffic network |
| `ConnectZoneGraph` | `private void ConnectZoneGraph(int[][] rowAisles, int[] topSpine, int[] botSpine, int[] leftVert, int[] rightVert, int rows, int cols)` | Wires up downstream and upstream links between all zones to form a circulation loop |
| `BuildRowAisleZones` | `private int[][] BuildRowAisleZones(int rows, int cols)` | Segments row aisles into discrete zones based on machine columns |
| `BuildSpineZones` | `private int[] BuildSpineZones(bool isTop, int cols)` | Segments spine aisles (top/bottom peripheral) into zones |
| `BuildVerticalZones` | `private int[] BuildVerticalZones(bool isLeft, int rows)` | Segments vertical connector aisles (left/right) into zones |

## Functions
None

## Notes
The `TrafficZoneManager` class relies on the `FactoryLayoutManager` to provide the physical factory layout. The `BuildZoneGraph` method is the main entry point for constructing the traffic network, and it orchestrates the creation of zones, links, and dock points. The class uses a combination of lists and dictionaries to manage the traffic network, and it provides methods for retrieving zones by ID and registering new zones.