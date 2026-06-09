# `Scheduling/Data/TaillardInstance.cs`
**Language:** C# | **Lines:** 91 | **Last Scanned:** 2026-06-04

## Purpose
This file contains classes for representing Taillard benchmark instances, used in scheduling simulations. It provides data structures for deserializing JSON benchmark data and accessing job operations. The classes in this file play a crucial role in loading and processing scheduling instances.

## Dependencies
* `System`: provides fundamental data types and serialization utilities.

## Classes
### TaillardInstance
Role: represents a Taillard benchmark instance with job operations and metadata.
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| GetJobOperations | `(int machine, int duration)[] GetJobOperations(int jobIndex)` | Returns the ordered operation sequence for a single job as typed value tuples. |

### TaillardMetadata
Role: represents known optimality bounds and bibliographic metadata for a Taillard instance.
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| None |  | This class only contains fields for storing metadata. |

## Notes
The `TaillardInstance` class uses Unity's `JsonUtility` or `System.Text.Json` for deserialization, with field names matching the JSON keys exactly. The `GetJobOperations` method is used by `DESSimulator.LoadInstance` to construct `Operation` objects. The `TaillardMetadata` class is used for post-simulation validation, comparing the achieved makespan against known bounds.