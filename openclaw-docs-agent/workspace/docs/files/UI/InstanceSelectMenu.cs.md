# `UI/InstanceSelectMenu.cs`
**Language:** C# | **Lines:** 191 | **Last Scanned:** 2026-06-04

## Purpose
This file implements the `InstanceSelectMenu` class, which provides a startup menu for selecting a Taillard instance before the simulation begins. The menu loads instance assets, populates a dropdown, and allows the user to confirm their selection and start the simulation. It plays a crucial role in configuring the simulation environment.

## Dependencies
* `UnityEngine`: for Unity-specific functionality and UI components
* `TMPro`: for text rendering and UI elements
* `Assets.Scripts.Simulation`: for simulation-related classes and interfaces
* `Newtonsoft.Json`: for JSON serialization and deserialization

## Classes
The `InstanceSelectMenu` class:
* Role: Provides a startup menu for selecting a Taillard instance
* Inheritance: Inherits from `MonoBehaviour`
| Method | Signature | Description |
|--------|-----------|-------------|
| `Start` | `void Start()` | Loads instance assets, populates the dropdown, and sets up UI event listeners |
| `OnDestroy` | `void OnDestroy()` | Removes UI event listeners to prevent stale callbacks |
| `Show` | `void Show()` | Re-shows the menu and refreshes the dropdown |
| `OnStartClicked` | `void OnStartClicked()` | Pushes the selected instance to the bridge and starts the episode |
| `OnSelectionChanged` | `void OnSelectionChanged(int index)` | Refreshes the preview label when the dropdown selection changes |
| `LoadInstances` | `void LoadInstances()` | Loads every `TextAsset` found under `Resources/Instances` |
| `PopulateDropdown` | `void PopulateDropdown()` | Clears and repopulates the dropdown with the loaded instance names |
| `UpdatePreview` | `void UpdatePreview(int index)` | Parses the instance at the given index and writes a summary to the preview text |

## Notes
The `InstanceSelectMenu` class relies on the `SimulationBridge` to configure and start the simulation. It also uses JSON serialization to parse instance metadata. The `UpdatePreview` method assumes that the instance metadata is in a specific format, which may need to be adjusted if the format changes.