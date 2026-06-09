# `Camera/OrbitCamera.cs`
**Language:** C# | **Lines:** 49 | **Last Scanned:** 2026-06-04

## Purpose
This file implements an orbiting camera system, allowing the camera to continuously rotate around a target object. The camera's position and rotation are updated in the `LateUpdate` method to achieve a smooth orbiting effect. This system is useful for creating dynamic camera movements in 3D applications.

## Dependencies
* `UnityEngine`: provides essential classes and functions for Unity game development, including `Transform`, `MonoBehaviour`, and `Quaternion`.

## Classes
The `OrbitCamera` class is responsible for managing the camera's orbit around a target object. It inherits from `MonoBehaviour`, which is the base class for all Unity scripts.
| Method | Signature | Description |
|--------|-----------|-------------|
| `LateUpdate` | `void LateUpdate()` | Updates the camera's position and rotation to maintain a smooth orbit around the target object. |

## Notes
The camera's orbit speed is unaffected by Unity's `Time.timeScale`, ensuring a consistent rotation speed regardless of the game's simulation speed. The camera's look-at target is also adjusted to account for the `heightOffset`, ensuring the subject remains centered in the frame. If no target is assigned, the camera will log a warning and skip movement.