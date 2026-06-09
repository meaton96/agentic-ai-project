# `UI/BillboardUI.cs`
**Language:** C# | **Lines:** 47 | **Last Scanned:** 2026-06-04

## Purpose
This file contains the `BillboardUI` class, which rotates a GameObject to face the main camera, making it suitable for world-space UI elements like floating nameplates or overhead labels. The class ensures the UI remains readable from the player's viewpoint. It uses the `LateUpdate` method to apply orientation after all camera movement has settled for the frame.

## Dependencies
* `UnityEngine`: provides access to Unity's core functionality, including camera and transform management, which is crucial for the `BillboardUI` class.

## Classes
The `BillboardUI` class: rotates a GameObject to face the main camera, inheriting from `MonoBehaviour`.
| Method | Signature | Description |
|--------|-----------|-------------|
| Start | `void Start()` | Caches the main camera's transform to avoid repeated lookups. |
| LateUpdate | `void LateUpdate()` | Orients the UI toward the camera after all movement updates have run, considering the `lockVertical` setting. |

## Notes
The `lockVertical` setting allows for rotation only on the Y-axis, which is useful for floating nameplates on sloped terrain. The `LateUpdate` method is used instead of `Update` to ensure the UI orientation is applied after all camera movement has settled for the frame.