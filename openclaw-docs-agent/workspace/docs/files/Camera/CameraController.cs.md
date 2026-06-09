# `Camera/CameraController.cs`
**Language:** C# | **Lines:** 118 | **Last Scanned:** 2026-06-04

## Purpose
This file controls the camera behavior in a Unity application, allowing the user to toggle between an orbit mode and a free-look mode. The `CameraController` class is responsible for handling user input and updating the camera's position and rotation accordingly. It requires an `OrbitCamera` component to be attached to the same GameObject.

## Dependencies
* `UnityEngine`: provides core Unity functionality, including GameObjects and components.
* `UnityEngine.InputSystem`: enables input processing and handling.

## Classes
The `CameraController` class: controls the camera behavior and handles user input. It inherits from `MonoBehaviour`, which is a Unity base class for scripts that can be attached to GameObjects.
| Method | Signature | Description |
|--------|-----------|-------------|
| Start | `void Start()` | Initializes the camera controller and sets the initial state. |
| Update | `void Update()` | Handles input processing per frame to toggle camera modes and update free-look mechanics. |
| ToggleCameraMode | `void ToggleCameraMode()` | Toggles the camera state between Orbit and FreeLook modes, synchronizing rotation values. |
| HandleFreeLook | `void HandleFreeLook()` | Processes mouse input to adjust the camera's pitch and yaw when in FreeLook mode. |
| HandleMovement | `void HandleMovement()` | Processes keyboard input to translate the camera through space when in FreeLook mode. |

## Notes
The `CameraController` class uses the `RequireComponent` attribute to ensure that an `OrbitCamera` component is attached to the same GameObject. The `Start` method initializes the camera controller and sets the initial state, while the `Update` method handles input processing per frame. The `ToggleCameraMode`, `HandleFreeLook`, and `HandleMovement` methods are used to update the camera's position and rotation based on user input.