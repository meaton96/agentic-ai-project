# `Scheduling/Validation/ValidationRunner.cs`
**Language:** C# | **Lines:** 233 | **Last Scanned:** Not specified

## Purpose
This file contains the `ValidationRunner` class, which validates the `DESSimulator` against Taillard JSP benchmark instances. It provides methods for running simulations with different dispatching rules and verifying correctness invariants. The class plays a crucial role in ensuring the accuracy and reliability of the simulator.

## Dependencies
* `Assets.Scripts.Scheduling.Data`: provides Taillard instance and metadata classes.
* `Assets.Scripts.Scheduling.Core`: provides the `DESSimulator` class and dispatching rules.

## Classes
### `ValidationRunner`
Role: validates the `DESSimulator` against Taillard JSP benchmark instances.
Inheritance: none.
| Method | Signature | Description |
|--------|-----------|-------------|
| `RunSingle` | `ValidationResult RunSingle(TaillardInstance instance, DispatchingRule rule)` | Simulates a single instance with one dispatching rule and returns a gap analysis. |
| `RunAllRules` | `List<ValidationResult> RunAllRules(TaillardInstance instance)` | Runs every dispatching rule on an instance and returns results sorted by makespan. |
| `ValidateConstraints` | `List<string> ValidateConstraints(TaillardInstance instance, DispatchingRule rule)` | Verifies correctness invariants of the `DESSimulator` against a benchmark instance. |

### `ValidationResult`
Role: encapsulates the outcome of a single simulation run against a benchmark instance.
Inheritance: none (struct).
| Method | Signature | Description |
|--------|-----------|-------------|
| `ToString` | `string ToString()` | Returns a single-line summary of the result suitable for console logging. |

## Notes
The `ValidationRunner` class uses a floating-point epsilon of 1e-9 to guard against rounding errors in double arithmetic when comparing times. This is particularly important when checking for machine overlap and precedence constraints.