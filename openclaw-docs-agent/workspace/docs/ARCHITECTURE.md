# Architecture Overview
**Project:** Factory Simulation | **Generated:** 2026-06-04 | **Files:** 63

## Project Overview
The Factory Simulation project is a complex system designed to model and simulate job shop scheduling in a manufacturing environment. Its primary purpose is to provide a realistic simulation of factory operations, allowing for the testing and optimization of various scheduling algorithms and strategies. The project utilizes a combination of C# and Python scripts to achieve this goal.

## Folder Structure
```markdown
codebase/
├── Camera/         # Camera-related classes
├── Core/           # Core simulation classes (e.g., DESSimulator, DispatchingRules)
├── Data/           # Data structures for simulation (e.g., TaillardInstance)
├── Simulation/     # Simulation-related classes (e.g., DecisionCoordinator, FactoryOrchestrator)
├── UI/             # User interface classes (e.g., SimulationHUD, InstanceSelectMenu)
└── Validation/     # Validation scripts and classes (e.g., CrossValidator, ValidationExporter)
```

## Key Components
* `DESSimulator`: A discrete-event simulator for the Job Shop Problem (JSP) validation
* `DispatchingEngine`: Evaluates and applies dispatching rules in the Flexible Job Shop Scheduling Problem (FJSSP) simulation
* `FactoryOrchestrator`: Manages episode lifecycle, coordinates machine/AGV/job systems, and interfaces with the learning agent
* `SchedulingAgent`: Drives job-shop scheduling decisions in a simulation environment
* `StochasticEventManager`: Manages stochastic events in the simulation

## Data Flow
```markdown
Input (Taillard instance) → DESSimulator → DispatchingEngine → FactoryOrchestrator → SchedulingAgent → Output (simulation results)
```

## Entry Points
The project can be invoked through the Unity Play mode or by running the `HeadlessBatchRunner` class.

## External Dependencies
* Unity ML-Agents framework
* Python libraries (e.g., `job_shop_lib`) for cross-validation and reference generation
* JSON files for configuration and data storage