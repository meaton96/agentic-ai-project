"""The experiment catalog: every model, augmentation, schedule and loss an
agent may choose, each with typed, bounded parameters. Agents propose
experiments as JSON over these entries (see experiment.py); they never
write code."""

from agentic_pdm.catalog.experiment import (
    DatasetFacts,
    build_experiment,
    catalog_summary,
    describe_entry,
    estimate_cost,
    validate_experiment,
)

__all__ = ["DatasetFacts", "build_experiment", "catalog_summary", "describe_entry", "estimate_cost",
           "validate_experiment"]
