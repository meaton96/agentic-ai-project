"""Real proof-of-concept: a sandbox GateStep wrapping one actual
agentic_ml harness function (not a fixture/toy heuristic), run against the
real Titanic dataset. This is the first concrete evidence that:

1. A gate can be the *first* step in a pipeline (reads the seed task via
   the reserved "__task__" outputs key — see pipeline_runner.py) instead of
   needing an LLM agent step just to pass through a dataset path.
2. A gate's real work can be a direct import of agentic_ml's own
   deterministic harness code, unmodified — no MCP wrapper needed for this
   case, since the dependency footprint (pandas + agentic_ml's own small
   modules) is light enough to just be importable wherever this gate runs.
3. A gate's "output" can be an artifact-path reference (here, a small JSON
   validation report) rather than prose — proving PipelineStepResult's
   text-only assumption doesn't need to hold for a deterministic pipeline
   stage's result.

Requires running under a Python that has both `sandbox_core` and
`agentic_ml` (+ pandas) importable — e.g. the repo-root .venv, which has
both editable-installed. Resolvable as a GateStep.gate path:
"agentic_ml_gates:validate_titanic_intake".
"""

import json
from pathlib import Path

import pandas as pd
from agentic_ml.harness.intake import validate_dataset_spec_proposal

# Hardcoded stand-in for what an LLM intake agent would normally propose —
# this proof is about the harness/gate plumbing, not intake automation.
_PROPOSED_SPEC = {
    "target_column": "Survived",
    "id_columns": ["PassengerId"],
}

_REPORT_PATH = Path("workspace/output/intake_validation_report.json")


def validate_titanic_intake(outputs: dict[str, str]) -> tuple[str, str]:
    csv_path = outputs["__task__"]
    df = pd.read_csv(csv_path)

    errors = validate_dataset_spec_proposal(df, _PROPOSED_SPEC)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REPORT_PATH.write_text(
        json.dumps(
            {
                "csv_path": csv_path,
                "n_rows": len(df),
                "n_columns": len(df.columns),
                "proposal": _PROPOSED_SPEC,
                "errors": errors,
            },
            indent=2,
        )
    )

    decision = "valid" if not errors else "invalid"
    return decision, str(_REPORT_PATH)
