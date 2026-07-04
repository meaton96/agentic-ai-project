"""
Phase-0 golden regression gate for orchestrator_sim.run_pipeline on real c28_demo.

Freeze once:   UPDATE_GOLDEN=1 pytest tests/test_golden.py
Gate:          pytest tests/test_golden.py

The golden is tied to c28_demo/ + c28_model.joblib; re-export or retrain -> re-freeze.
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent

ROOT = HERE.parent
sys.path.insert(0, str(HERE)) 
sys.path.insert(0, str(ROOT / "scripts"))

from golden_lib import run_and_snapshot, check_invariants, EXCLUDE_KEYS
from golden_harness import save_golden, compare_to_golden
from conftest import UPDATE_GOLDEN


def test_golden(paths, run_pipeline):
    snap = run_and_snapshot(run_pipeline, paths["flight_dir"], paths["metadata"],
                            paths["model"], paths["workdir"], paths["top_k"])

    # structural invariants first -- these must hold no matter what is frozen
    errs = check_invariants(snap, paths["top_k"])
    assert not errs, "invariant violations:\n" + "\n".join(errs)

    gp = Path(paths["golden"]); gp.parent.mkdir(parents=True, exist_ok=True)
    frozen = {"result": snap["result"], "predictions": snap["predictions"]}

    if UPDATE_GOLDEN or not gp.exists():
        save_golden(frozen, str(gp), exclude_keys=EXCLUDE_KEYS)
        import warnings; warnings.warn(f"golden written to {gp} (no comparison this run)")
        return

    diffs = compare_to_golden(frozen, str(gp), exclude_keys=EXCLUDE_KEYS, tol=1e-6)
    assert not diffs, "golden regression:\n" + "\n".join(diffs[:25])
