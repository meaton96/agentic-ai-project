"""
Phase-1 acceptance: driving run_pipeline from the manifest must reproduce the SAME
frozen golden the hardcoded path produced. If this passes, manifest-driven == the
old constant-driven behavior -- Phase 1 is done by construction, no new golden.

Uses the SAME golden fixture as test_golden.py (tests/golden/aviation_pipeline.json).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from task_manifest import load_manifest, validate_manifest
from run_from_manifest import run_from_manifest
from golden_lib import clean_workdir, check_invariants, EXCLUDE_KEYS
from golden_harness import compare_to_golden
import json

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "manifests" / "aviation_c28.json"


def _snapshot_from_manifest(run_pipeline, manifest, workdir, base_dir, top_k):
    Path(workdir).mkdir(parents=True, exist_ok=True)
    clean_workdir(workdir)
    out = run_from_manifest(manifest, run_pipeline, workdir=workdir, base_dir=base_dir)
    preds = json.loads((Path(workdir) / "preds.json").read_text())["predictions"]
    preds = sorted(preds, key=lambda p: str(p.get("flight", p.get("filename", ""))))
    return {"result": out, "predictions": preds, "_hitl_summary_raw": out.get("hitl_summary", "")}


def test_manifest_matches_repo_layout(paths):
    """The manifest's resolved paths (base_dir=REPO) must point at the same files
    the hardcoded conftest paths use -- otherwise 'identical' would be accidental."""
    m = load_manifest(MANIFEST)
    assert validate_manifest(m) == []
    assert (REPO / m.data_path / "flights").resolve() == Path(paths["flight_dir"]).resolve()
    assert (REPO / m.model_path).resolve() == Path(paths["model"]).resolve()
    assert (REPO / m.feature_spec_path).resolve() == Path(paths["spec"]).resolve()


def test_golden_via_manifest(paths, run_pipeline):
    m = load_manifest(MANIFEST)
    snap = _snapshot_from_manifest(run_pipeline, m,
                                   workdir=Path(paths["workdir"]).parent / "_manifest_work",
                                   base_dir=REPO, top_k=m.top_k)

    errs = check_invariants(snap, m.top_k)
    assert not errs, "invariant violations:\n" + "\n".join(errs)

    gp = paths["golden"]
    assert Path(gp).exists(), "freeze the golden first: UPDATE_GOLDEN=1 pytest tests/test_golden.py"
    frozen = {"result": snap["result"], "predictions": snap["predictions"]}
    diffs = compare_to_golden(frozen, str(gp), exclude_keys=EXCLUDE_KEYS, tol=1e-6)
    assert not diffs, "manifest-driven run diverged from the hardcoded golden:\n" + "\n".join(diffs[:25])
