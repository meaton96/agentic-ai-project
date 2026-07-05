"""run_pipeline must be bit-reproducible (it underpins the whole golden approach).
Run it twice into different workdirs; the substantive output must be identical."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from golden_lib import run_and_snapshot, EXCLUDE_KEYS
from golden_harness import compare_objects


def test_determinism(paths, run_pipeline):
    a = run_and_snapshot(run_pipeline, paths["flight_dir"], paths["metadata"],
                         paths["model"], Path(paths["workdir"]).parent / "_det_a", paths["top_k"], spec=paths["spec"])
    b = run_and_snapshot(run_pipeline, paths["flight_dir"], paths["metadata"],
                         paths["model"], Path(paths["workdir"]).parent / "_det_b", paths["top_k"], spec=paths["spec"])
    fa = {"result": a["result"], "predictions": a["predictions"]}
    fb = {"result": b["result"], "predictions": b["predictions"]}
    diffs = compare_objects(fa, fb, EXCLUDE_KEYS, tol=0.0)
    assert not diffs, "run_pipeline is non-deterministic:\n" + "\n".join(diffs[:15])