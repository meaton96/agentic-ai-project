"""
run_from_manifest.py
====================
Phase-1 wiring: derive run_pipeline's arguments from a TaskManifest instead of
hardcoded constants. run_pipeline itself is unchanged, so a manifest whose values
match the old constants produces byte-identical output -- the existing golden is
the acceptance test for "manifest-driven == same behavior".
"""
from __future__ import annotations
from pathlib import Path

try:
    from .task_manifest import TaskManifest, validate_manifest
except ImportError:
    from task_manifest import TaskManifest, validate_manifest


def run_from_manifest(manifest: TaskManifest, run_pipeline, workdir,
                      base_dir: str | Path = ".", validate: bool = True) -> dict:
    """Map a manifest onto run_pipeline(flight_dir, metadata, model, workdir,
    top_k, spec). base_dir resolves the manifest's relative paths."""
    if validate:
        errs = validate_manifest(manifest)
        if errs:
            raise ValueError("invalid manifest:\n  " + "\n  ".join(errs))

    base = Path(base_dir)
    def resolve(p):
        return str(base / p) if p else None

    return run_pipeline(
        flight_dir=resolve(manifest.flight_dir),
        metadata=resolve(manifest.metadata_path),
        model=resolve(manifest.model_path),
        spec=resolve(manifest.feature_spec_path),
        workdir=str(workdir),
        top_k=manifest.top_k,
    )
