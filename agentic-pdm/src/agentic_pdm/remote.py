"""
Tensor stores built from pdm-data-server's bulk export API.

fetch_store downloads a dataset's long-format CSV export, ingests it with
ingest_long_csv, and caches the resulting store under

    <cache_root>/<dataset_id>/<key>/     key = hash of (content_hash, max_len)

so a dataset is downloaded once per content version. The store is built in
a hidden temp directory next to it and published with one rename, so a
concurrent run either sees a complete store or none. Neither the download
nor the ingest holds the whole export in memory.

Configuration: $PDM_DATA_URL (default https://agentsandbox.gccis.rit.edu)
and $PDM_DATA_TOKEN (a pdm_export token with `export` scope). Proxy settings
come from $HTTPS_PROXY via requests, including credentials in the proxy URL.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

import requests

from agentic_pdm.ingest import MANIFEST_FILENAME, IngestConfig, ingest_long_csv

DEFAULT_BASE_URL = "https://agentsandbox.gccis.rit.edu"
CHUNK_BYTES = 1 << 20
PROGRESS_BYTES = 256 << 20
CACHE_VERSION = 1


class RemoteDataError(RuntimeError):
    pass


def _check(response: requests.Response, dataset_id: str, what: str) -> None:
    if response.status_code < 400:
        return
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        detail = None
    meaning = {401: "token invalid, expired or revoked", 403: "token lacks export scope for this dataset",
               404: "unknown dataset"}.get(response.status_code, "request failed")
    raise RemoteDataError(f"pdm-data {what} for {dataset_id!r}: HTTP {response.status_code} ({meaning})"
                          + (f": {detail}" if detail else ""))


def _roles(manifest: dict) -> tuple[IngestConfig, list[str]]:
    """IngestConfig (original column names) and the SQL columns to export."""
    roles = manifest["column_roles"]
    orig = {sql: original for original, sql in (manifest.get("column_names") or {}).items()}

    def name(sql: Optional[str]) -> Optional[str]:
        return None if sql is None else orig.get(sql, sql)

    if not roles.get("sequence_id"):
        raise RemoteDataError("dataset is tabular (no sequence_id column); only sequence datasets are supported")
    if not roles.get("label"):
        raise RemoteDataError("dataset has no label column")
    positive = (manifest.get("label_info") or {}).get("positive_class")
    config = IngestConfig(
        id_column=name(roles["sequence_id"]), label_column=name(roles["label"]),
        group_column=name(roles.get("group")), fold_column=name(roles.get("fold")),
        positive_label=None if positive is None else str(positive),
    )
    wanted = [roles["sequence_id"], roles["label"], roles.get("group"), roles.get("fold"), *roles["channels"]]
    columns = list(dict.fromkeys(c for c in wanted if c))
    return config, columns


def fetch_store(dataset_id: str, cache_root: Path, *, max_len: int = 4096,
                base_url: str | None = None, token: str | None = None) -> Path:
    """Returns the directory of a tensor store (manifest.json, meta.csv, X.f32)
    for `dataset_id`, built from pdm-data-server's export and cached under
    cache_root by content hash."""
    base_url = (base_url or os.environ.get("PDM_DATA_URL") or DEFAULT_BASE_URL).rstrip("/")
    token = token or os.environ.get("PDM_DATA_TOKEN")
    if not token:
        raise RemoteDataError("PDM_DATA_TOKEN is not set: add `credentials: {PDM_DATA_TOKEN: <ref>}` "
                              "to the init step")
    headers = {"Authorization": f"Bearer {token}"}
    dataset_url = f"{base_url}/v1/datasets/{dataset_id}"

    response = requests.get(f"{dataset_url}/manifest", headers=headers, timeout=60)
    _check(response, dataset_id, "manifest")
    manifest = response.json()
    content_hash = manifest["content_hash"]

    key = hashlib.sha256(json.dumps({"hash": content_hash, "max_len": max_len, "v": CACHE_VERSION},
                                    sort_keys=True).encode()).hexdigest()[:16]
    store_dir = Path(cache_root) / dataset_id / key
    if (store_dir / MANIFEST_FILENAME).exists():
        return store_dir

    config, columns = _roles(manifest)
    config.max_len = max_len
    tmp_dir = Path(cache_root) / dataset_id / f".{key}.{uuid.uuid4().hex[:8]}.tmp"
    tmp_dir.mkdir(parents=True)
    try:
        csv_path = tmp_dir / "export.csv"
        with requests.get(f"{dataset_url}/export", headers=headers, stream=True, timeout=(30, 900),
                          params={"format": "csv", "names": "original", "columns": ",".join(columns)}) as r:
            _check(r, dataset_id, "export")
            if r.headers.get("X-Content-Hash") != content_hash:
                raise RemoteDataError("dataset changed between manifest and export; retry")
            written, reported = 0, 0
            with open(csv_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_BYTES):
                    f.write(chunk)
                    written += len(chunk)
                    if written - reported >= PROGRESS_BYTES:
                        reported = written
                        print(f"pdm-data: downloaded {written >> 20} MB", flush=True)
        print(f"pdm-data: download complete ({written >> 20} MB), ingesting", flush=True)

        ingest_long_csv(csv_path, tmp_dir, config)
        store_manifest = json.loads((tmp_dir / MANIFEST_FILENAME).read_text())
        store_manifest.update(source=dataset_url, content_hash=content_hash, remote_dataset=dataset_id)
        (tmp_dir / MANIFEST_FILENAME).write_text(json.dumps(store_manifest, indent=2, default=str))
        csv_path.unlink()

        try:
            os.rename(tmp_dir, store_dir)
        except OSError:
            if not (store_dir / MANIFEST_FILENAME).exists():
                raise
            # Another run published the same store first.
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return store_dir
