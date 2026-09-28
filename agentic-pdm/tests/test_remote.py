"""fetch_store and the pipeline's remote path, against a fake pdm-data-server
patched in for requests.get."""
import json

import numpy as np
import pandas as pd
import pytest

from agentic_pdm import pipeline, remote
from agentic_pdm.remote import RemoteDataError, fetch_store
from conftest import make_long_frame

TOKEN = "pdm_export_s3cr3t"


def sql(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, payload=None):
        self.status_code, self.body, self.headers, self.payload = status, body, headers or {}, payload

    def json(self):
        if self.payload is None:
            raise ValueError("no JSON body")
        return self.payload

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeServer:
    """Serves one long-format frame (original column names) as a dataset.
    `roles` uses original names; the manifest reports SQL names."""

    def __init__(self, frame, roles, positive_class=None, content_hash="h1"):
        self.frame, self.content_hash, self.export_hash = frame, content_hash, None
        self.column_names = {c: sql(c) for c in frame.columns}
        self.manifest_status, self.calls = 200, []
        self.roles = {k: (None if v is None else [sql(c) for c in v] if isinstance(v, list) else sql(v))
                      for k, v in roles.items()}
        self.positive_class = positive_class

    def get(self, url, headers=None, params=None, **kwargs):
        assert headers == {"Authorization": f"Bearer {TOKEN}"}
        self.calls.append(url.rsplit("/", 1)[-1])
        if url.endswith("/manifest"):
            if self.manifest_status != 200:
                return FakeResponse(self.manifest_status, payload={"detail": "token lacks scope"})
            return FakeResponse(payload={
                "content_hash": self.content_hash,
                "column_roles": {**self.roles, "order": "row_idx", "excluded": [], "columns": []},
                "column_names": self.column_names,
                "label_info": {"column": self.roles["label"], "classes": [], "positive_class": self.positive_class},
                "fold_info": {"column": self.roles.get("fold"), "values": []},
            })
        assert url.endswith("/export") and kwargs.get("stream") and params["format"] == "csv"
        orig = {s: o for o, s in self.column_names.items()}
        frame = self.frame[[orig[c] for c in params["columns"].split(",")]] if params.get("columns") else self.frame
        return FakeResponse(body=frame.to_csv(index=False).encode(),
                            headers={"X-Content-Hash": self.export_hash or self.content_hash})


def small_frame(n_seq=6):
    rng = np.random.default_rng(0)
    parts = []
    for i in range(n_seq):
        length = 20 + 5 * i
        parts.append(pd.DataFrame({
            "Flight ID": i, "row_idx": np.arange(length),
            "Alt (ft)": rng.normal(i % 2, 1, length), "IAS": rng.normal(0, 1, length),
            "Before After": i % 2, "Tail": i // 2, "Fold": i % 3,
        }))
    return pd.concat(parts, ignore_index=True)


SMALL_ROLES = {"sequence_id": "Flight ID", "label": "Before After", "group": "Tail", "fold": "Fold",
               "channels": ["Alt (ft)", "IAS"]}


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer(small_frame(), SMALL_ROLES, positive_class=1)
    monkeypatch.setattr(remote.requests, "get", fake.get)
    return fake


def test_builds_store_then_hits_cache(server, tmp_path):
    store_dir = fetch_store("ngafid_toy", tmp_path, max_len=32, token=TOKEN)
    manifest = json.loads((store_dir / "manifest.json").read_text())
    assert manifest["content_hash"] == "h1" and manifest["remote_dataset"] == "ngafid_toy"
    assert manifest["channels"] == ["Alt (ft)", "IAS"]  # no row_idx
    assert manifest["classes"] == [0, 1] and manifest["positive_label"] == "1"
    assert len(pd.read_csv(store_dir / "meta.csv")) == 6
    assert not (store_dir / "export.csv").exists()
    assert server.calls == ["manifest", "export"]

    assert fetch_store("ngafid_toy", tmp_path, max_len=32, token=TOKEN) == store_dir
    assert server.calls == ["manifest", "export", "manifest"]


def test_new_content_hash_builds_a_new_store(server, tmp_path):
    first = fetch_store("ngafid_toy", tmp_path, token=TOKEN)
    server.content_hash = "h2"
    second = fetch_store("ngafid_toy", tmp_path, token=TOKEN)
    assert first != second and (second / "manifest.json").exists()
    assert server.calls.count("export") == 2


def test_hash_mismatch_raises_and_cleans_up(server, tmp_path):
    server.export_hash = "other"
    with pytest.raises(RemoteDataError, match="changed between manifest and export"):
        fetch_store("ngafid_toy", tmp_path, token=TOKEN)
    assert list((tmp_path / "ngafid_toy").iterdir()) == []


def test_auth_error_does_not_leak_the_token(server, tmp_path):
    server.manifest_status = 403
    with pytest.raises(RemoteDataError) as info:
        fetch_store("ngafid_toy", tmp_path, token=TOKEN)
    assert "403" in str(info.value) and "ngafid_toy" in str(info.value) and TOKEN not in str(info.value)


def test_missing_token(monkeypatch, tmp_path):
    monkeypatch.delenv("PDM_DATA_TOKEN", raising=False)
    with pytest.raises(RemoteDataError, match="PDM_DATA_TOKEN is not set"):
        fetch_store("ngafid_toy", tmp_path)


# -- through the pipeline ------------------------------------------------------------


@pytest.fixture
def sandbox_env(tmp_path, monkeypatch):
    for var in ("PDM_DATASETS_DIR", "DATASETS_DIR", "SANDBOX_DATASETS_DIR", "PDM_WORK_DIR", "PDM_DATA_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GATE_SCRATCH_DIR", str(tmp_path / "scratch"))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    return tmp_path


def contract():
    return json.dumps({"dataset": "ngafid_toy", "budget": {"max_experiments": 2, "max_minutes": 60}})


def test_init_without_token_or_local_store_is_invalid(sandbox_env):
    decision, out = pipeline.init_run({"__task__": contract()})
    assert decision == "invalid_contract" and "PDM_DATA_TOKEN" in out


def test_remote_run_resolves_store_once_and_trains_from_it(sandbox_env, monkeypatch):
    fake = FakeServer(make_long_frame().drop(columns=["leak"]),
                      {"sequence_id": "seq", "label": "target", "group": "unit", "fold": "fold",
                       "channels": ["a", "b", "c"]}, positive_class="bad")
    monkeypatch.setattr(remote.requests, "get", fake.get)
    monkeypatch.setenv("PDM_DATA_TOKEN", TOKEN)

    outputs = {"__task__": contract()}
    decision, out = pipeline.init_run(outputs)
    assert decision == "ready", out
    outputs["init"] = out
    run_dir = json.loads(out)["pdm_run_dir"]
    store_dir = json.loads(open(f"{run_dir}/dataset.json").read())["store_dir"]
    assert store_dir == json.loads(out)["store_dir"]
    assert store_dir.startswith(str(sandbox_env / "scratch" / "pdm-datasets" / "ngafid_toy"))

    tiny = {"name": "tiny", "architecture": {"id": "small_cnn", "params": {"width": 8, "depth": 3}},
            "input": {"max_len": 64, "pool": 1}, "training": {"epochs": 1, "batch_size": 16}, "folds": [0]}
    outputs["plan"] = "```json\n" + json.dumps(tiny) + "\n```"
    assert pipeline.validate_proposal(outputs)[0] == "valid"
    monkeypatch.delenv("PDM_DATA_TOKEN")  # the job reads the recorded path, no download
    assert pipeline.train_experiment(outputs)[0] == "done"
    assert fake.calls == ["manifest", "export"]
