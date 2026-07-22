from __future__ import annotations


def test_list_datasets_empty_when_root_missing(client):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    assert resp.json() == []


def test_list_datasets_returns_filename_size_modified(client, dataset_csv):
    resp = client.get("/api/datasets")
    assert resp.status_code == 200
    entries = resp.json()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["filename"] == dataset_csv
    assert entry["size"] > 0
    assert entry["modified"] > 0


def test_launch_run_rejects_dataset_outside_root(client, dataset_csv):
    resp = client.post("/api/runs", json={
        "dataset": "../outside.csv",
        "orchestrator": "static",
        "target": "churned",
    })
    assert resp.status_code == 422


def test_launch_run_rejects_missing_dataset(client):
    resp = client.post("/api/runs", json={
        "dataset": "does_not_exist.csv",
        "orchestrator": "static",
        "target": "churned",
    })
    assert resp.status_code == 422
