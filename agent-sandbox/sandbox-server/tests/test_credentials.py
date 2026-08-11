from fastapi.testclient import TestClient

from sandbox_server.main import create_app

SECRET = "sk-super-secret-value-should-never-leak"


def client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("SANDBOX_CREDENTIALS_PATH", str(tmp_path / "credentials.yaml"))
    app = create_app(specs_dir=tmp_path / "agents", output_root=tmp_path / "runs")
    return TestClient(app)


def test_set_credential_returns_204_with_no_body(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    resp = c.post("/credentials/my-ref", json={"value": SECRET})
    assert resp.status_code == 204
    assert resp.content == b""


def test_list_credentials_never_leaks_the_value(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    c.post("/credentials/my-ref", json={"value": SECRET})
    c.post("/credentials/other-ref", json={"value": "another-secret-xyz"})

    resp = c.get("/credentials")
    assert resp.status_code == 200

    # assert on the raw response bytes, not just the parsed model — the
    # requirement is that the value never appears in the body at all
    assert SECRET not in resp.text
    assert "another-secret-xyz" not in resp.text

    assert sorted(resp.json()) == ["my-ref", "other-ref"]


def test_set_credential_actually_persists_for_the_runtime_to_resolve(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    c.post("/credentials/my-ref", json={"value": SECRET})

    from sandbox_core.runtime.credential_store import YamlCredentialStore

    assert YamlCredentialStore().resolve("my-ref") == SECRET


def test_updating_an_existing_ref_does_not_drop_other_refs(tmp_path, monkeypatch):
    c = client(tmp_path, monkeypatch)
    c.post("/credentials/ref-a", json={"value": "a"})
    c.post("/credentials/ref-b", json={"value": "b"})
    c.post("/credentials/ref-a", json={"value": "a-updated"})

    resp = c.get("/credentials")
    assert sorted(resp.json()) == ["ref-a", "ref-b"]
