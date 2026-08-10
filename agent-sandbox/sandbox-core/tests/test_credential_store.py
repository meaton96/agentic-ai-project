import pytest

from sandbox_core.runtime.credential_store import (
    CredentialFileError,
    CredentialNotFoundError,
    YamlCredentialStore,
)


def write_store(tmp_path, contents: str):
    path = tmp_path / "credentials.yaml"
    path.write_text(contents)
    return path


def test_resolve_returns_value_for_known_ref(tmp_path):
    path = write_store(tmp_path, "rit-api-key: sk-abc123\n")
    store = YamlCredentialStore(path)
    assert store.resolve("rit-api-key") == "sk-abc123"


def test_resolve_missing_ref_raises_with_ref_and_path(tmp_path):
    path = write_store(tmp_path, "some-other-ref: value\n")
    store = YamlCredentialStore(path)

    with pytest.raises(CredentialNotFoundError) as exc_info:
        store.resolve("rit-api-key")

    message = str(exc_info.value)
    assert "rit-api-key" in message
    assert str(path) in message


def test_resolve_missing_file_raises_clear_error(tmp_path):
    path = tmp_path / "does-not-exist.yaml"
    store = YamlCredentialStore(path)

    with pytest.raises(CredentialFileError) as exc_info:
        store.resolve("anything")

    assert str(path) in str(exc_info.value)


def test_resolve_non_mapping_yaml_raises_clear_error(tmp_path):
    path = write_store(tmp_path, "- just\n- a\n- list\n")
    store = YamlCredentialStore(path)

    with pytest.raises(CredentialFileError):
        store.resolve("anything")


def test_resolve_coerces_non_string_values_to_str(tmp_path):
    path = write_store(tmp_path, "numeric-ref: 12345\n")
    store = YamlCredentialStore(path)
    assert store.resolve("numeric-ref") == "12345"
