import os
from pathlib import Path

import yaml

DEFAULT_CREDENTIALS_PATH = Path.home() / ".sandbox" / "credentials.yaml"
CREDENTIALS_PATH_ENV_VAR = "SANDBOX_CREDENTIALS_PATH"


def credentials_path() -> Path:
    override = os.environ.get(CREDENTIALS_PATH_ENV_VAR)
    return Path(override).expanduser() if override else DEFAULT_CREDENTIALS_PATH


class CredentialNotFoundError(LookupError):
    def __init__(self, ref: str, path: Path):
        self.ref = ref
        self.path = path
        super().__init__(
            f"credential ref {ref!r} not found in {path}. "
            f"Add a `{ref}: <value>` entry to that file "
            f"(or point {CREDENTIALS_PATH_ENV_VAR} at the YAML file that has it)."
        )


class CredentialFileError(RuntimeError):
    def __init__(self, path: Path, reason: str):
        self.path = path
        super().__init__(f"could not read credential store at {path}: {reason}")


class YamlCredentialStore:
    """CredentialResolver backed by a flat `ref -> value` YAML file."""

    def __init__(self, path: Path | None = None):
        self.path = path if path is not None else credentials_path()

    def _load(self) -> dict:
        if not self.path.exists():
            raise CredentialFileError(
                self.path,
                f"file does not exist. Create it (e.g. `mkdir -p {self.path.parent} && touch {self.path}`) "
                "and add `<ref>: <value>` entries.",
            )
        try:
            data = yaml.safe_load(self.path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise CredentialFileError(self.path, f"invalid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise CredentialFileError(self.path, "top-level YAML content must be a mapping of ref -> value")
        return data

    def resolve(self, ref: str) -> str:
        data = self._load()
        if ref not in data:
            raise CredentialNotFoundError(ref, self.path)
        return str(data[ref])
