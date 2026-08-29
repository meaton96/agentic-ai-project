"""Write-only over the M1 YAML credential store. This router must NEVER put a
secret value in a response body — see `list_credentials` below and its test
in tests/test_credentials.py, which asserts on the raw response bytes, not
just the parsed model, for exactly this reason."""

from pathlib import Path

import yaml
from fastapi import APIRouter, Response
from pydantic import BaseModel

from sandbox_core.runtime.credential_store import credentials_path

router = APIRouter(prefix="/credentials", tags=["credentials"])


class SetCredentialRequest(BaseModel):
    value: str


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    return data if isinstance(data, dict) else {}


@router.post("/{ref}", status_code=204)
def set_credential(ref: str, body: SetCredentialRequest) -> Response:
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _load(path)
    data[ref] = body.value
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return Response(status_code=204)


@router.get("")
def list_credentials() -> list[str]:
    """Ref names only — no GET /credentials/{ref} exists, and never will;
    the store is write-only from this API's perspective."""
    return sorted(_load(credentials_path()).keys())
