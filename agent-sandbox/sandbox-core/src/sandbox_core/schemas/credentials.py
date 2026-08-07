from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class CredentialRef(BaseModel):
    ref: str
    description: str | None = None


@runtime_checkable
class CredentialResolver(Protocol):
    def resolve(self, ref: str) -> str:
        ...
