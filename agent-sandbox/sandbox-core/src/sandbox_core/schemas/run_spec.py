from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field


class RunSpec(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str
    task: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    max_turns: int | None = None
