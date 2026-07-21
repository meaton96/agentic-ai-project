from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Orchestrator = Literal["static", "dynamic"]
ModelEndpoint = Literal["rit", "gateway", "local"]
SplitStrategy = Literal["random", "stratified", "group", "time", "group_time"]


class LaunchRunRequest(BaseModel):
    """Request shape only. --target vs --goal is intentionally NOT
    enforced as either/or here: the pipeline itself doesn't error when
    both are given (--target simply skips intake and --goal goes
    unused) — duplicating that as a stricter API-level rule would be
    inventing validation the orchestrator doesn't have."""

    dataset: str = Field(..., min_length=1, description="Path relative to the configured datasets root")
    orchestrator: Orchestrator = "static"
    target: Optional[str] = None
    goal: Optional[str] = None
    strategy: Optional[SplitStrategy] = None
    max_candidates: Optional[int] = Field(default=None, gt=0)
    model_endpoint: ModelEndpoint = "rit"
    skip_feature_engineering: bool = False


class LaunchRunResponse(BaseModel):
    run_id: str
    status: str


class DatasetInfo(BaseModel):
    filename: str
    size: int
    modified: float
