from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..pipeline_specs import read_pipeline_spec

router = APIRouter(prefix="/pipeline-runs", tags=["pipeline-runs"])


class LaunchPipelineRunRequest(BaseModel):
    pipeline_id: str
    task: str


class LaunchPipelineRunResponse(BaseModel):
    pipeline_run_id: str


@router.post("", status_code=202)
async def launch_pipeline_run(body: LaunchPipelineRunRequest, request: Request) -> LaunchPipelineRunResponse:
    # Must be `async def`, not a plain `def` — same reason routes/runs.py's
    # launch_run is: PipelineRunManager.launch_run's asyncio.create_task(...)
    # needs the app's own running event loop, which a sync route (dispatched
    # via a worker thread) doesn't have.
    pipeline = read_pipeline_spec(request.app.state.pipelines_dir, body.pipeline_id)
    if pipeline is None:
        raise HTTPException(404, f"pipeline {body.pipeline_id!r} not found")
    pipeline_run_id = request.app.state.pipeline_run_manager.launch_run(pipeline, body.task)
    return LaunchPipelineRunResponse(pipeline_run_id=pipeline_run_id)


@router.get("")
def list_pipeline_runs(request: Request) -> list[dict]:
    return request.app.state.pipeline_run_manager.list_pipeline_runs()


@router.get("/{pipeline_run_id}")
def get_pipeline_run(pipeline_run_id: str, request: Request) -> dict:
    detail = request.app.state.pipeline_run_manager.get_pipeline_run_detail(pipeline_run_id)
    if detail is None:
        raise HTTPException(404, f"pipeline run {pipeline_run_id!r} not found")
    return detail
