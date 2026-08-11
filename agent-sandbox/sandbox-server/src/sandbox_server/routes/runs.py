from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..specs import read_agent_spec

router = APIRouter(prefix="/runs", tags=["runs"])


class LaunchRunRequest(BaseModel):
    agent_id: str
    task: str


class LaunchRunResponse(BaseModel):
    run_id: str


@router.post("", status_code=202)
async def launch_run(body: LaunchRunRequest, request: Request) -> LaunchRunResponse:
    # Must be `async def`, not a plain `def`: FastAPI runs sync route
    # functions in a worker thread (via run_in_threadpool), which has no
    # running asyncio loop of its own — RunManager.launch_run's
    # `asyncio.create_task(...)` needs to find the *app's* loop, which only
    # an `async def` route (dispatched directly on that loop) provides.
    agent = read_agent_spec(request.app.state.specs_dir, body.agent_id)
    if agent is None:
        raise HTTPException(404, f"agent {body.agent_id!r} not found")
    run_id = request.app.state.run_manager.launch_run(agent, body.task)
    return LaunchRunResponse(run_id=run_id)


@router.get("")
def list_runs(request: Request) -> list[dict]:
    return request.app.state.run_manager.list_runs()


@router.get("/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    detail = request.app.state.run_manager.get_run_detail(run_id)
    if detail is None:
        raise HTTPException(404, f"run {run_id!r} not found")
    return detail
