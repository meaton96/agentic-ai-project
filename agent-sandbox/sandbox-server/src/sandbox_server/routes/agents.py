from fastapi import APIRouter, HTTPException, Request, Response

from sandbox_core.schemas.agent_spec import AgentSpec

from ..specs import delete_agent_spec, list_agent_specs, read_agent_spec, write_agent_spec

router = APIRouter(prefix="/agents", tags=["agents"])


def _specs_dir(request: Request):
    return request.app.state.specs_dir


@router.get("")
def list_agents(request: Request) -> list[AgentSpec]:
    return list_agent_specs(_specs_dir(request))


@router.get("/{agent_id}")
def get_agent(agent_id: str, request: Request) -> AgentSpec:
    spec = read_agent_spec(_specs_dir(request), agent_id)
    if spec is None:
        raise HTTPException(404, f"agent {agent_id!r} not found")
    return spec


@router.post("", status_code=201)
def create_agent(spec: AgentSpec, request: Request) -> AgentSpec:
    # `spec: AgentSpec` above is the validation point: FastAPI/pydantic
    # rejects a body that doesn't match AgentSpec with its own 422 response
    # (field paths + messages) before this function body ever runs — no
    # manual try/except needed to avoid swallowing that into a 500.
    specs_dir = _specs_dir(request)
    if read_agent_spec(specs_dir, spec.id) is not None:
        raise HTTPException(409, f"agent {spec.id!r} already exists")
    write_agent_spec(specs_dir, spec)
    return spec


@router.put("/{agent_id}")
def update_agent(agent_id: str, spec: AgentSpec, request: Request) -> AgentSpec:
    if spec.id != agent_id:
        raise HTTPException(400, f"body id {spec.id!r} does not match path id {agent_id!r}")
    write_agent_spec(_specs_dir(request), spec)
    return spec


@router.delete("/{agent_id}", status_code=204)
def delete_agent(agent_id: str, request: Request) -> Response:
    found = delete_agent_spec(_specs_dir(request), agent_id)
    if not found:
        raise HTTPException(404, f"agent {agent_id!r} not found")
    return Response(status_code=204)
