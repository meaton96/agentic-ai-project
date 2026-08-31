from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from sandbox_core.runtime.operation_log import read_operations
from sandbox_core.schemas.operations import Operation, OperationError, OperationRecord
from sandbox_core.schemas.pipeline_spec import PipelineSpec

from ..operations import apply_pipeline_operation
from ..pipeline_specs import delete_pipeline_spec, list_pipeline_specs, read_pipeline_spec, write_pipeline_spec

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


class PipelineOperationResult(BaseModel):
    spec: PipelineSpec
    record: OperationRecord


def _pipelines_dir(request: Request):
    return request.app.state.pipelines_dir


def _specs_dir(request: Request):
    return request.app.state.specs_dir


def _operations_root(request: Request):
    return request.app.state.operations_root


@router.get("")
def list_pipelines(request: Request) -> list[PipelineSpec]:
    return list_pipeline_specs(_pipelines_dir(request))


@router.get("/{pipeline_id}")
def get_pipeline(pipeline_id: str, request: Request) -> PipelineSpec:
    spec = read_pipeline_spec(_pipelines_dir(request), pipeline_id)
    if spec is None:
        raise HTTPException(404, f"pipeline {pipeline_id!r} not found")
    return spec


@router.post("", status_code=201)
def create_pipeline(spec: PipelineSpec, request: Request) -> PipelineSpec:
    pipelines_dir = _pipelines_dir(request)
    if read_pipeline_spec(pipelines_dir, spec.id) is not None:
        raise HTTPException(409, f"pipeline {spec.id!r} already exists")
    write_pipeline_spec(pipelines_dir, spec)
    return spec


@router.put("/{pipeline_id}")
def update_pipeline(pipeline_id: str, spec: PipelineSpec, request: Request) -> PipelineSpec:
    if spec.id != pipeline_id:
        raise HTTPException(400, f"body id {spec.id!r} does not match path id {pipeline_id!r}")
    write_pipeline_spec(_pipelines_dir(request), spec)
    return spec


@router.delete("/{pipeline_id}", status_code=204)
def delete_pipeline(pipeline_id: str, request: Request) -> Response:
    found = delete_pipeline_spec(_pipelines_dir(request), pipeline_id)
    if not found:
        raise HTTPException(404, f"pipeline {pipeline_id!r} not found")
    return Response(status_code=204)


@router.post("/{pipeline_id}/operations", status_code=201)
def apply_pipeline_operation_route(pipeline_id: str, operation: Operation, request: Request) -> PipelineOperationResult:
    try:
        spec, record = apply_pipeline_operation(
            pipelines_dir=_pipelines_dir(request),
            specs_dir=_specs_dir(request),
            operations_root=_operations_root(request),
            pipeline_id=pipeline_id,
            operation=operation,
        )
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except OperationError as exc:
        raise HTTPException(400, str(exc)) from exc
    return PipelineOperationResult(spec=spec, record=record)


@router.get("/{pipeline_id}/operations")
def list_pipeline_operations(pipeline_id: str, request: Request) -> list[OperationRecord]:
    if read_pipeline_spec(_pipelines_dir(request), pipeline_id) is None:
        raise HTTPException(404, f"pipeline {pipeline_id!r} not found")
    path = _operations_root(request) / "pipeline" / f"{pipeline_id}.jsonl"
    return read_operations(path)
