import argparse
import asyncio
import os
import re
import sys
from pathlib import Path

import yaml

from sandbox_core.runtime.agent_loop import execute_run
from sandbox_core.runtime.credential_store import CredentialFileError, CredentialNotFoundError, YamlCredentialStore
from sandbox_core.runtime.event_log import EventLog
from sandbox_core.runtime.pipeline_runner import execute_pipeline
from sandbox_core.schemas.agent_spec import AgentSpec
from sandbox_core.schemas.events import ErrorEvent
from sandbox_core.schemas.pipeline_run import PipelineRunSpec
from sandbox_core.schemas.pipeline_spec import PipelineSpec
from sandbox_core.schemas.run_spec import RunSpec

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env_vars(text: str, source: Path) -> str:
    """Lets an AgentSpec YAML reference ${VAR} placeholders (e.g. for model
    base_url/model_name) instead of hardcoding an endpoint into a committed file."""

    def replace(match: re.Match) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise KeyError(f"{source} references ${{{name}}}, but that environment variable is not set")
        return os.environ[name]

    return _ENV_VAR_PATTERN.sub(replace, text)


def _load_agent_spec(path: Path) -> AgentSpec:
    data = yaml.safe_load(_expand_env_vars(path.read_text(), path))
    return AgentSpec.model_validate(data)


def _load_pipeline_spec(path: Path) -> PipelineSpec:
    data = yaml.safe_load(_expand_env_vars(path.read_text(), path))
    return PipelineSpec.model_validate(data)


class DirectoryAgentSpecLoader:
    """AgentSpecLoader over a directory of `<agent_id>.yaml` files — the
    headless-CLI counterpart to sandbox-server's specs.py, for running a
    pipeline with no server involved."""

    def __init__(self, agents_dir: Path):
        self._agents_dir = agents_dir

    def load(self, agent_id: str) -> AgentSpec:
        path = self._agents_dir / f"{agent_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"pipeline references agent {agent_id!r}, but {path} does not exist")
        return _load_agent_spec(path)


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        agent = _load_agent_spec(Path(args.agent_spec))
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run_kwargs = {"agent_id": agent.id, "task": args.task}
    if args.run_id:
        run_kwargs["run_id"] = args.run_id
    run = RunSpec(**run_kwargs)

    event_log = EventLog(run.run_id, output_root=Path(args.output_root))
    resolver = YamlCredentialStore()

    try:
        result = asyncio.run(execute_run(agent, run, resolver=resolver, event_log=event_log))
    except (CredentialNotFoundError, CredentialFileError) as exc:
        print(f"run {run.run_id} failed before starting: {exc}", file=sys.stderr)
        return 1

    print(f"run_id: {run.run_id}")
    print(f"events: {event_log.path}")

    if isinstance(result, ErrorEvent):
        print(f"run failed: {result.message}", file=sys.stderr)
        return 1

    print(f"final answer:\n{result.final_output}")
    return 0


def _cmd_pipeline_run(args: argparse.Namespace) -> int:
    try:
        pipeline = _load_pipeline_spec(Path(args.pipeline_spec))
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    run = PipelineRunSpec(pipeline_id=pipeline.id, task=args.task)
    agent_loader = DirectoryAgentSpecLoader(Path(args.agents_dir))
    resolver = YamlCredentialStore()

    def on_step(step_result):
        print(f"  [{step_result.status}] {step_result.step_id} (run_id={step_result.run_id})")

    print(f"pipeline_run_id: {run.pipeline_run_id}")
    try:
        record = asyncio.run(
            execute_pipeline(
                pipeline,
                run,
                agent_loader=agent_loader,
                resolver=resolver,
                output_root=Path(args.output_root),
                on_step=on_step,
            )
        )
    except (CredentialNotFoundError, CredentialFileError, FileNotFoundError) as exc:
        print(f"pipeline run {run.pipeline_run_id} failed before starting: {exc}", file=sys.stderr)
        return 1

    if record.status != "completed":
        print(f"pipeline failed: {record.error}", file=sys.stderr)
        return 1

    print(f"pipeline completed. final step output:\n{record.steps[-1].output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sandbox")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one agent against one task")
    run_parser.add_argument("agent_spec", help="Path to an AgentSpec YAML file")
    run_parser.add_argument("--task", required=True, help="The task/user message to seed the run with")
    run_parser.add_argument("--run-id", default=None, help="Override the generated run id")
    run_parser.add_argument(
        "--output-root", default="./runs", help="Directory to write runs/<run_id>/events.jsonl under"
    )
    run_parser.set_defaults(func=_cmd_run)

    pipeline_parser = subparsers.add_parser("pipeline", help="Pipeline commands")
    pipeline_subparsers = pipeline_parser.add_subparsers(dest="pipeline_command", required=True)

    pipeline_run_parser = pipeline_subparsers.add_parser("run", help="Run one pipeline against one seed task")
    pipeline_run_parser.add_argument("pipeline_spec", help="Path to a PipelineSpec YAML file")
    pipeline_run_parser.add_argument("--task", required=True, help="The seed task, available to steps as {{task}}")
    pipeline_run_parser.add_argument(
        "--agents-dir", default="./agents", help="Directory of <agent_id>.yaml files the pipeline's steps reference"
    )
    pipeline_run_parser.add_argument(
        "--output-root", default="./runs", help="Directory to write each step's runs/<run_id>/events.jsonl under"
    )
    pipeline_run_parser.set_defaults(func=_cmd_pipeline_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
