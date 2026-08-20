#!/usr/bin/env python3
"""
Renders a completed run's report files (whatever exists under
runs/<run_id>/ -- works for a full orchestrator run OR a handful of
standalone single-agent script runs sharing a --run-id) into one
self-contained HTML file: no external CSS/JS, nothing else to install,
open it in any browser or send it to someone.

Deliberately NOT a Jupyter notebook (the ML pipeline's own convention
for viewing results) -- this is meant to be handed to someone who isn't
going to set up a Python environment first. Pure stdlib, no new
dependency.

Usage:
    python scripts/render_run_report.py --run-id run_260117_140000_ab12
    python scripts/render_run_report.py            # most recent run
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resource_scheduler.paths import runs_root


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _most_recent_run_dir() -> Optional[Path]:
    base = runs_root()
    if not base.exists():
        return None
    run_dirs = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.name)
    return run_dirs[-1] if run_dirs else None


def _esc(value) -> str:
    return html.escape(str(value), quote=True)


def _json_block(value) -> str:
    if value is None:
        return "<span class='muted'>none</span>"
    return f"<pre>{_esc(json.dumps(value, indent=2, default=str))}</pre>"


def _badge(label: str, ok: bool) -> str:
    cls = "badge-ok" if ok else "badge-warn"
    return f"<span class='badge {cls}'>{_esc(label)}</span>"


def _transcript_section(run_dir: Path, agent_name: str) -> str:
    transcripts_dir = run_dir / "transcripts"
    if not transcripts_dir.exists():
        return ""
    matches = sorted(transcripts_dir.glob(f"{agent_name}_*.json"))
    if not matches:
        return ""
    parts = []
    for path in matches:
        messages = _load_json(path)
        if messages is None:
            continue
        rows = []
        for m in messages:
            role = m.get("role", "?")
            content = m.get("content")
            tool_calls = m.get("tool_calls")
            body = ""
            if content is not None:
                body += _json_block(content) if isinstance(content, (dict, list)) else f"<div class='msg-text'>{_esc(content)}</div>"
            if tool_calls:
                body += _json_block(tool_calls)
            rows.append(f"<div class='msg msg-{_esc(role)}'><div class='msg-role'>{_esc(role)}</div>{body}</div>")
        parts.append(
            f"<details><summary>Transcript: {_esc(path.name)}</summary><div class='transcript'>{''.join(rows)}</div></details>"
        )
    return "".join(parts)


def _agent_section(title: str, run_dir: Path, agent_key: str, report: Optional[dict], summary_html: str) -> str:
    if report is None:
        return f"<section><h2>{_esc(title)}</h2><p class='muted'>No report file found for this agent in this run.</p></section>"
    transcript_html = _transcript_section(run_dir, agent_key)
    return (
        f"<section><h2>{_esc(title)}</h2>"
        f"{summary_html}"
        f"<details><summary>Raw report JSON</summary>{_json_block(report)}</details>"
        f"{transcript_html}"
        f"</section>"
    )


def render_html(run_dir: Path) -> str:
    orchestrator_report = _load_json(run_dir / "orchestrator_report.json")
    lm = _load_json(run_dir / "load_monitor_report.json")
    tp = _load_json(run_dir / "task_prioritization_report.json")
    ra = _load_json(run_dir / "resource_allocation_report.json")
    fr = _load_json(run_dir / "failure_recovery_report.json")
    opt = _load_json(run_dir / "optimization_report.json")
    ho = _load_json(run_dir / "human_oversight_report.json")

    run_id = run_dir.name
    status = (orchestrator_report or {}).get("status", "unknown")
    model = (orchestrator_report or {}).get("model", "?")
    issues = (orchestrator_report or {}).get("issues", [])
    final_summary = (orchestrator_report or {}).get("final_summary")

    issues_html = (
        "<ul class='issues'>" + "".join(f"<li>{_esc(i)}</li>" for i in issues) + "</ul>"
        if issues else "<p class='muted'>None flagged.</p>"
    )

    lm_summary = ""
    if lm:
        n_flags = len((lm.get("deterministic_report") or {}).get("flags", []))
        lm_summary = f"<p>{_badge(f'{n_flags} flag(s)', n_flags == 0)} " \
                      f"{_badge('narration matched', not lm.get('flag_mismatch'))}</p>"

    tp_summary = ""
    if tp:
        tp_summary = (
            f"<p>{_badge('valid' if tp.get('valid') else 'invalid', bool(tp.get('valid')))} "
            f"{_badge('sent to Resource Allocation' if tp.get('sent_to_resource_allocation') else 'not sent', bool(tp.get('sent_to_resource_allocation')))}</p>"
        )
        ranked = ((tp.get("llm_proposal") or {}).get("ranked_task_ids")) or []
        if ranked:
            tp_summary += f"<p class='muted'>Ranked {len(ranked)} task(s): {_esc(', '.join(ranked[:8]))}{'…' if len(ranked) > 8 else ''}</p>"

    ra_summary = ""
    if ra:
        accepted = ra.get("accepted_assignments") or []
        env_rej = ra.get("environment_rejected") or []
        ra_summary = (
            f"<p>{_badge(f'{len(accepted)} accepted', True)} "
            f"{_badge(f'{len(env_rej)} environment-rejected', len(env_rej) == 0)}</p>"
        )
        if accepted:
            rows = "".join(
                f"<tr><td>{_esc(a.get('task_id'))}</td><td>{_esc(a.get('machine_id'))}</td>"
                f"<td>{_esc(a.get('network_slice_id'))}</td><td>{_esc(a.get('rationale', ''))}</td></tr>"
                for a in accepted
            )
            ra_summary += (
                "<table><tr><th>Task</th><th>Machine</th><th>Slice</th><th>Rationale</th></tr>" + rows + "</table>"
            )

    fr_summary = ""
    if fr:
        n_inc = len(fr.get("incidents") or [])
        n_aff = len(fr.get("affected_tasks") or [])
        fr_summary = f"<p>{_badge(f'{n_inc} incident(s)', n_inc == 0)} {_badge(f'{n_aff} affected task(s)', n_aff == 0)}</p>"
        rv = fr.get("reroute_validation")
        if rv:
            n_rv_accepted = len(rv.get("accepted_reroutes") or [])
            n_rv_rejected = len(rv.get("environment_rejected") or [])
            fr_summary += (
                f"<p>{_badge(f'{n_rv_accepted} reroutes accepted', True)} "
                f"{_badge(f'{n_rv_rejected} environment-rejected', n_rv_rejected == 0)}</p>"
            )

    opt_summary = ""
    if opt:
        opt_summary = f"<p>{_badge('valid' if opt.get('valid') else 'invalid/none', bool(opt.get('valid')))} " \
                       f"{_badge('sent to Human Oversight' if opt.get('sent_to_human_oversight') else 'not sent', bool(opt.get('sent_to_human_oversight')))}</p>"
        ev = opt.get("evidence") or {}
        if ev:
            opt_summary += f"<p class='muted'>Evidence scanned {ev.get('n_runs_scanned')} run(s).</p>"

    ho_summary = ""
    if ho:
        verdict = ho.get("verdict", "?")
        ho_summary = f"<p>{_badge(f'verdict: {verdict}', verdict == 'approved')}</p>"
        if ho.get("reasoning"):
            ho_summary += f"<p class='muted'>{_esc(ho['reasoning'])}</p>"

    summary_block = (
        f"<section class='summary'><h2>Final summary</h2><p>{_esc(final_summary)}</p></section>"
        if final_summary else ""
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Run {_esc(run_id)}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; background: #fff; }}
  h1 {{ font-size: 1.5rem; }}
  h2 {{ font-size: 1.15rem; border-bottom: 1px solid #e2e2e2; padding-bottom: 0.3rem; margin-top: 2rem; }}
  .muted {{ color: #666; font-size: 0.9rem; }}
  .meta {{ color: #555; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.55rem; border-radius: 1rem; font-size: 0.8rem; margin-right: 0.4rem; }}
  .badge-ok {{ background: #e3f5e9; color: #1a7a3d; }}
  .badge-warn {{ background: #fdecea; color: #b3261e; }}
  .issues {{ background: #fff8e6; border: 1px solid #f0d98a; border-radius: 6px; padding: 0.75rem 1.25rem; }}
  .summary {{ background: #f4f7ff; border-radius: 8px; padding: 1rem 1.25rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 0.5rem; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  pre {{ background: #f5f5f5; padding: 0.75rem; border-radius: 6px; overflow-x: auto; font-size: 0.8rem; }}
  details {{ margin-top: 0.75rem; }}
  summary {{ cursor: pointer; color: #2554c7; font-size: 0.9rem; }}
  .transcript {{ margin-top: 0.5rem; }}
  .msg {{ border-left: 3px solid #ccc; padding: 0.4rem 0.75rem; margin-bottom: 0.5rem; }}
  .msg-role {{ font-weight: 600; font-size: 0.75rem; text-transform: uppercase; color: #666; }}
  .msg-system {{ border-color: #999; }}
  .msg-user {{ border-color: #2554c7; }}
  .msg-assistant {{ border-color: #1a7a3d; }}
  .msg-tool {{ border-color: #b3261e; }}
  .msg-text {{ white-space: pre-wrap; font-size: 0.9rem; }}
</style></head>
<body>
<h1>Resource Scheduler — Run {_esc(run_id)}</h1>
<div class="meta">status: {_esc(status)} &nbsp;|&nbsp; model: {_esc(model)}</div>

{summary_block}

<section><h2>Issues flagged</h2>{issues_html}</section>

{_agent_section("1 — Load Monitor", run_dir, "load_monitor", lm, lm_summary)}
{_agent_section("2 — Task Prioritization", run_dir, "task_prioritization", tp, tp_summary)}
{_agent_section("3 — Resource Allocation", run_dir, "resource_allocation", ra, ra_summary)}
{_agent_section("4 — Failure Recovery", run_dir, "failure_recovery", fr, fr_summary)}
{_agent_section("5 — Optimization", run_dir, "optimization", opt, opt_summary)}
{_agent_section("6 — Human Oversight", run_dir, "human_oversight", ho, ho_summary)}

</body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None, help="defaults to the most recent run under runs/")
    args = parser.parse_args()

    if args.run_id:
        run_dir = runs_root() / args.run_id
        if not run_dir.exists():
            print(f"No such run directory: {run_dir}")
            sys.exit(1)
    else:
        run_dir = _most_recent_run_dir()
        if run_dir is None:
            print(f"No runs found under {runs_root()} — run an agent or the orchestrator first.")
            sys.exit(1)
        print(f"Using most recent run: {run_dir.name}")

    out_path = run_dir / "report.html"
    out_path.write_text(render_html(run_dir), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
