#!/usr/bin/env python3
"""
doc_generator.py — OpenClaw Codebase Documentation Generator
Generates and maintains structured markdown docs for every code file in a mounted codebase.
Uses a local vLLM endpoint to write the docs; uses git diff or mtime to detect changes.

Usage:
    python3 doc_generator.py --full                        # Document all files
    python3 doc_generator.py --incremental                 # Only changed files (git diff / mtime)
    python3 doc_generator.py --files src/foo.py src/bar.cs # Specific files
    python3 doc_generator.py --architecture-only           # Rebuild architecture doc only
"""

import os
import sys
import json
import re
import argparse
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── Configuration (all overridable via env vars) ──────────────────────────────
WORKSPACE       = Path(os.environ.get('OPENCLAW_WORKSPACE', '/root/.openclaw/workspace'))
CODEBASE        = Path(os.environ.get('CODEBASE_PATH',     str(WORKSPACE / 'codebase')))
DOCS_DIR        = Path(os.environ.get('DOCS_OUTPUT',       str(WORKSPACE / 'docs')))
TOOLS_DIR       = WORKSPACE / 'tools'
VLLM_BASE_URL   = os.environ.get('VLLM_BASE_URL',  'http://localhost:8000/v1')
VLLM_API_KEY    = os.environ.get('VLLM_API_KEY',   'dummy-key')
MODEL           = os.environ.get('DOC_MODEL',       'Qwen/Qwen3-VL-32B-Instruct')
MAX_FILE_CHARS  = int(os.environ.get('DOC_MAX_FILE_CHARS', '12000'))
MAX_TOKENS      = int(os.environ.get('DOC_MAX_TOKENS',     '2048'))
VLLM_TIMEOUT    = int(os.environ.get('DOC_VLLM_TIMEOUT',   '300'))

LAST_COMMIT_FILE = DOCS_DIR / '.last_scan_commit'
MANIFEST_FILE    = DOCS_DIR / '.file_manifest.json'
INDEX_FILE       = DOCS_DIR / 'FILE_INDEX.md'
ARCH_FILE        = DOCS_DIR / 'ARCHITECTURE.md'

SUPPORTED_EXTENSIONS = {'.py', '.cs', '.js', '.ts', '.jsx', '.tsx', '.lua', '.cpp', '.h', '.hpp'}
EXCLUDE_DIRS = {
    '.git', '__pycache__', 'node_modules', '.venv', 'venv', 'env',
    'bin', 'obj', 'Build', 'Library', 'Temp', 'Logs', '.idea', '.vscode',
    'dist', 'build', '.cache', 'Packages', 'ProjectSettings',
}


# ── vLLM API ──────────────────────────────────────────────────────────────────
def call_vllm(messages: list, max_tokens: int = MAX_TOKENS) -> str:
    """POST to the vLLM chat completions endpoint and return the response text."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }).encode('utf-8')

    req = urllib.request.Request(
        f"{VLLM_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}"
        }
    )
    try:
        with urllib.request.urlopen(req, timeout=VLLM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        content = data["choices"][0]["message"]["content"]
        return content
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8', errors='replace')
        raise RuntimeError(f"vLLM HTTP {e.code}: {body[:400]}")
    except Exception as e:
        raise RuntimeError(f"vLLM call failed: {e}")


# ── AST Parsing ───────────────────────────────────────────────────────────────
def run_ast_parser(filepath: Path) -> Optional[dict]:
    """Run ast_parser.py and return parsed JSON, or None on failure."""
    parser_path = TOOLS_DIR / 'ast_parser.py'
    if not parser_path.exists():
        return None
    try:
        result = subprocess.run(
            ['python3', str(parser_path), str(filepath)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception as e:
        print(f"  ⚠  AST parse failed for {filepath.name}: {e}", file=sys.stderr)
    return None


# ── File Discovery ────────────────────────────────────────────────────────────
def find_all_code_files() -> list:
    files = []
    for root, dirs, filenames in os.walk(CODEBASE):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        for fname in sorted(filenames):
            p = Path(root) / fname
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)
    return sorted(files)


def get_current_commit() -> Optional[str]:
    r = subprocess.run(
        ['git', '-C', str(CODEBASE), 'log', '-1', '--format=%H'],
        capture_output=True, text=True, timeout=10
    )
    return r.stdout.strip() if r.returncode == 0 else None


def get_changed_files_git(all_files: list) -> Optional[list]:
    """Try git diff against last stored commit. Returns None if git unavailable."""
    if not LAST_COMMIT_FILE.exists():
        return None
    last_commit = LAST_COMMIT_FILE.read_text().strip()
    if not last_commit:
        return None
    r = subprocess.run(
        ['git', '-C', str(CODEBASE), 'diff', '--name-only', f'{last_commit}..HEAD'],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        print(f"  git diff failed: {r.stderr.strip()[:200]}")
        return None
    changed_rel = set(r.stdout.strip().splitlines())
    result = [f for f in all_files if str(f.relative_to(CODEBASE)) in changed_rel]
    print(f"  git diff: {len(result)} changed file(s) since {last_commit[:8]}")
    return result


def get_changed_files_mtime(all_files: list) -> Optional[list]:
    """Fallback: compare modification times against stored manifest."""
    if not MANIFEST_FILE.exists():
        return None
    manifest = json.loads(MANIFEST_FILE.read_text())
    changed = [
        f for f in all_files
        if str(f.relative_to(CODEBASE)) not in manifest
        or manifest[str(f.relative_to(CODEBASE))] != f.stat().st_mtime
    ]
    print(f"  mtime diff: {len(changed)} changed file(s)")
    return changed


def save_manifest(all_files: list) -> None:
    manifest = {str(f.relative_to(CODEBASE)): f.stat().st_mtime for f in all_files}
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2))


# ── Per-File Doc Generation ───────────────────────────────────────────────────
FILE_DOC_SYSTEM = """You are a technical documentation writer. You generate concise, structured markdown documentation for source code files.

The docs will be fed to an AI development assistant as context — so they must be information-dense and accurate, not verbose or padded. Prefer tables over prose for method listings.

Output ONLY the markdown document. No preamble. Start directly with the # heading."""


def build_file_doc_prompt(rel_path: str, ast_data: Optional[dict], source: str) -> str:
    ast_section = json.dumps(ast_data, indent=2)[:3000] if ast_data else "(AST unavailable — infer from source)"
    truncated = source[:MAX_FILE_CHARS]
    trunc_note = f"\n\n[File truncated at {MAX_FILE_CHARS} chars — {len(source) - MAX_FILE_CHARS} chars omitted]" \
        if len(source) > MAX_FILE_CHARS else ""
    date = datetime.now().strftime('%Y-%m-%d')

    return f"""Document this source file for an AI development assistant.

**File:** `{rel_path}`

=== AST STRUCTURE ===
{ast_section}

=== SOURCE ===
```
{truncated}{trunc_note}
```

Write a markdown document with these sections (skip sections with nothing to say):

# `{rel_path}`
**Language:** X | **Lines:** N | **Last Scanned:** {date}

## Purpose
2-3 sentences: what this file does and its role in the project.

## Dependencies
Bullet list of key imports and why they matter. Skip trivial stdlib.

## Classes
For each class: one-line role description, inheritance note, then a method table:
| Method | Signature | Description |
|--------|-----------|-------------|

## Functions
For top-level / standalone functions (not class methods):
| Function | Signature | Description |
|----------|-----------|-------------|

## Notes
(Optional) Important patterns, gotchas, architectural decisions, or anything non-obvious.

Keep the whole document under 300 lines."""


def generate_file_doc(filepath: Path) -> Optional[str]:
    rel_path = str(filepath.relative_to(CODEBASE))
    try:
        source = filepath.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        print(f"  ✗ Cannot read {rel_path}: {e}")
        return None

    ast_data = run_ast_parser(filepath)
    prompt = build_file_doc_prompt(rel_path, ast_data, source)

    try:
        return call_vllm([
            {"role": "system", "content": FILE_DOC_SYSTEM},
            {"role": "user", "content": prompt}
        ])
    except RuntimeError as e:
        print(f"  ✗ vLLM failed for {rel_path}: {e}")
        return None


def write_file_doc(filepath: Path, doc: str) -> Path:
    rel_path = filepath.relative_to(CODEBASE)
    out_path = DOCS_DIR / 'files' / (str(rel_path) + '.md')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding='utf-8')
    return out_path


# ── Architecture Doc ──────────────────────────────────────────────────────────
ARCH_SYSTEM = """You are a software architect writing a technical overview document for an AI development assistant.

Output ONLY the markdown. No preamble. Start directly with the # heading."""


def build_file_tree_text(all_files: list) -> str:
    """Build an indented text tree of the codebase."""
    lines = [f"{CODEBASE.name}/"]
    by_dir: dict = {}
    for f in sorted(all_files):
        rel = f.relative_to(CODEBASE)
        by_dir.setdefault(rel.parent, []).append(rel.name)

    last_dir = None
    for directory in sorted(by_dir.keys()):
        if str(directory) != '.':
            depth = len(directory.parts)
            indent = '    ' * (depth - 1)
            lines.append(f"{indent}├── {directory.name}/")
        else:
            depth = 0
            indent = ''
        file_indent = '    ' * depth
        for fname in sorted(by_dir[directory]):
            lines.append(f"{file_indent}│   ├── {fname}")
    return '\n'.join(lines)


def extract_purpose(doc: str) -> str:
    """Pull the Purpose section text from a generated doc."""
    lines, in_purpose = doc.splitlines(), False
    parts = []
    for line in lines:
        if '## Purpose' in line:
            in_purpose = True
            continue
        if in_purpose:
            if line.startswith('## '):
                break
            if line.strip():
                parts.append(line.strip())
    return ' '.join(parts)[:220]


def generate_architecture_doc(all_files: list, file_docs: dict) -> str:
    file_tree = build_file_tree_text(all_files)
    summaries = []
    for f in sorted(all_files):
        rel = str(f.relative_to(CODEBASE))
        if rel in file_docs:
            purpose = extract_purpose(file_docs[rel])
            if purpose:
                summaries.append(f"- `{rel}`: {purpose}")

    summaries_text = '\n'.join(summaries[:100])  # stay within context budget

    prompt = f"""Generate an architecture overview for this codebase. It will be fed to an AI assistant as context.

=== FILE TREE ===
{file_tree}

=== FILE PURPOSES ===
{summaries_text}

Write a markdown document:

# Architecture Overview
**Project:** <inferred name> | **Generated:** {datetime.now().strftime('%Y-%m-%d')} | **Files:** {len(all_files)}

## Project Overview
3-5 sentences: what this project does, its domain, primary purpose.

## Folder Structure
Annotated tree — what each major directory is responsible for.

## Key Components
3-8 bullet points: most important modules/subsystems and what they do.

## Data Flow
ASCII diagram or bullet list showing how data moves through the system
(e.g., input → transform → output, or request → handler → storage).

## Entry Points
How the project is typically invoked: CLI commands, main scripts, Unity Play mode, etc.

## External Dependencies
Key external libraries/frameworks inferred from imports, and their role.

Keep under 200 lines."""

    try:
        return call_vllm(
            [{"role": "system", "content": ARCH_SYSTEM}, {"role": "user", "content": prompt}],
            max_tokens=3000
        )
    except RuntimeError as e:
        print(f"  ✗ Architecture generation failed: {e}")
        return f"# Architecture Overview\n\n*Generation failed: {e}*\n\nSee `FILE_INDEX.md` for individual file summaries."


# ── File Index ────────────────────────────────────────────────────────────────
def update_file_index(all_files: list, file_docs: dict) -> None:
    lines = [
        "# File Index",
        f"*Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')} — {len(all_files)} files documented*",
        "",
        "See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the high-level overview.",
        "Individual file docs are in `docs/files/` mirroring the source tree.",
        ""
    ]

    by_dir: dict = {}
    for f in sorted(all_files):
        rel = str(f.relative_to(CODEBASE))
        top = rel.split(os.sep)[0] if os.sep in rel else rel.split('/')[0] if '/' in rel else '.'
        purpose = extract_purpose(file_docs[rel]) if rel in file_docs else ''
        by_dir.setdefault(top, []).append((rel, purpose))

    for top_dir in sorted(by_dir.keys()):
        lines += [f"## `{top_dir}/`", "", "| File | Purpose |", "|------|---------|"]
        for rel, purpose in sorted(by_dir[top_dir]):
            name = Path(rel).name
            link = f"files/{rel}.md"
            lines.append(f"| [`{name}`]({link}) | {purpose} |")
        lines.append("")

    INDEX_FILE.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  ✓ FILE_INDEX.md ({len(all_files)} entries)")


# ── Orchestration ─────────────────────────────────────────────────────────────
def run(mode: str, specific_files: Optional[list] = None, architecture_only: bool = False):
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / 'files').mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{'='*62}")
    print(f"  OpenClaw Doc Generator  |  {mode.upper()}  |  {ts}")
    print(f"  Codebase : {CODEBASE}")
    print(f"  Docs out : {DOCS_DIR}")
    print(f"  Model    : {MODEL}")
    print(f"{'='*62}\n")

    # ── Load existing docs from disk ────────────────────────────────────────
    def load_existing_docs(all_files_list: list) -> dict:
        docs = {}
        for f in all_files_list:
            rel = str(f.relative_to(CODEBASE))
            doc_path = DOCS_DIR / 'files' / (rel + '.md')
            if doc_path.exists():
                docs[rel] = doc_path.read_text()
        return docs

    # ── Architecture-only rebuild ────────────────────────────────────────────
    if architecture_only:
        all_files = find_all_code_files()
        print(f"Rebuilding architecture from {len(all_files)} existing file docs...")
        existing = load_existing_docs(all_files)
        arch = generate_architecture_doc(all_files, existing)
        ARCH_FILE.write_text(arch, encoding='utf-8')
        print("  ✓ ARCHITECTURE.md")
        return

    # ── Determine target files ───────────────────────────────────────────────
    all_files = find_all_code_files()

    if specific_files:
        files_to_process = [CODEBASE / f for f in specific_files if (CODEBASE / f).exists()]
        missing = [f for f in specific_files if not (CODEBASE / f).exists()]
        if missing:
            print(f"  ⚠  Not found: {missing}")
    elif mode == 'full':
        files_to_process = all_files
    elif mode == 'incremental':
        changed = get_changed_files_git(all_files) or get_changed_files_mtime(all_files)
        if changed is None:
            print("  No previous state found — running full scan.")
            files_to_process = all_files
            mode = 'full'
        else:
            files_to_process = changed
    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"Codebase total : {len(all_files)} files")
    print(f"To document    : {len(files_to_process)} files\n")

    existing_docs = load_existing_docs(all_files)

    # ── Process each file ────────────────────────────────────────────────────
    new_docs, failed = {}, []
    for i, filepath in enumerate(files_to_process):
        rel = str(filepath.relative_to(CODEBASE))
        print(f"[{i+1}/{len(files_to_process)}] {rel}")
        doc = generate_file_doc(filepath)
        if doc:
            write_file_doc(filepath, doc)
            new_docs[rel] = doc
            print(f"  ✓")
        else:
            failed.append(rel)

    existing_docs.update(new_docs)

    # ── Update index ─────────────────────────────────────────────────────────
    print("\nUpdating FILE_INDEX.md...")
    update_file_index(all_files, existing_docs)

    # ── Update architecture (always on full; on incremental if >10% changed) ─
    arch_threshold = max(1, len(all_files) // 10)
    if mode == 'full' or len(new_docs) >= arch_threshold:
        print("\nGenerating ARCHITECTURE.md...")
        arch = generate_architecture_doc(all_files, existing_docs)
        ARCH_FILE.write_text(arch, encoding='utf-8')
        print("  ✓ ARCHITECTURE.md")
    else:
        print(f"\n(Architecture regen skipped — {len(new_docs)}/{arch_threshold} threshold not met)")

    # ── Save state for next incremental run ──────────────────────────────────
    commit = get_current_commit()
    if commit:
        LAST_COMMIT_FILE.write_text(commit, encoding='utf-8')
        print(f"\nStored commit: {commit[:12]}")
    save_manifest(all_files)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Done!  Documented: {len(new_docs)}  |  Failed: {len(failed)}")
    if failed:
        print(f"  Failed files: {', '.join(failed[:5])}{'...' if len(failed) > 5 else ''}")
    print(f"  Docs at: {DOCS_DIR}")
    print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description='OpenClaw Documentation Generator')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--full',              action='store_true', help='Document all files')
    g.add_argument('--incremental',       action='store_true', help='Document changed files only')
    g.add_argument('--files',  nargs='+', metavar='FILE',      help='Document specific files (relative to codebase root)')
    g.add_argument('--architecture-only', action='store_true', help='Rebuild architecture doc from existing file docs')
    args = p.parse_args()

    if args.full:
        run('full')
    elif args.incremental:
        run('incremental')
    elif args.files:
        run('files', specific_files=args.files)
    elif args.architecture_only:
        run('', architecture_only=True)


if __name__ == '__main__':
    main()