#!/usr/bin/env python3
"""
ast_parser.py — Multi-language AST / structure parser.
Outputs JSON to stdout describing the classes, methods, functions, and imports in a source file.

Supports:
  - Python  : native ast module (full fidelity)
  - C#      : regex-based (classes, methods, properties, fields, enums — handles Unity MonoBehaviours)
  - JS/TS   : regex-based (imports, classes, top-level functions)
  - Others  : returns language + line count only
"""

import ast
import sys
import json
import re
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# Python
# ═══════════════════════════════════════════════════════════════════════════════

def parse_python(source: str, filepath: str) -> dict:
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError as e:
        return {'language': 'python', 'error': str(e), 'imports': [], 'classes': [], 'functions': []}

    result = {
        'language': 'python',
        'module_docstring': ast.get_docstring(tree),
        'imports': [],
        'classes': [],
        'functions': []
    }

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                result['imports'].append({
                    'module': alias.name,
                    'alias': alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            result['imports'].append({
                'from': node.module or '',
                'names': [{'name': a.name, 'alias': a.asname} for a in node.names],
                'level': node.level  # relative import dots
            })
        elif isinstance(node, ast.ClassDef):
            result['classes'].append(_py_class(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result['functions'].append(_py_func(node))

    return result


def _py_class(node: ast.ClassDef) -> dict:
    methods, class_vars = [], []

    for item in ast.iter_child_nodes(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_py_func(item))
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            class_vars.append({
                'name': item.target.id,
                'type': _safe_unparse(item.annotation)
            })
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    class_vars.append({'name': t.id, 'type': None})

    return {
        'name': node.name,
        'line': node.lineno,
        'bases': [_safe_unparse(b) for b in node.bases],
        'decorators': [_safe_unparse(d) for d in node.decorator_list],
        'docstring': ast.get_docstring(node),
        'class_variables': class_vars[:20],
        'methods': methods
    }


def _py_func(node) -> dict:
    args = []
    # Positional args
    all_args = node.args.posonlyargs + node.args.args
    defaults_offset = len(all_args) - len(node.args.defaults)

    for i, arg in enumerate(all_args):
        info = {'name': arg.arg}
        if arg.annotation:
            info['type'] = _safe_unparse(arg.annotation)
        if i >= defaults_offset:
            info['default'] = _safe_unparse(node.args.defaults[i - defaults_offset])
        args.append(info)

    if node.args.vararg:
        args.append({'name': f'*{node.args.vararg.arg}', 'variadic': True})
    if node.args.kwarg:
        args.append({'name': f'**{node.args.kwarg.arg}', 'keyword_variadic': True})

    return {
        'name': node.name,
        'line': node.lineno,
        'is_async': isinstance(node, ast.AsyncFunctionDef),
        'decorators': [_safe_unparse(d) for d in node.decorator_list],
        'args': args,
        'returns': _safe_unparse(node.returns) if node.returns else None,
        'docstring': ast.get_docstring(node)
    }


def _safe_unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return '?'


# ═══════════════════════════════════════════════════════════════════════════════
# C#
# ═══════════════════════════════════════════════════════════════════════════════

def parse_csharp(source: str) -> dict:
    result = {
        'language': 'csharp',
        'usings': re.findall(r'using\s+([\w.]+)\s*;', source),
        'namespace': None,
        'classes': [],
        'enums': []
    }

    ns = re.search(r'namespace\s+([\w.]+)', source)
    if ns:
        result['namespace'] = ns.group(1)

    # Match class/struct/interface declarations
    type_pat = re.compile(
        r'(?:(?:public|private|protected|internal|static|abstract|sealed|partial)\s+)*'
        r'(class|struct|interface)\s+'
        r'(\w+)'                          # name
        r'(?:\s*<[^>]+>)?'               # optional generics
        r'(?:\s*:\s*([^\n{]+?))?'        # optional : Base, IFace
        r'\s*\n?\s*\{',
        re.MULTILINE
    )

    for m in type_pat.finditer(source):
        kind, name = m.group(1), m.group(2)
        bases_raw = m.group(3) or ''
        bases = [b.strip() for b in bases_raw.split(',') if b.strip()]

        body = _brace_block(source, m.end() - 1)
        result['classes'].append({
            'kind': kind,
            'name': name,
            'bases': bases,
            'fields':     _cs_fields(body),
            'properties': _cs_properties(body),
            'methods':    _cs_methods(body)
        })

    # Enums
    for m in re.finditer(r'enum\s+(\w+)\s*\{([^}]+)\}', source, re.MULTILINE):
        values = [
            v.strip().split('//')[0].strip().split('=')[0].strip()
            for v in m.group(2).split(',') if v.strip()
        ]
        result['enums'].append({'name': m.group(1), 'values': [v for v in values if v]})

    return result


def _brace_block(source: str, start: int) -> str:
    """Extract content between matching braces, starting at position of '{'."""
    depth = 0
    for i in range(start, len(source)):
        if source[i] == '{':
            depth += 1
        elif source[i] == '}':
            depth -= 1
            if depth == 0:
                return source[start + 1:i]
    return source[start:]


_CS_NOISE = {'if', 'while', 'for', 'foreach', 'switch', 'catch', 'using',
             'lock', 'get', 'set', 'return', 'new', 'else', 'yield'}

def _cs_methods(body: str) -> list:
    pat = re.compile(
        r'(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|new)'
        r'(?:\s+(?:public|private|protected|internal|static|virtual|override|abstract|async|sealed|new|readonly))*'
        r'\s+([\w<>\[\]?,.\s]+?)\s+(\w+)\s*'
        r'(?:<[^>]*>)?\s*'               # optional generics on method name
        r'\(([^)]*)\)',
        re.MULTILINE
    )
    methods, seen = [], set()
    for m in pat.finditer(body):
        ret, name, params_raw = m.group(1).strip(), m.group(2), m.group(3).strip()
        if name in _CS_NOISE:
            continue
        key = f"{name}/{len(params_raw.split(','))}"
        if key not in seen:
            seen.add(key)
            methods.append({'name': name, 'return_type': ret, 'params': _cs_params(params_raw)})
    return methods


def _cs_params(raw: str) -> list:
    params = []
    for p in raw.split(','):
        p = re.sub(r'\[[^\]]*\]', '', p).strip()   # strip [attributes]
        p = re.sub(r'^(out|ref|in|params)\s+', '', p)
        parts = p.rsplit(' ', 1)
        if len(parts) == 2:
            params.append({'type': parts[0].strip(), 'name': parts[1].strip()})
        elif p:
            params.append({'type': '?', 'name': p})
    return params


def _cs_fields(body: str) -> list:
    pat = re.compile(
        r'(?:\[[^\]]*\]\s*)?'
        r'(?:public|private|protected|internal|static|readonly|const)\s+'
        r'([\w<>\[\]?,.\s]+?)\s+(\w+)\s*(?:=|;)',
        re.MULTILINE | re.DOTALL
    )
    fields, seen = [], set()
    for m in pat.finditer(body):
        name = m.group(2)
        if name not in seen and not name.startswith('//'):
            seen.add(name)
            fields.append({'type': m.group(1).strip(), 'name': name})
        if len(fields) >= 30:
            break
    return fields


def _cs_properties(body: str) -> list:
    pat = re.compile(
        r'(?:public|protected|internal)\s+'
        r'(?:static\s+|virtual\s+|override\s+)?'
        r'([\w<>\[\]?,.\s]+?)\s+(\w+)\s*\{'
        r'[^}]*(?:get|set)',
        re.MULTILINE
    )
    props, seen = [], set()
    for m in pat.finditer(body):
        name = m.group(2)
        if name not in seen:
            seen.add(name)
            props.append({'type': m.group(1).strip(), 'name': name})
        if len(props) >= 20:
            break
    return props


# ═══════════════════════════════════════════════════════════════════════════════
# JavaScript / TypeScript (lightweight)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_javascript(source: str) -> dict:
    imports_es = re.findall(r"import\s+.*?from\s+['\"]([^'\"]+)['\"]", source)
    imports_cjs = re.findall(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", source)
    classes = re.findall(r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)', source)
    functions = re.findall(r'(?:export\s+)?(?:async\s+)?function\s+(\w+)', source)
    arrow_exports = re.findall(r'export\s+const\s+(\w+)\s*=\s*(?:async\s+)?\(', source)
    interfaces = re.findall(r'(?:export\s+)?interface\s+(\w+)', source)
    types = re.findall(r'(?:export\s+)?type\s+(\w+)\s*=', source)

    return {
        'language': 'javascript',
        'imports': imports_es + imports_cjs,
        'classes': [{'name': c} for c in classes],
        'interfaces': interfaces,
        'type_aliases': types,
        'functions': [{'name': f} for f in functions + arrow_exports]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch
# ═══════════════════════════════════════════════════════════════════════════════

def parse_file(filepath: str) -> dict:
    ext = Path(filepath).suffix.lower()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
    except Exception as e:
        return {'error': str(e), 'language': 'unknown', 'filepath': filepath}

    lines = source.count('\n') + 1

    if ext == '.py':
        result = parse_python(source, filepath)
    elif ext == '.cs':
        result = parse_csharp(source)
    elif ext in ('.js', '.ts', '.jsx', '.tsx'):
        result = parse_javascript(source)
    elif ext in ('.cpp', '.h', '.hpp'):
        # Minimal C++ — just flag it; regex C++ parsing is unreliable
        result = {'language': 'cpp', 'note': 'Use source directly; AST parsing not supported for C++'}
    else:
        result = {'language': 'unknown', 'extension': ext}

    result['filepath'] = filepath
    result['lines'] = lines
    return result


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: ast_parser.py <filepath>'}))
        sys.exit(1)
    print(json.dumps(parse_file(sys.argv[1]), indent=2))
