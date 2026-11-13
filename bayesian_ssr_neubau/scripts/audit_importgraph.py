"""Audit production imports without executing historical reference modules."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


DEPENDENCY_MODULES = {"numpy", "scipy", "sklearn", "pandas", "numba", "threadpoolctl"}
FORBIDDEN_PREFIXES = ("final_candidate", "research_", "old_core", "results")


def audit(root=None, out=None):
    """Check a project (or its workspace) and write a generated JSON report."""
    base = Path(root).resolve() if root is not None else Path(__file__).resolve().parents[1]
    project = base if (base / "pyproject.toml").is_file() else base / "bayesian_ssr_neubau"
    source = project / "src/bayesian_ring_ssr"
    paths = sorted(source.rglob("*.py")) if source.is_dir() else []
    if not paths or not (source / "__init__.py").is_file():
        raise FileNotFoundError(f"Production package is missing or empty: {source}")

    fixture_core = project / "tests/reference_parity/fixtures/legacy/final_candidate_v52/final_core"
    historical = {path.stem for path in fixture_core.glob("*.py")}
    if not historical:
        raise FileNotFoundError(f"Vendored reference sources are missing: {fixture_core}")

    names = {}
    for path in paths:
        parts = list(path.relative_to(source).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        names[path] = ".".join(("bayesian_ring_ssr", *parts))
    local_modules = set(names.values())
    modules, legacy, unapproved, dynamic = {}, [], [], []

    for path, name in names.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend({"module": item.name, "line": node.lineno, "relative": False}
                               for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                record = {"module": target, "names": [item.name for item in node.names],
                          "line": node.lineno, "relative": bool(node.level)}
                try:
                    record["resolved_module"] = (importlib.util.resolve_name(target, package)
                                                 if node.level else target)
                except (ImportError, ValueError) as error:
                    record["resolution_error"] = str(error)
                imports.append(record)
            elif isinstance(node, ast.Call):
                called = node.func
                if ((isinstance(called, ast.Name) and called.id == "__import__")
                        or (isinstance(called, ast.Attribute) and called.attr in
                            ("import_module", "spec_from_file_location", "exec_module"))):
                    dynamic.append({"source": name, "line": node.lineno,
                                    "call": ast.unparse(called)})

        for record in imports:
            target = record.get("resolved_module", record["module"].lstrip("."))
            parts = target.split(".")
            if (any(part in historical for part in parts)
                    or any(part.startswith(FORBIDDEN_PREFIXES) for part in parts)):
                legacy.append({"source": name, **record})
            top_level = parts[0]
            if ("resolution_error" in record
                    or (top_level == "bayesian_ring_ssr" and target not in local_modules)
                    or (top_level != "bayesian_ring_ssr"
                        and top_level not in sys.stdlib_module_names
                        and top_level not in DEPENDENCY_MODULES)):
                unapproved.append({"source": name, **record})
        modules[name] = {"path": path.relative_to(project).as_posix(),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "imports": imports}

    failed = bool(legacy or unapproved or dynamic)
    report = {
        "schema": "bayesian-ring-ssr-import-audit/2",
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "method": "AST audit; standard library, declared dependency and local module allowlist",
        "module_count": len(modules), "modules": modules,
        "legacy_import_violations": legacy, "unapproved_imports": unapproved,
        "dynamic_import_calls": dynamic,
        "status": "REVIEW_REQUIRED" if failed else "PASS",
        "scope": "Production src only; historical sources are restricted to test fixtures",
    }
    output = Path(out).resolve() if out is not None else project / "outputs/verification/importgraph.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failed:
        raise AssertionError(f"Production import audit failed; inspect {output}")
    print(f"PASS import graph: {len(modules)} modules; report: {output}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Generated JSON report path")
    args = parser.parse_args()
    audit(out=args.out)


if __name__ == "__main__":
    main()
