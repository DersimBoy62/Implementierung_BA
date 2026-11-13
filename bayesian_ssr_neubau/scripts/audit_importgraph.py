"""AST-Audit der neuen Produktivimporte; keine alten Module ausführen."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path


def audit(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    source = root / "bayesian_ssr_neubau/src/bayesian_ring_ssr"
    historical = {p.stem for p in (root / "final_candidate_v52/final_core").glob("*.py")}
    forbidden_prefixes = ("final_candidate", "research_", "old_core", "results")
    modules, violations, dynamic = {}, [], []
    for path in sorted(source.rglob("*.py")):
        name = "bayesian_ring_ssr." + ".".join(path.relative_to(source).with_suffix("").parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [{"module": alias.name, "line": node.lineno, "relative": False} for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = ("." * node.level) + (node.module or "")
                imports.append({"module": target, "names": [a.name for a in node.names],
                                "line": node.lineno, "relative": bool(node.level)})
            elif isinstance(node, ast.Call):
                called = node.func
                if ((isinstance(called, ast.Name) and called.id == "__import__")
                    or (isinstance(called, ast.Attribute) and called.attr in ("import_module", "spec_from_file_location", "exec_module"))):
                    dynamic.append({"module": name, "line": node.lineno, "call": ast.unparse(called)})
        for record in imports:
            target = record["module"].lstrip(".")
            parts = target.split(".")
            if (any(part in historical for part in parts)
                or any(part.startswith(forbidden_prefixes) for part in parts)):
                violations.append({"source": name, **record})
        modules[name] = {"path": path.relative_to(root).as_posix(),
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "imports": imports}
    report = {"schema": "bayesian-ring-ssr-import-audit/1", "audit_date": "2026-11-13",
              "method": "Python AST for every src .py; historical names matched; dynamic imports flagged",
              "module_count": len(modules), "modules": modules, "legacy_import_violations": violations,
              "dynamic_import_calls": dynamic, "status": "PASS" if not violations and not dynamic else "REVIEW_REQUIRED",
              "scope": "Productive src only; read-only reference imports are restricted to tests/reference_parity"}
    output = root / "bayesian_ssr_neubau/docs/IMPORTGRAPH.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if violations or dynamic:
        raise AssertionError("Produktivimporte benötigen Review: " + str(violations + dynamic))
    print(f"PASS import graph: {len(modules)} modules, zero legacy imports, zero dynamic imports")
    return report


if __name__ == "__main__":
    audit()
