"""Verify the portable source snapshot without depending on generated runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT / "SOURCE_SHA256.json"


def source_hashes():
    paths = [PROJECT / "pyproject.toml"]
    paths.extend(sorted((PROJECT / "src").rglob("*.py")))
    paths.extend(sorted((PROJECT / "configs").rglob("*.json")))
    if not (PROJECT / "src/bayesian_ring_ssr/__init__.py").is_file():
        raise FileNotFoundError("Missing Bayesian SSR package sources")
    return {
        path.relative_to(PROJECT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Record an intentionally updated source snapshot")
    args = parser.parse_args(argv)
    actual = source_hashes()
    if args.write:
        report = {
            "schema": "bayesian-ring-ssr-source-sha256/1",
            "scope": "Production Python sources, JSON configurations and pyproject.toml; no run artifacts",
            "files": actual,
        }
        MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Recorded {len(actual)} source files: {MANIFEST}")
        return 0
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if expected.get("schema") != "bayesian-ring-ssr-source-sha256/1":
        raise ValueError("Unsupported source manifest schema")
    recorded = expected["files"]
    differences = [name for name in sorted(set(recorded) | set(actual))
                   if recorded.get(name) != actual.get(name)]
    if differences:
        raise ValueError("Source manifest mismatch: " + ", ".join(differences))
    print(f"PASS: {len(actual)} source files match SOURCE_SHA256.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
