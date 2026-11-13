"""Run pytest and save actual test evidence in a new local output directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET


PROJECT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--name", help="New run name under outputs/verification/")
    destination.add_argument("--out", type=Path, help="New directory for this run's reports")
    parser.add_argument("--data", type=Path, help="Private input file; sets BAYESIAN_SSR_DATA for pytest")
    args = parser.parse_args(argv)
    name = args.name or (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                         + "_" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        parser.error("--name must be a single run name without directory separators")
    output = args.out.expanduser().resolve() if args.out else PROJECT / "outputs/verification" / name
    output.mkdir(parents=True, exist_ok=False)
    log, json_path, junit = (output / filename for filename in ("pytest.log", "verification.json", "junit.xml"))
    environment = os.environ.copy()
    if args.data is not None:
        environment["BAYESIAN_SSR_DATA"] = str(args.data.expanduser().resolve())
    command = [sys.executable, "-m", "pytest", str(PROJECT / "tests"), "-q", f"--junitxml={junit}"]
    started = time.perf_counter()
    result = subprocess.run(command, cwd=PROJECT, env=environment, capture_output=True,
                            text=True, encoding="utf-8", errors="replace")
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
    if junit.is_file():
        suites = list(ET.parse(junit).getroot().iter("testsuite"))
        for key in counts:
            counts[key] = sum(int(suite.attrib.get(key, 0)) for suite in suites)
    packages = {distribution.metadata["Name"]: distribution.version
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]}
    report = {
        "command": command, "cwd": str(PROJECT), "exitcode": result.returncode,
        "elapsed_seconds": time.perf_counter() - started, **counts,
        "junit_created": junit.is_file(), "python": sys.version,
        "data_override": environment.get("BAYESIAN_SSR_DATA"), "packages": packages,
    }
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(result.stdout + result.stderr, end="")
    print(json.dumps({"exitcode": result.returncode, **counts, "report": str(json_path)}, indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
