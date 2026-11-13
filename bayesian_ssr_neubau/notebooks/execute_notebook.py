"""Execute all cells into a local output copy with this exact Python."""
from pathlib import Path
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

import nbformat
from nbclient import NotebookClient
from jupyter_client import AsyncKernelManager
from jupyter_client.kernelspec import KernelSpecManager

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
RUNTIME = HERE / ".runtime"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def execute(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code_count = sum(c.cell_type == "code" for c in notebook.cells)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    started = time.perf_counter()
    identifier = uuid.uuid4().hex[:10]
    execution_dir = PROJECT / "outputs/notebook_executions" / identifier
    execution_dir.mkdir(parents=True, exist_ok=False)
    executed_path = execution_dir / path.name
    report_path = execution_dir / "execution.json"
    source_hash = sha(path)
    notebook.metadata["execution_evidence"] = {
        "status": "RUNNING", "python": sys.executable,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "report_file": report_path.name, "source_notebook_sha256": source_hash,
    }
    run_parent = PROJECT / "outputs/notebook_demos"
    runs_before = set(run_parent.iterdir()) if run_parent.exists() else set()

    def save():
        nbformat.write(notebook, executed_path)

    def on_cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"START cell {cell_index+1}/{len(notebook.cells)}: {cell.metadata.get('name', '')}", flush=True)

    def on_cell_executed(cell, cell_index, **kwargs):
        save()
        print(f"SAVED cell {cell_index+1}/{len(notebook.cells)} ({time.perf_counter()-started:.1f}s)", flush=True)

    RUNTIME.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kernel_", dir=RUNTIME) as temporary:
        temporary = Path(temporary).resolve()
        assert temporary.is_relative_to(RUNTIME.resolve())
        spec_dir = temporary / "python3"
        spec_dir.mkdir()
        kernel_spec = {
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Python (.venv Implementierung_BA)", "language": "python",
        }
        (spec_dir / "kernel.json").write_text(json.dumps(kernel_spec), encoding="utf-8")
        spec_manager = KernelSpecManager(kernel_dirs=[str(temporary)], ensure_native_kernel=False)
        km = AsyncKernelManager(kernel_name="python3", kernel_spec_manager=spec_manager)
        client = NotebookClient(notebook, km=km, kernel_name="python3", timeout=None,
                                startup_timeout=120, allow_errors=False, ipython_hist_file=":memory:",
                                resources={"metadata": {"path": str(HERE)}},
                                on_cell_start=on_cell_start, on_cell_executed=on_cell_executed)
        kernel_env = {**os.environ, "PYTHONUTF8": "1", "IPYTHONDIR": str(RUNTIME / "ipython"),
                      "JUPYTER_RUNTIME_DIR": str(RUNTIME / "jupyter")}
        start_options = {"env": kernel_env, "cleanup_kc": True}
        if sys.platform == "win32":
            start_options["creationflags"] = subprocess.CREATE_NO_WINDOW
        status, error = "FAILED", None
        try:
            task = asyncio.create_task(client.async_execute(**start_options))
            while not task.done():
                await asyncio.wait({task}, timeout=15)
                save()
                if not task.done():
                    completed = sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells)
                    print(f"RUNNING: {completed}/{code_count} code cells dispatched, {time.perf_counter()-started:.0f}s", flush=True)
            await task
            executed = [c for c in notebook.cells if c.cell_type == "code" and c.execution_count is not None]
            errors = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
            if len(executed) != code_count or errors:
                raise AssertionError("Not all code cells finished without errors")
            status = "COMPLETE"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            notebook.metadata["execution_evidence"]["status"] = status
            notebook.metadata["execution_evidence"]["elapsed_seconds"] = time.perf_counter()-started
            save()
            new_runs = sorted(set(run_parent.iterdir())-runs_before) if run_parent.exists() else []
            summaries = [p / "notebook_summary.json" for p in new_runs if (p / "notebook_summary.json").is_file()]
            summary = json.loads(summaries[0].read_text(encoding="utf-8")) if len(summaries) == 1 else None
            report = {
                "status": status, "notebook": str(executed_path.relative_to(ROOT)),
                "notebook_sha256": sha(executed_path),
                "source_notebook": str(path), "source_notebook_sha256": source_hash,
                "python": sys.executable, "code_cells": code_count,
                "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in notebook.cells),
                "error_outputs": sum(o.output_type == "error" for c in notebook.cells if c.cell_type == "code" for o in c.outputs),
                "elapsed_seconds": time.perf_counter()-started, "error": error,
                "run_directories": [str(p.relative_to(ROOT)) for p in new_runs], "summary": summary,
            }
            with report_path.open("x", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(json.dumps({k: report[k] for k in ("status", "executed_code_cells", "error_outputs", "elapsed_seconds", "run_directories")}, indent=2), flush=True)
            print(f"Executed notebook: {executed_path}", flush=True)
            print(f"Execution report: {report_path}", flush=True)
            if km.has_kernel:
                await km.shutdown_kernel(now=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=HERE / "Bayesian_SSR_HMM_Demonstration.ipynb")
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(execute(args.notebook.resolve()))


if __name__ == "__main__":
    main()
