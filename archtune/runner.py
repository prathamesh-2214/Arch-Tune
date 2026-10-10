"""Execute the project notebooks headlessly, in order, on the current machine.

Used by ``notebooks/run_all.ipynb`` (Colab: one runtime for everything) and by
``scripts/run_notebooks_local.py`` (a local GPU). Each notebook runs in a fresh Jupyter kernel
on the same machine, so the pinned stack is installed once.

Design for debugging:

* executed copies (with all outputs) are saved even when a cell fails, so the traceback is
  in the saved notebook *and* in the returned summary;
* the cell being executed is logged as it starts (long cells can be followed in
  ``results/<gpu>/progress.log``, written by the sweep);
* ``stop_on_error`` stops at the first failing notebook so later notebooks do not run on
  missing data.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

#: Execution order (01 needs the sweep from 02).
ORDER = ("00_setup_smoke", "02_sweep", "01_ir_exploration", "03_predictor", "04_search_eval",
         "05_compiler_experiments", "07_integration", "08_ptx_liveness")

LOCAL_BOOTSTRAP = '''# --- local bootstrap (injected by archtune.runner; replaces the Colab cell) ---
import os, sys
PROJECT = os.environ["ARCHTUNE_PROJECT"]
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
from archtune import env
ENV = env.capture()
MINI = env.mini_mode()
RESULTS = os.environ.get("ARCHTUNE_RESULTS") or os.path.join(PROJECT, "results_mini" if MINI else "results")
print("MINI mode" if MINI else "FULL mode", "->", RESULTS)
print(ENV)'''


def _first_line(src: str) -> str:
    for line in src.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            return line.strip()[:90]
    return src.strip()[:90]


def run_notebooks(project: str | Path, names: Sequence[str] = ORDER, *, mini: bool,
                  out_dir: str | Path, local_bootstrap: bool = False, results_dir: str | Path | None = None,
                  timeout: int | None = None, stop_on_error: bool = True,
                  log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Execute notebooks in order; return one summary dict per notebook.

    Args:
        project: project root (contains ``notebooks/`` and ``archtune/``).
        names: notebook names without ``.ipynb``.
        mini: set ``ARCHTUNE_MINI=1`` for the kernels (tiny workload, results in ``results_mini/``).
        out_dir: where executed copies and ``run_summary.md`` are written.
        local_bootstrap: replace each notebook's first code cell with :data:`LOCAL_BOOTSTRAP`
            (for machines without ``google.colab``).
        results_dir: override ``RESULTS`` in the kernels (``ARCHTUNE_RESULTS``).
        timeout: per-cell timeout in seconds (``None`` = unlimited, needed for the sweep).
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    project = Path(project).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARCHTUNE_PROJECT"] = str(project)
    os.environ["ARCHTUNE_MINI"] = "1" if mini else "0"
    if results_dir is not None:
        os.environ["ARCHTUNE_RESULTS"] = str(results_dir)
    else:
        os.environ.pop("ARCHTUNE_RESULTS", None)
    summaries: list[dict[str, Any]] = []
    for name in names:
        path = project / "notebooks" / f"{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        if local_bootstrap:
            first = next(c for c in nb.cells if c.cell_type == "code")
            first.source = LOCAL_BOOTSTRAP
        code_cells = [i for i, c in enumerate(nb.cells) if c.cell_type == "code"]
        state: dict[str, Any] = {"cell": None, "t": time.perf_counter()}

        def on_start(cell: Any, cell_index: int, **_: Any) -> None:
            state["cell"], state["t"] = cell_index, time.perf_counter()
            k = code_cells.index(cell_index) + 1 if cell_index in code_cells else "?"
            log(f"  [{name}] cell {k}/{len(code_cells)} start: {_first_line(cell.source)}")

        def on_done(cell: Any, cell_index: int, **_: Any) -> None:
            log(f"  [{name}] cell {cell_index} done in {time.perf_counter() - state['t']:.1f}s")

        client = NotebookClient(nb, timeout=timeout, kernel_name="python3", allow_errors=False,
                                resources={"metadata": {"path": str(project / "notebooks")}},
                                on_cell_start=on_start, on_cell_executed=on_done)
        log(f"[run] {name} ({'mini' if mini else 'full'}) started {datetime.now().strftime('%H:%M:%S')}")
        t0 = time.perf_counter()
        status, error = "ok", None
        try:
            client.execute()
        except CellExecutionError as exc:
            status, error = "cell_error", str(exc)
        except CellTimeoutError as exc:
            status, error = "timeout", str(exc)
        except DeadKernelError as exc:
            status, error = "kernel_died", f"{exc} (out of memory? os.kill restart in the bootstrap?)"
        except Exception as exc:  # noqa: BLE001
            status, error = "runner_error", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        saved = out_dir / f"{name}.ipynb"
        nbformat.write(nb, saved)
        summary = {"notebook": name, "status": status, "seconds": round(elapsed, 1),
                   "failed_cell": state["cell"] if status != "ok" else None,
                   "error": (error or "")[-3000:] or None, "saved": str(saved)}
        summaries.append(summary)
        log(f"[run] {name}: {status.upper()} in {elapsed:.0f}s -> {saved}")
        if error:
            log(error[-3000:])
        _write_summary(out_dir, summaries, mini)
        if status != "ok" and stop_on_error:
            log(f"[run] stopping: {name} failed (stop_on_error=True)")
            break
    return summaries


def _write_summary(out_dir: Path, summaries: list[dict[str, Any]], mini: bool) -> None:
    lines = [f"# Notebook run ({'mini' if mini else 'full'}) — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             "", "| notebook | status | seconds | failed cell |", "|---|---|---|---|"]
    lines += [f"| {s['notebook']} | **{s['status']}** | {s['seconds']} | {s['failed_cell'] if s['failed_cell'] is not None else ''} |"
              for s in summaries]
    for s in summaries:
        if s["error"]:
            lines += ["", f"## {s['notebook']}", "```", s["error"], "```"]
    (out_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def all_ok(summaries: Sequence[dict[str, Any]], expected: Sequence[str] = ORDER) -> bool:
    """True when every expected notebook ran and succeeded."""
    done = {s["notebook"]: s["status"] for s in summaries}
    return all(done.get(n) == "ok" for n in expected)
