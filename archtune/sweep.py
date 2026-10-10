"""Resumable exhaustive sweep (spec §7-8) writing incremental parquet parts.

Layout under ``<results_dir>/<gpu_tag>/``::

    <table>/part-<session>-<n>.parquet   # incremental parts (append-only, Drive-friendly)
    <table>.parquet, <table>.csv         # consolidated by :func:`consolidate`
    artifacts/<kernel>/<cfg_hash>/       # gzip PTX / cubin / SASS
    sessions.jsonl                       # per-session estimate vs actual time

Rows are keyed by (gpu_tag, kernel, shape_id, cfg_hash, run_id). ``run_id`` names the
*experiment* (e.g. ``"sweep_v1"``), not the Colab session, so a resumed session skips rows
written by earlier sessions. Pure helpers (keys, pending tasks, estimates, row store) are
CPU-testable; :func:`evaluate_one` and :func:`run_sweep` need a GPU.
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from archtune.space import config_hash, split_launch
from archtune.workloads import Shape

KEY_COLUMNS = ("gpu_tag", "kernel", "shape_id", "cfg_hash", "run_id")
DEFAULT_RUN_ID = "sweep_v1"


@dataclass(frozen=True)
class Task:
    """One (shape, config) evaluation."""

    shape: Shape
    cfg: dict[str, Any]

    @property
    def cfg_hash(self) -> str:
        """Stable config hash."""
        return config_hash(self.cfg)


def task_key(gpu_tag: str, kernel: str, shape_id: str, cfg_hash: str, run_id: str) -> tuple[str, ...]:
    """Resume key of one row."""
    return (gpu_tag, kernel, shape_id, cfg_hash, run_id)


def make_tasks(shapes: Sequence[Shape], configs_for: Callable[[Shape], Iterable[dict[str, Any]]]
               ) -> list[Task]:
    """Shape-major task list (all configs of a shape are consecutive: inputs are reused)."""
    return [Task(s, dict(c)) for s in shapes for c in configs_for(s)]


def pending(tasks: Sequence[Task], done: set[tuple[str, ...]], gpu_tag: str, run_id: str) -> list[Task]:
    """Tasks whose key is not in *done* (order preserved, duplicates removed)."""
    seen: set[tuple[str, ...]] = set(done)
    out = []
    for t in tasks:
        k = task_key(gpu_tag, t.shape.kernel, t.shape.shape_id, t.cfg_hash, run_id)
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


class RowStore:
    """Append-only parquet store: buffered rows flushed to small part files.

    Parquet cannot be appended in place, so each flush writes a new part; :meth:`load`
    concatenates parts and drops duplicate keys (keeping the latest).
    """

    def __init__(self, directory: str | os.PathLike[str], flush_every: int = 25,
                 flush_seconds: float = 120.0, session: str | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = flush_every
        self.flush_seconds = flush_seconds
        self.session = session or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self._buf: list[dict[str, Any]] = []
        self._n = 0
        self._last = time.monotonic()

    def append(self, row: dict[str, Any]) -> None:
        """Buffer one row; flush when the buffer is full or old."""
        self._buf.append(row)
        if len(self._buf) >= self.flush_every or time.monotonic() - self._last > self.flush_seconds:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows as a new part file."""
        if not self._buf:
            return
        df = pd.DataFrame(self._buf)
        path = self.dir / f"part-{self.session}-{self._n:05d}.parquet"
        tmp = path.with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        self._n += 1
        self._buf = []
        self._last = time.monotonic()

    def load(self) -> pd.DataFrame:
        """All rows written so far (deduplicated on :data:`KEY_COLUMNS`, latest wins)."""
        return load_parts(self.dir)

    def done_keys(self) -> set[tuple[str, ...]]:
        """Keys already present on disk."""
        return keys_of(self.load())


def load_parts(directory: str | os.PathLike[str]) -> pd.DataFrame:
    """Concatenate ``part-*.parquet`` files in *directory* and drop duplicate keys."""
    parts = sorted(Path(directory).glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    keys = [c for c in KEY_COLUMNS if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df


def keys_of(df: pd.DataFrame) -> set[tuple[str, ...]]:
    """Set of resume keys present in a DataFrame."""
    if df.empty or not set(KEY_COLUMNS) <= set(df.columns):
        return set()
    return set(map(tuple, df[list(KEY_COLUMNS)].astype(str).itertuples(index=False, name=None)))


def consolidate(gpu_dir: str | os.PathLike[str], table: str = "sweep") -> pd.DataFrame:
    """Merge parts of *table* into ``<table>.parquet`` and ``<table>.csv``."""
    gpu_dir = Path(gpu_dir)
    df = load_parts(gpu_dir / table)
    if not df.empty:
        df.to_parquet(gpu_dir / f"{table}.parquet", index=False)
        df.drop(columns=[c for c in df.columns if c == "error_msg"]).to_csv(gpu_dir / f"{table}.csv", index=False)
    return df


def estimate_seconds(sample: pd.DataFrame, n_tasks: int, n_cold_compiles: int) -> dict[str, float]:
    """Extrapolate the full sweep time from a small-first sample.

    Cost per task = correctness + benchmark (+ cached compile lookup); cold compiles are
    counted once per (config, specialization).
    """
    miss = sample["cache_miss"].astype("boolean").fillna(False).astype(bool)
    compile_s = sample["compile_time_s"].fillna(0.0)
    cold_compile_s = compile_s.where(miss, 0.0)
    per_compile = float(compile_s[miss].mean()) if miss.any() else 0.0
    per_task = float((sample["task_wall_s"] - cold_compile_s).mean())
    total = per_compile * n_cold_compiles + max(per_task, 0.0) * n_tasks
    return {"per_cold_compile_s": per_compile, "per_task_s": per_task,
            "n_tasks": n_tasks, "n_cold_compiles": n_cold_compiles, "estimate_s": total}


def _tee(log: Callable[[str], None], path: Path) -> Callable[[str], None]:
    """Log to *log* and append timestamped lines to *path* (readable on Drive while a
    notebook runs headless under ``run_all``)."""

    def out(msg: str) -> None:
        log(msg)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}" + "\n")
        except OSError:
            pass

    return out


def save_arch(gpu_dir: str | os.PathLike[str], log: Callable[[str], None] = print) -> None:
    """Write the live architecture parameters to ``<gpu_dir>/arch.json`` (used by analysis)."""
    try:
        from archtune import arch

        a = arch.live()
        Path(gpu_dir).mkdir(parents=True, exist_ok=True)
        (Path(gpu_dir) / "arch.json").write_text(json.dumps(a.to_dict(), indent=2, default=str))
        if any(v != v for v in (a.peak_fp16_tflops, a.mem_bw_gbs)):
            log("[sweep] WARNING: peak throughput unknown for this GPU; predictor scores will be NaN")
    except Exception as exc:  # noqa: BLE001 - never block a sweep on this
        log(f"[sweep] could not save arch.json: {exc}")


def log_session(gpu_dir: str | os.PathLike[str], record: dict[str, Any]) -> None:
    """Append one JSON record to ``sessions.jsonl``."""
    record = {"logged_at": datetime.now(timezone.utc).isoformat(), **record}
    with open(Path(gpu_dir) / "sessions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


# --------------------------------------------------------------------------- GPU part


class _KernelAPI:
    """Uniform access to a kernel module: inputs, args, grid, reference."""

    def __init__(self, name: str) -> None:
        import triton

        self.name = name
        if name == "matmul":
            from archtune.kernels import matmul as mod

            base = mod.matmul_kernel
        elif name == "rmsnorm":
            from archtune.kernels import rmsnorm as mod

            base = mod.rmsnorm_kernel
        else:
            raise ValueError(f"unknown kernel {name!r}")
        # A fresh JITFunction per sweep session: its in-memory cache is empty, so the first
        # compile of every config in this session is cold (with the fresh TRITON_CACHE_DIR),
        # even if an earlier phase in the same process compiled the same config.
        self.jit = triton.jit(base.fn)
        self.mod = mod

    def inputs(self, shape: Shape, seed: int) -> tuple:
        if self.name == "matmul":
            return self.mod.make_inputs(shape.M, shape.N, shape.K, seed=seed)
        return self.mod.make_inputs(shape.M, shape.N, seed=seed)

    def ref(self, inputs: tuple) -> Any:
        return self.mod.reference(inputs[0], inputs[1])

    def grid(self, shape: Shape, cfg: dict[str, Any]) -> tuple:
        return self.mod.grid(shape.M, shape.N, cfg)


def evaluate_one(api: _KernelAPI, task: Task, inputs: tuple, ref: Any, env: dict[str, Any],
                 run_id: str, artifacts_dir: str | os.PathLike[str] | None,
                 feature_cache: dict[str, dict[str, Any]], seed: int = 0,
                 n_repetitions: int = 1, bench_ms: tuple[int, int] | None = None) -> dict[str, Any]:
    """Compile, check and time one config; never raises (fail-soft, spec §8)."""
    import torch

    from archtune import bench, compile_info, ptx_features

    t_start = time.perf_counter()
    shape, cfg, h = task.shape, task.cfg, task.cfg_hash
    row: dict[str, Any] = {
        "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "seed": seed,
        **{k: env.get(k) for k in ("gpu_name", "gpu_tag", "cc_major", "cc_minor", "sm_count",
                                    "driver", "cuda", "torch_version", "triton_version", "stack_id")},
        "kernel": shape.kernel, "shape_id": shape.shape_id, "M": shape.M, "N": shape.N, "K": shape.K,
        "dtype": "float16", "cfg_hash": h, **{f"cfg_{k}": v for k, v in cfg.items()},
        "status": "ok", "error_msg": None,
    }
    meta, launch = split_launch(cfg)
    args = api.mod.kernel_args(*inputs)
    g = api.grid(shape, cfg)
    res = compile_info.compile_config(api.jit, args, g, meta, launch)
    row.update(compile_time_s=res.compile_time_s, load_time_s=res.load_time_s,
               cache_miss=res.cache_miss, compile_method=res.method, n_regs=res.n_regs,
               n_spills=res.n_spills, shared_bytes=res.shared_bytes)
    if h not in feature_cache and res.asm:
        sass = compile_info.disassemble(res.asm.get("cubin")) if res.asm.get("cubin") else None
        feats = ptx_features.row_features(res.asm.get("ptx"), sass)
        path = None
        if artifacts_dir is not None:
            path = compile_info.save_artifacts(res.asm, Path(artifacts_dir) / shape.kernel, h, sass=sass)
        feature_cache[h] = {**feats, "artifact_path": path}
    row.update(feature_cache.get(h, {}))
    if res.status != "ok":
        row.update(status=res.status, error_msg=res.error_msg, task_wall_s=time.perf_counter() - t_start)
        return row

    fn = lambda: api.jit[g](*args, **meta, **launch)  # noqa: E731
    try:
        inputs[-1].zero_()
        fn()
        torch.cuda.synchronize()
        ok, err = bench.check_correct(inputs[-1], ref)
        row.update(correct=ok, max_abs_err=err)
        if not ok:
            row["status"] = "incorrect"
            row["error_msg"] = (f"max_abs_err={err:.4g} exceeds rtol={bench.RTOL}, atol={bench.ATOL} "
                                f"vs fp32 reference")
    except Exception as exc:  # noqa: BLE001
        row.update(status="launch_error", error_msg=compile_info.truncate(f"{type(exc).__name__}: {exc}"),
                   correct=False, task_wall_s=time.perf_counter() - t_start)
        return row
    try:
        warmup, rep = bench_ms or (bench.SWEEP_WARMUP_MS, bench.SWEEP_REP_MS)
        if res.cache_miss or (res.compile_time_s or 0.0) > 0.5:
            warmup = max(warmup, bench.POST_IDLE_WARMUP_MS)  # GPU idled during the compile
        row.update(bench.time_config(fn, warmup=warmup, rep=rep, n_repetitions=n_repetitions))
        row.update(bench_warmup_ms=warmup, bench_rep_ms=rep)
    except Exception as exc:  # noqa: BLE001
        row.update(status="launch_error", error_msg=compile_info.truncate(f"{type(exc).__name__}: {exc}"))
    row["task_wall_s"] = time.perf_counter() - t_start
    return row


def run_sweep(tasks: Sequence[Task], results_dir: str | os.PathLike[str], env: dict[str, Any],
              run_id: str = DEFAULT_RUN_ID, table: str = "sweep", limit: int | None = None,
              seed: int = 0, n_repetitions: int = 1, save_artifacts: bool = True,
              flush_every: int = 25, progress_every: int = 25, estimate_s: float | None = None,
              bench_ms: tuple[int, int] | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run (or resume) a sweep over *tasks*; returns a session summary.

    Uses a fresh ``TRITON_CACHE_DIR`` per session so cold compile times are measured.
    ``limit`` caps the number of *pending* tasks evaluated (small-first runs). Pass the
    small-first ``estimate_s`` for the full run so estimate vs actual is logged together.
    """
    from archtune import compile_info

    gpu_tag = env["gpu_tag"]
    gpu_dir = Path(results_dir) / gpu_tag
    gpu_dir.mkdir(parents=True, exist_ok=True)
    log = _tee(log, gpu_dir / "progress.log")
    store = RowStore(gpu_dir / table, flush_every=flush_every)
    todo = pending(tasks, store.done_keys(), gpu_tag, run_id)
    if limit is not None:
        todo = todo[:limit]
    log(f"[sweep] {len(tasks)} tasks, {len(todo)} pending this session (run_id={run_id})")
    cache_dir = compile_info.fresh_cache_dir()
    save_arch(gpu_dir, log)
    artifacts_dir = gpu_dir / "artifacts" if save_artifacts else None
    feature_cache: dict[str, dict[str, Any]] = {}
    apis: dict[str, _KernelAPI] = {}
    cur_shape, inputs, ref = None, None, None
    t0 = time.perf_counter()
    n_done = 0
    try:
        for task in todo:
            kernel = task.shape.kernel
            api = apis.setdefault(kernel, _KernelAPI(kernel))
            if task.shape != cur_shape:
                inputs = api.inputs(task.shape, seed)
                ref = api.ref(inputs)
                cur_shape = task.shape
            try:
                row = evaluate_one(api, task, inputs, ref, env, run_id, artifacts_dir, feature_cache,
                                   seed=seed, n_repetitions=n_repetitions, bench_ms=bench_ms)
            except Exception as exc:  # noqa: BLE001 - last-resort fail-soft
                row = {"run_id": run_id, "gpu_tag": gpu_tag, "kernel": kernel,
                       "shape_id": task.shape.shape_id, "cfg_hash": task.cfg_hash,
                       "status": "launch_error",
                       "error_msg": compile_info.truncate(traceback.format_exc())}
                log(f"[sweep] unexpected error: {exc}")
            store.append(row)
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                el = time.perf_counter() - t0
                log(f"[sweep] {n_done}/{len(todo)} done, {el:.0f}s elapsed, "
                    f"~{el / n_done * (len(todo) - n_done):.0f}s left")
    finally:
        store.flush()
    elapsed = time.perf_counter() - t0
    summary = {"session": store.session, "run_id": run_id, "table": table, "gpu_tag": gpu_tag,
               "n_evaluated": n_done, "n_pending_before": len(todo), "elapsed_s": elapsed,
               "triton_cache_dir": cache_dir, "stack_id": env.get("stack_id"),
               "estimate_s": estimate_s}
    log_session(gpu_dir, summary)
    return summary
