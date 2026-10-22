"""Running the project: sweep, analysis, experiments, pipeline steps, checks, figures and reports.

One command line covers everything::

    python -m archtune.run pipeline --mini        # quick end-to-end run -> results_mini/
    python -m archtune.run pipeline               # full run -> results/
    python -m archtune.run pipeline --only phase2_sweep phase4_5_analysis
    python -m archtune.run preflight              # check every GPU code path first
    python -m archtune.run demo --unpruned        # time triton.autotune with and without ArchTune
    python -m archtune.run report                 # figures, docs/RESULTS.md and the README block
    python -m archtune.run notebooks --mini       # execute the notebooks headlessly

Every step reads what earlier steps saved under ``results/`` (``results_mini/`` for a quick run),
so any step can be re-run on its own, and the sweep and experiments resume where they stopped.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from archtune import compiler, config, gpu, model
from archtune.model import Shape

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# =========================================================================== sweep
#
# Rows are written to small parquet files as they are measured, so an interrupted run loses at most
# a few rows and continues where it stopped:
#
#     results/<gpu_tag>/<table>/part-*.parquet          written during the run
#     results/<gpu_tag>/<table>.parquet, .csv           merged by consolidate()
#     results/<gpu_tag>/artifacts/<kernel>/<cfg_hash>/  PTX, cubin and SASS per config
#
# A row is identified by (gpu_tag, kernel, shape_id, cfg_hash, run_id); run_id names the experiment,
# not the session, so a resumed session skips what earlier sessions measured.

KEY_COLUMNS = ("gpu_tag", "kernel", "shape_id", "cfg_hash", "run_id")
ENV_COLUMNS = ("gpu_name", "gpu_tag", "cc_major", "cc_minor", "sm_count", "driver", "cuda",
               "torch_version", "triton_version", "stack_id")


@dataclass(frozen=True)
class Task:
    """One (shape, config) evaluation."""

    shape: Shape
    cfg: dict[str, Any]

    @property
    def cfg_hash(self) -> str:
        return model.config_hash(self.cfg)


def task_key(gpu_tag: str, kernel: str, shape_id: str, cfg_hash: str, run_id: str) -> tuple[str, ...]:
    """Resume key of one row."""
    return (gpu_tag, kernel, shape_id, cfg_hash, run_id)


def make_tasks(shapes: Sequence[Shape], configs_for: Callable[[Shape], Iterable[dict[str, Any]]]) -> list[Task]:
    """Shape-major task list: all configs of a shape in a row, so inputs are reused."""
    return [Task(s, dict(c)) for s in shapes for c in configs_for(s)]


def pending(tasks: Sequence[Task], done: set[tuple[str, ...]], gpu_tag: str, run_id: str) -> list[Task]:
    """Tasks whose key is not in *done* (order kept, duplicates removed)."""
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

    Parquet cannot be appended to, so each flush writes a new part; :meth:`load` concatenates
    them and keeps the latest row per key.
    """

    def __init__(self, directory: str | os.PathLike[str], flush_every: int | None = None,
                 flush_seconds: float = 120.0, session: str | None = None) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_every = config.BENCH.flush_every if flush_every is None else flush_every
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
        """Write the buffered rows as a new part file."""
        if not self._buf:
            return
        path = self.dir / f"part-{self.session}-{self._n:05d}.parquet"
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(self._buf).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        self._n += 1
        self._buf = []
        self._last = time.monotonic()

    def load(self) -> pd.DataFrame:
        return load_parts(self.dir)

    def done_keys(self) -> set[tuple[str, ...]]:
        return keys_of(self.load())


def load_parts(directory: str | os.PathLike[str]) -> pd.DataFrame:
    """Concatenate the ``part-*.parquet`` files in *directory*, latest row per key."""
    parts = sorted(Path(directory).glob("part-*.parquet"))
    if not parts:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    keys = [c for c in KEY_COLUMNS if c in df.columns]
    if keys:
        df = df.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)
    return df


def keys_of(df: pd.DataFrame) -> set[tuple[str, ...]]:
    """Resume keys present in a frame."""
    if df.empty or not set(KEY_COLUMNS) <= set(df.columns):
        return set()
    return set(map(tuple, df[list(KEY_COLUMNS)].astype(str).itertuples(index=False, name=None)))


def consolidate(gpu_dir: str | os.PathLike[str], table: str = "sweep") -> pd.DataFrame:
    """Merge the parts of *table* into ``<table>.parquet`` and ``<table>.csv``."""
    gpu_dir = Path(gpu_dir)
    df = load_parts(gpu_dir / table)
    if not df.empty:
        df.to_parquet(gpu_dir / f"{table}.parquet", index=False)
        df.drop(columns=[c for c in df.columns if c == "error_msg"]).to_csv(gpu_dir / f"{table}.csv", index=False)
    return df


def estimate_seconds(sample: pd.DataFrame, n_tasks: int, n_cold_compiles: int) -> dict[str, float]:
    """Extrapolate the full sweep from a small sample: cold compiles counted once per config,
    correctness + benchmark once per task."""
    miss = sample["cache_miss"].astype("boolean").fillna(False).astype(bool)
    compile_s = sample["compile_time_s"].fillna(0.0)
    per_compile = float(compile_s[miss].mean()) if miss.any() else 0.0
    per_task = float((sample["task_wall_s"] - compile_s.where(miss, 0.0)).mean())
    total = per_compile * n_cold_compiles + max(per_task, 0.0) * n_tasks
    return {"per_cold_compile_s": per_compile, "per_task_s": per_task,
            "n_tasks": n_tasks, "n_cold_compiles": n_cold_compiles, "estimate_s": total}


def _tee(log: Callable[[str], None], path: Path) -> Callable[[str], None]:
    """Log to *log* and append timestamped lines to *path*, to follow a long run from elsewhere."""

    def out(msg: str) -> None:
        log(msg)
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}\n")
        except OSError:
            pass

    return out


def save_arch(gpu_dir: str | os.PathLike[str], log: Callable[[str], None] = print) -> None:
    """Write the live architecture parameters to ``<gpu_dir>/arch.json``."""
    try:
        a = gpu.live()
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


class KernelSession:
    """A registered kernel with its own fresh ``JITFunction``.

    Its in-memory cache starts empty, so together with a fresh ``TRITON_CACHE_DIR`` the first
    compile of every config in a session is cold, even if an earlier step compiled it.
    """

    def __init__(self, name: str) -> None:
        self.kernel = gpu.kernel(name)
        self.name = name
        self.jit = self.kernel.fresh_jit()


def evaluate_one(sess: KernelSession, task: Task, inputs: tuple, ref: Any, env: dict[str, Any],
                 run_id: str, artifacts_dir: str | os.PathLike[str] | None,
                 feature_cache: dict[str, dict[str, Any]], seed: int = 0,
                 n_repetitions: int = 1, bench_ms: tuple[int, int] | None = None) -> dict[str, Any]:
    """Compile, check and time one config. Never raises: a failure is a status in the row."""
    import torch

    b = config.BENCH
    t_start = time.perf_counter()
    shape, cfg, h = task.shape, task.cfg, task.cfg_hash
    k = sess.kernel
    row: dict[str, Any] = {
        "run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), "seed": seed,
        **{c: env.get(c) for c in ENV_COLUMNS},
        "kernel": shape.kernel, "shape_id": shape.shape_id, "M": shape.M, "N": shape.N, "K": shape.K,
        "dtype": "float16", "cfg_hash": h, **{f"cfg_{key}": v for key, v in cfg.items()},
        "status": "ok", "error_msg": None,
    }
    meta, launch = gpu.split_launch(cfg)
    args = k.kernel_args(*inputs)
    g = k.shape_grid(shape, cfg)
    res = gpu.compile_config(sess.jit, args, g, meta, launch)
    row.update(compile_time_s=res.compile_time_s, load_time_s=res.load_time_s,
               cache_miss=res.cache_miss, compile_method=res.method, n_regs=res.n_regs,
               n_spills=res.n_spills, shared_bytes=res.shared_bytes)
    if h not in feature_cache and res.asm:
        sass = gpu.disassemble(res.asm.get("cubin")) if res.asm.get("cubin") else None
        feats = compiler.row_features(res.asm.get("ptx"), sass)
        path = None
        if artifacts_dir is not None:
            path = gpu.save_artifacts(res.asm, Path(artifacts_dir) / shape.kernel, h, sass=sass)
        feature_cache[h] = {**feats, "artifact_path": path}
    row.update(feature_cache.get(h, {}))
    if res.status != "ok":
        row.update(status=res.status, error_msg=res.error_msg, task_wall_s=time.perf_counter() - t_start)
        return row

    fn = lambda: sess.jit[g](*args, **meta, **launch)  # noqa: E731
    try:
        inputs[-1].zero_()
        fn()
        torch.cuda.synchronize()
        ok, err = gpu.check_correct(inputs[-1], ref)
        row.update(correct=ok, max_abs_err=err)
        if not ok:
            row["status"] = "incorrect"
            row["error_msg"] = f"max_abs_err={err:.4g} exceeds rtol={b.rtol}, atol={b.atol} vs fp32 reference"
    except Exception as exc:  # noqa: BLE001
        row.update(status="launch_error", error_msg=gpu.truncate(f"{type(exc).__name__}: {exc}"),
                   correct=False, task_wall_s=time.perf_counter() - t_start)
        return row
    try:
        warmup, rep = bench_ms or (b.sweep_warmup_ms, b.sweep_rep_ms)
        if res.cache_miss or (res.compile_time_s or 0.0) > b.slow_compile_s:
            warmup = max(warmup, b.post_idle_warmup_ms)  # the GPU idled during the compile
        row.update(gpu.time_config(fn, warmup=warmup, rep=rep, n_repetitions=n_repetitions))
        row.update(bench_warmup_ms=warmup, bench_rep_ms=rep)
    except Exception as exc:  # noqa: BLE001
        row.update(status="launch_error", error_msg=gpu.truncate(f"{type(exc).__name__}: {exc}"))
    row["task_wall_s"] = time.perf_counter() - t_start
    return row


def run_sweep(tasks: Sequence[Task], results_dir: str | os.PathLike[str], env: dict[str, Any],
              run_id: str | None = None, table: str = "sweep", limit: int | None = None,
              seed: int = 0, n_repetitions: int = 1, save_artifacts: bool = True,
              flush_every: int | None = None, progress_every: int | None = None,
              estimate_s: float | None = None, bench_ms: tuple[int, int] | None = None,
              log: Callable[[str], None] = print) -> dict[str, Any]:
    """Run (or resume) a sweep over *tasks* and return a session summary.

    Each session gets a fresh ``TRITON_CACHE_DIR``, so compile times are cold. ``limit`` caps how
    many pending tasks run (for small-first runs); ``estimate_s`` is logged next to the actual time.
    """
    run_id = run_id or config.EXPERIMENTS.sweep_run_id
    progress_every = config.BENCH.progress_every if progress_every is None else progress_every
    gpu_tag = env["gpu_tag"]
    gpu_dir = Path(results_dir) / gpu_tag
    gpu_dir.mkdir(parents=True, exist_ok=True)
    log = _tee(log, gpu_dir / "progress.log")
    store = RowStore(gpu_dir / table, flush_every=flush_every)
    todo = pending(tasks, store.done_keys(), gpu_tag, run_id)
    if limit is not None:
        todo = todo[:limit]
    log(f"[sweep] {len(tasks)} tasks, {len(todo)} pending this session (run_id={run_id})")
    cache_dir = gpu.fresh_cache_dir()
    save_arch(gpu_dir, log)
    artifacts_dir = gpu_dir / "artifacts" if save_artifacts else None
    feature_cache: dict[str, dict[str, Any]] = {}
    sessions: dict[str, KernelSession] = {}
    cur_shape, inputs, ref = None, None, None
    t0 = time.perf_counter()
    n_done = 0
    try:
        for task in todo:
            name = task.shape.kernel
            sess = sessions.get(name) or sessions.setdefault(name, KernelSession(name))
            if task.shape != cur_shape:
                inputs = sess.kernel.inputs(task.shape, seed)
                ref = sess.kernel.reference_of(inputs)
                cur_shape = task.shape
            try:
                row = evaluate_one(sess, task, inputs, ref, env, run_id, artifacts_dir, feature_cache,
                                   seed=seed, n_repetitions=n_repetitions, bench_ms=bench_ms)
            except Exception as exc:  # noqa: BLE001 - one bad config must not stop the sweep
                row = {"run_id": run_id, "gpu_tag": gpu_tag, "kernel": name,
                       "shape_id": task.shape.shape_id, "cfg_hash": task.cfg_hash,
                       "status": "launch_error", "error_msg": gpu.truncate(traceback.format_exc())}
                log(f"[sweep] unexpected error: {exc}")
            store.append(row)
            n_done += 1
            if progress_every and n_done % progress_every == 0:
                el = time.perf_counter() - t0
                log(f"[sweep] {n_done}/{len(todo)} done, {el:.0f}s elapsed, "
                    f"~{el / n_done * (len(todo) - n_done):.0f}s left")
    finally:
        store.flush()
    summary = {"session": store.session, "run_id": run_id, "table": table, "gpu_tag": gpu_tag,
               "n_evaluated": n_done, "n_pending_before": len(todo),
               "elapsed_s": time.perf_counter() - t0, "triton_cache_dir": cache_dir,
               "stack_id": env.get("stack_id"), "estimate_s": estimate_s}
    log_session(gpu_dir, summary)
    return summary


# =========================================================================== experiments
#
# maxnreg: cap the registers ptxas may use and watch spills, occupancy and runtime.
# num_stages: vary the software-pipelining depth and watch shared memory, cp.async and runtime.
# Both reuse the sweep with their own tables, so they resume the same way.


def with_maxnreg(configs: Sequence[dict[str, Any]], caps: Sequence[int | None] | None = None
                 ) -> list[dict[str, Any]]:
    """Every config at every register cap (``None`` = uncapped)."""
    caps = config.EXPERIMENTS.maxnreg_caps if caps is None else caps
    return [{**c, "maxnreg": cap} if cap is not None else dict(c) for c in configs for cap in caps]


def with_num_stages(configs: Sequence[dict[str, Any]], stages: Sequence[int] | None = None
                    ) -> list[dict[str, Any]]:
    """Every config at every pipelining depth (duplicates removed, order kept)."""
    stages = config.EXPERIMENTS.num_stages if stages is None else stages
    out, seen = [], set()
    for c in configs:
        base = {k: v for k, v in c.items() if k != "num_stages"}
        for s in stages:
            key = tuple(sorted({**base, "num_stages": s}.items()))
            if key not in seen:
                seen.add(key)
                out.append({**base, "num_stages": s})
    return out


def representative_configs(sweep: pd.DataFrame, shape_id: str) -> list[dict[str, Any]]:
    """Up to three contrasting configs on one shape for the register-cap study: the fastest, the
    non-spilling one with the most registers, and the fastest that spills (or the runner-up)."""
    d = sweep[sweep["shape_id"] == shape_id].copy()
    v = d[model.valid_mask(d)]
    picks: list[pd.Series] = []
    if not v.empty:
        picks.append(v.loc[v["time_ms_median"].idxmin()])
        nospill = v[v["n_spills"].fillna(0) == 0].sort_values(["n_regs", "time_ms_median"], ascending=[False, True])
        if not nospill.empty:
            picks.append(nospill.iloc[0])
        spill = v[v["n_spills"].fillna(0) > 0]
        if not spill.empty:
            picks.append(spill.loc[spill["time_ms_median"].idxmin()])
        elif len(nospill) > 1:
            picks.append(nospill.iloc[1])
    out, seen = [], set()
    for r in picks:
        if r["cfg_hash"] not in seen:
            seen.add(r["cfg_hash"])
            out.append(model.row_config(r))
    return out


def check_single_stack(df: pd.DataFrame) -> str:
    """The stack id of the rows; raises if they come from more than one software stack."""
    ids = df["stack_id"].dropna().unique() if "stack_id" in df else []
    if len(ids) != 1:
        raise ValueError(f"expected exactly one stack_id, found {list(ids)}; never compare across stacks")
    return str(ids[0])


def _label(cfg: dict[str, Any]) -> str:
    return (f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']} "
            f"w{cfg['num_warps']} s{cfg['num_stages']}")


def maxnreg_frame(df: pd.DataFrame, arch: gpu.ArchParams, uncapped_as: int = 256) -> pd.DataFrame:
    """Figure-3 frame from the register-cap rows of one shape. Resident blocks/SM come from the
    measured registers and shared memory, i.e. what ptxas actually allocated."""
    d = df.copy()
    if "cfg_maxnreg" not in d:
        d["cfg_maxnreg"] = np.nan
    d["cap"] = d["cfg_maxnreg"].fillna(uncapped_as).astype(int)
    base_cols = [c for c in d.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_maxnreg")]
    d["cfg_label"] = d.apply(lambda r: _label({c[4:]: int(r[c]) for c in base_cols}), axis=1)
    d["blocks_per_sm"] = [
        model.occupancy(int(r.n_regs), int(r.shared_bytes), int(r.cfg_num_warps), arch)["blocks_per_sm"]
        if pd.notna(r.n_regs) and pd.notna(r.shared_bytes) else np.nan
        for r in d.itertuples()]
    unc = d[d["cap"] == uncapped_as].set_index("cfg_label")["time_ms_median"]
    d["time_rel"] = d["time_ms_median"] / d["cfg_label"].map(unc)
    return d[["cfg_label", "cap", "n_regs", "n_spills", "shared_bytes", "blocks_per_sm",
              "time_ms_median", "time_rel", "status"]].sort_values(["cfg_label", "cap"])


def num_stages_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-4 frame: runtime relative to the best stage count of each (GPU, shape, base config)."""
    d = df.copy()
    base_cols = [c for c in d.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_num_stages")]
    d["base"] = d[base_cols].astype(str).agg("|".join, axis=1)
    d["num_stages"] = d["cfg_num_stages"].astype(int)
    best = d[model.valid_mask(d)].groupby(["gpu_tag", "shape_id", "base"])["time_ms_median"].min().rename("best")
    d = d.join(best, on=["gpu_tag", "shape_id", "base"])
    d["time_rel"] = d["time_ms_median"] / d["best"]
    return d[["gpu_tag", "shape_id", "base", "num_stages", "shared_bytes", "ptx_cp_async",
              "n_regs", "n_spills", "time_ms_median", "time_rel", "status"]]


# =========================================================================== analysis
#
# A finished sweep becomes predictor accuracy numbers, replay curves and the cost-to-target table
# under results/<gpu_tag>/analysis/. The predictor is fitted on the training shapes only and saved
# to results/<gpu_tag>/models.json; the held-out shapes are only used for evaluation.


def load_sweep(results_dir: str | Path, gpu_tag: str, run_id: str | None = None,
               table: str = "sweep") -> pd.DataFrame:
    """The consolidated rows of one GPU and run id (merging parts if they exist)."""
    run_id = run_id or config.EXPERIMENTS.sweep_run_id
    gpu_dir = Path(results_dir) / gpu_tag
    df = consolidate(gpu_dir, table) if (gpu_dir / table).exists() else pd.read_parquet(gpu_dir / f"{table}.parquet")
    return df[df["run_id"] == run_id].reset_index(drop=True)


def arch_for_tag(gpu_tag: str, sm_count: int | None = None,
                 results_dir: str | Path | None = None) -> gpu.ArchParams:
    """Architecture parameters for a GPU tag: ``arch.json`` from the sweep if there is one,
    otherwise the reference tables with the SM count seen in the rows."""
    if results_dir is not None and (Path(results_dir) / gpu_tag / "arch.json").exists():
        return gpu.ArchParams.from_dict(json.loads((Path(results_dir) / gpu_tag / "arch.json").read_text()))
    prefix, sm = gpu_tag.split("_sm")
    if prefix in config.GPU_REFERENCE:
        a = gpu.reference(prefix)
    else:
        a = gpu.from_cc((int(sm[0]), int(sm[1:])), sm_count or 1, name=gpu_tag)
    if sm_count and sm_count != a.sm_count:
        a = replace(a, sm_count=int(sm_count))
    return a


def shapes_of(df: pd.DataFrame) -> list[Shape]:
    """Distinct shapes in a sweep frame."""
    u = df[["kernel", "shape_id", "M", "N", "K"]].drop_duplicates("shape_id")
    return [Shape(r.kernel, r.shape_id, int(r.M), int(r.N), int(r.K), ()) for r in u.itertuples()]


def fit_models(sweep: pd.DataFrame, split: dict[str, str]) -> dict[str, Any]:
    """Fit the register and shared-memory models on matmul rows of the training shapes."""
    d = sweep[(sweep["kernel"] == "matmul") & (sweep["shape_id"].map(split) == "train")]
    per_cfg = model.configs_from_sweep(d)
    reg = model.fit_regs(per_cfg)
    smem, rate = model.fit_smem(per_cfg)
    return {"reg_model": model.model_to_dict(reg), "smem_model": model.model_to_dict(smem),
            "smem_train_exact_match": rate, "n_train_configs": int(len(per_cfg)),
            "train_shapes": sorted(k for k, v in split.items() if v == "train")}


def load_models(results_dir: str | Path, gpu_tag: str) -> tuple[model.RegModel, model.SmemModel] | None:
    """Fitted models from ``<results_dir>/<gpu_tag>/models.json`` (``None`` before fitting)."""
    p = Path(results_dir) / gpu_tag / "models.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    return model.RegModel(**m["reg_model"]), model.SmemModel(**m["smem_model"])


def predictor_table(sweep: pd.DataFrame, a: gpu.ArchParams, reg: model.RegModel,
                    smem: model.SmemModel) -> pd.DataFrame:
    """Predicted vs measured registers / spills / shared memory per matmul config."""
    per_cfg = model.configs_from_sweep(sweep[sweep["kernel"] == "matmul"])
    rows = []
    for _, r in per_cfg.iterrows():
        cfg = model.row_config(r)
        rows.append({
            "cfg_hash": r["cfg_hash"], **{k: v for k, v in r.items() if k.startswith("cfg_")},
            "n_regs": r.get("n_regs"), "n_spills": r.get("n_spills"), "shared_bytes": r.get("shared_bytes"),
            "regs_pred": model.regs_est(cfg, a, reg, cfg.get("maxnreg")),
            "regs_pred_analytic": model.regs_est(cfg, a, model.ANALYTIC_REGS, cfg.get("maxnreg")),
            "spill_pred": model.spill_pred(cfg, a, reg, cfg.get("maxnreg")),
            "spill_pred_analytic": model.spill_pred(cfg, a, model.ANALYTIC_REGS, cfg.get("maxnreg")),
            "smem_pred": model.smem_est(cfg, smem),
            "smem_pred_analytic": model.smem_est(cfg, model.ANALYTIC_SMEM),
        })
    return pd.DataFrame(rows)


def predictor_metrics(pm: pd.DataFrame) -> dict[str, Any]:
    """Accuracy of the fitted and analytic variants."""
    d = pm.dropna(subset=["n_regs"])
    nospill = d[d["n_spills"].fillna(0) == 0]
    out: dict[str, Any] = {}
    for variant, sfx in (("fitted", ""), ("analytic", "_analytic")):
        out[variant] = {
            "regs_nonspilling": model.regression_report(nospill[f"regs_pred{sfx}"], nospill["n_regs"]),
            "smem": model.regression_report(pm[f"smem_pred{sfx}"], pm["shared_bytes"]),
            "spill": model.spill_confusion(d[f"spill_pred{sfx}"], d["n_spills"].fillna(0) > 0),
        }
    return out


def top_mispredictions(pm: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The *n* configs with the largest register misprediction (fitted model)."""
    d = pm.dropna(subset=["n_regs"]).copy()
    d["regs_abs_err"] = (d["regs_pred"] - d["n_regs"]).abs()
    d["explanation"] = "unexplained"  # filled in by hand after comparing the PTX and SASS
    return d.sort_values("regs_abs_err", ascending=False).head(n)


def analyze(results_dir: str | Path, gpu_tag: str, run_id: str | None = None,
            n_seeds: int | None = None, refit: bool = False) -> dict[str, Any]:
    """Fit the predictor, score it and replay the search strategies for one GPU.

    Returns a summary, also written to ``analysis/summary.json``.
    """
    run_id = run_id or config.EXPERIMENTS.sweep_run_id
    n_seeds = config.SEARCH.n_seeds if n_seeds is None else n_seeds
    results_dir = Path(results_dir)
    out_dir = results_dir / gpu_tag / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(results_dir, gpu_tag, run_id)
    stack = check_single_stack(sweep)
    mm_ids = sorted(sweep.loc[sweep["kernel"] == "matmul", "shape_id"].unique())
    rms_ids = sorted(sweep.loc[sweep["kernel"] == "rmsnorm", "shape_id"].unique())
    split = {**model.train_test_split(mm_ids), **model.train_test_split(rms_ids)}

    models_path = results_dir / gpu_tag / "models.json"
    if refit or not models_path.exists():
        models_path.write_text(json.dumps({**fit_models(sweep, split), "stack_id": stack}, indent=2))
    reg, smem = load_models(results_dir, gpu_tag)

    sm_count = int(sweep["sm_count"].dropna().iloc[0]) if sweep["sm_count"].notna().any() else None
    a = arch_for_tag(gpu_tag, sm_count, results_dir)

    pm = predictor_table(sweep, a, reg, smem)
    pm.to_parquet(out_dir / "predictor.parquet", index=False)
    metrics = predictor_metrics(pm)
    top_mispredictions(pm).to_csv(out_dir / "top_mispredictions.csv", index=False)

    prepared = model.prepare(sweep)
    curves_all, targets_all = [], []
    for variant, (rm, smm) in {"smart": (reg, smem),
                               "smart_analytic": (model.ANALYTIC_REGS, model.ANALYTIC_SMEM)}.items():
        orders, ptime = {}, {}
        for shape in shapes_of(sweep):
            cfgs = [model.row_config(r) for _, r in
                    prepared[prepared["shape_id"] == shape.shape_id].drop_duplicates("cfg_hash").iterrows()]
            t0 = time.perf_counter()
            order = model.rank_smart(model.predict_frame(cfgs, [shape], a, rm, smm))
            key = (gpu_tag, shape.kernel, shape.shape_id)
            ptime[key] = time.perf_counter() - t0
            orders[key] = order
        c, t = model.replay(prepared, orders, split=split, predictor_time_s=ptime, n_seeds=n_seeds)
        if variant != "smart":
            c, t = c[c["strategy"] == "smart"], t[t["strategy"] == "smart"]
        curves_all.append(c.assign(strategy=c["strategy"].replace({"smart": variant})))
        targets_all.append(t.assign(strategy=t["strategy"].replace({"smart": variant})))
    curves = pd.concat(curves_all, ignore_index=True)
    targets = pd.concat(targets_all, ignore_index=True)
    curves.to_parquet(out_dir / "curves.parquet", index=False)
    targets.to_parquet(out_dir / "targets.parquet", index=False)
    table6 = model.summarize_targets(targets)
    table6.to_csv(out_dir / "table6.csv", index=False)

    status_counts = sweep.groupby(["kernel", "status"]).size().rename("n").reset_index()
    summary = {"gpu_tag": gpu_tag, "run_id": run_id, "stack_id": stack, "n_rows": int(len(sweep)),
               "status_counts": status_counts.to_dict("records"), "split": split,
               "predictor": metrics, "models": json.loads(models_path.read_text()),
               "cost_ratio_random_over_smart": model.cost_ratio(table6).to_dict("records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# =========================================================================== figures
#
# Colours follow a fixed order checked for colour-vision deficiency, and a series keeps its colour
# across figures (random search blue, ArchTune orange, exhaustive green). Different measures get
# their own panels rather than a second y-axis.


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def gpu_color(tag: str) -> str:
    """A stable colour per GPU tag."""
    series, order = config.PALETTE["series"], config.PALETTE["gpu_order"]
    return series[order.index(tag)] if tag in order else series[3 + hash(tag) % 4]


def style() -> Any:
    """Apply the shared matplotlib style; returns ``pyplot``."""
    plt = _plt()
    p = config.PALETTE
    plt.rcParams.update({
        "figure.facecolor": p["surface"], "axes.facecolor": p["surface"], "savefig.facecolor": p["surface"],
        "axes.edgecolor": p["grid"], "axes.labelcolor": p["ink2"], "axes.titlecolor": p["ink"],
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": p["ink2"], "ytick.color": p["ink2"], "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": p["grid"], "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
        "legend.fontsize": 8, "legend.labelcolor": p["ink"], "font.size": 9, "text.color": p["ink"],
    })
    return plt


def _save(fig: Any, path: str | Path | None) -> Any:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def fig1_quality_vs_cost(curves: pd.DataFrame, gpu_tag: str, path: str | Path | None = None,
                         target: float | None = None, kernel: str = "matmul") -> Any:
    """Quality vs tuning cost, one panel per split.

    Each budget k is a separate choice, so strategies are markers (one per k) joined by a thin
    dotted line that only shows the order of increasing k. Random-k: mean over shapes x seeds with
    an IQR bar; smart-k (fitted: filled squares, analytic: hollow triangles): median over shapes;
    exhaustive: one point, since it always tries every config. x = median cost (s), log scale.
    """
    target = config.SEARCH.target if target is None else target
    plt = style()
    pal = config.PALETTE
    colors = pal["strategy"]
    d = curves[(curves["gpu_tag"] == gpu_tag) & (curves["kernel"] == kernel)]
    splits = [s for s in ("train", "test", "all") if s in set(d["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.4 * len(splits), 4.0), sharey=True, squeeze=False)
    connector = dict(linestyle=":", linewidth=1.0, alpha=0.6)
    for ax, split in zip(axes[0], splits):
        ds = d[d["split"] == split]
        e = ds[ds["strategy"] == "exhaustive"]
        n_all = int(e["n_configs"].max()) if not e.empty else None
        r = ds[ds["strategy"] == "random"]
        if n_all is not None:  # random with k = all is exhaustive, shown once as the diamond
            r = r[r["n_configs"] < n_all]
        if not r.empty:
            g = r.groupby("n_configs")
            agg = pd.DataFrame({"cost": g["cost_s"].median(), "mean": g["quality"].mean(),
                                "q25": g["quality"].quantile(0.25), "q75": g["quality"].quantile(0.75)})
            ax.plot(agg["cost"], agg["mean"], color=colors["random"], **connector)
            ax.errorbar(agg["cost"], agg["mean"],
                        yerr=[(agg["mean"] - agg["q25"]).clip(lower=0), (agg["q75"] - agg["mean"]).clip(lower=0)],
                        fmt="o", color=colors["random"], markersize=6.5, elinewidth=1.2, capsize=0,
                        markeredgecolor=pal["surface"], markeredgewidth=1.2,
                        label="random-k (mean; bar = IQR over seeds × shapes)")
        for strat, marker, filled, label in (
                ("smart", "s", True, "smart-k, fitted (median over shapes)"),
                ("smart_analytic", "^", False, "smart-k, analytic (median over shapes)")):
            s = ds[ds["strategy"] == strat]
            if s.empty:
                continue
            g = s.groupby("n_configs")
            x, y = g["cost_s"].median(), g["quality"].median()
            col = colors[strat]
            ax.plot(x, y, color=col, **connector)
            ax.plot(x, y, marker, linestyle="none", color=col, markersize=6.5, label=label,
                    markerfacecolor=col if filled else pal["surface"],
                    markeredgecolor=pal["surface"] if filled else col, markeredgewidth=1.2 if filled else 1.5)
            if strat == "smart":
                for k, xi, yi in zip(x.index, x.to_numpy(), y.to_numpy()):
                    ax.annotate(f"{k}", (xi, yi), xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color=pal["ink2"])
        if not e.empty:
            ax.plot([e["cost_s"].median()], [1.0], "D", color=colors["exhaustive"], linestyle="none",
                    markersize=9, label=f"exhaustive (all {n_all} configs)", zorder=5,
                    markeredgecolor=pal["surface"], markeredgewidth=1.5)
        ax.axhline(target, color=pal["ink2"], linestyle="--", linewidth=1, zorder=0,
                   label=f"{target:.0%} quality target")
        ax.set_ylim(top=1.09)
        ax.set_xscale("log")
        ax.set_xlabel("tuning cost: compile + benchmark seconds (log)")
        ax.set_title(f"{gpu_tag} {kernel} — {split} shapes (n={ds['shape_id'].nunique()})", pad=10)
    axes[0][0].set_ylabel("quality = best / best found")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    prefixes = ["random", "smart-k, fitted", "smart-k, analytic", "exhaustive", f"{target:.0%}"]
    order = sorted(range(len(labels)),
                   key=lambda i: next(j for j, p in enumerate(prefixes) if labels[i].startswith(p)))
    axes[0][-1].legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right",
                       title="numbers above squares = k (configs tried)", title_fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def fig2_predictor(pm: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """Predicted vs measured registers and shared memory, and the spill confusion matrix."""
    plt = style()
    import matplotlib

    pal = config.PALETTE
    blue = pal["series"][0]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    ax.scatter(pm["n_regs"], pm["regs_pred"], s=14, color=blue, alpha=0.6, linewidths=0)
    lim = [0, 260]
    ax.plot(lim, lim, color=pal["ink2"], linewidth=1, linestyle="--")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("measured registers / thread (ptxas)")
    ax.set_ylabel("predicted registers / thread")
    ax.set_title("Registers")
    ax = axes[1]
    kb = 1024
    ax.scatter(pm["shared_bytes"] / kb, pm["smem_pred"] / kb, s=14, color=blue, alpha=0.6, linewidths=0)
    m = float(np.nanmax([pm["shared_bytes"].max(), pm["smem_pred"].max()])) / kb * 1.05
    ax.plot([0, m], [0, m], color=pal["ink2"], linewidth=1, linestyle="--")
    ax.set_xlabel("measured shared memory (KB)")
    ax.set_ylabel("predicted shared memory (KB)")
    ax.set_title("Shared memory")
    ax = axes[2]
    meas = pm["n_spills"].fillna(0) > 0
    pred = pm["spill_pred"].astype(bool)
    cm = np.array([[int((~pred & ~meas).sum()), int((pred & ~meas).sum())],
                   [int((~pred & meas).sum()), int((pred & meas).sum())]])
    ax.imshow(cm, cmap=matplotlib.colors.ListedColormap(list(pal["sequential"])), vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                color=pal["surface"] if v > cm.max() / 2 else pal["ink"])
    ax.set_xticks([0, 1], ["no spill", "spill"]), ax.set_yticks([0, 1], ["no spill", "spill"])
    ax.set_xlabel("predicted"), ax.set_ylabel("measured (ptxas n_spills > 0)")
    ax.grid(False)
    ax.set_title("Spill prediction")
    fig.suptitle(title, color=pal["ink"], fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig3_maxnreg(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """Register cap vs spills, resident blocks/SM and runtime, one line per config."""
    plt = style()
    pal = config.PALETTE
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    panels = [("n_spills", "spilled registers (ptxas)"), ("blocks_per_sm", "resident blocks / SM"),
              ("time_rel", "runtime / uncapped runtime")]
    labels = list(dict.fromkeys(df["cfg_label"]))
    for ax, (col, ylab) in zip(axes, panels):
        for i, lab in enumerate(labels):
            d = df[df["cfg_label"] == lab].sort_values("cap")
            ax.plot(d["cap"], d[col], "-o", color=pal["series"][i % 3], label=lab, markersize=5)
        ax.set_ylabel(ylab)
        ax.set_xlabel("maxnreg cap (256 = uncapped)")
    axes[2].axhline(1.0, color=pal["ink2"], linewidth=1, linestyle="--")
    axes[0].legend(loc="upper right")
    fig.suptitle(title, color=pal["ink"], fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig4_num_stages(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """num_stages vs shared memory, static cp.async count and runtime, per GPU."""
    plt = style()
    pal = config.PALETTE
    order = pal["gpu_order"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    panels = [("shared_bytes", "shared memory (KB)", 1 / 1024), ("ptx_cp_async", "cp.async in PTX (static)", 1),
              ("time_rel", "runtime / best runtime", 1)]
    tags = [t for t in order if t in set(df["gpu_tag"])] + sorted(set(df["gpu_tag"]) - set(order))
    for ax, (col, ylab, scale) in zip(axes, panels):
        for tag in tags:
            d = df[df["gpu_tag"] == tag].groupby("num_stages")[col].median().reset_index()
            ax.plot(d["num_stages"], d[col] * scale, "-o", color=gpu_color(tag), label=tag, markersize=5)
        ax.set_ylabel(ylab)
        ax.set_xlabel("num_stages")
    axes[0].legend(loc="upper left")
    fig.suptitle(title, color=pal["ink"], fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_liveness(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """PTX peak-live estimate vs ptxas registers: non-spilling kernels against the identity line,
    and all kernels, showing how far the PTX peak goes past the cap when ptxas has to spill."""
    plt = style()
    pal = config.PALETTE
    blue, orange = pal["series"][0], pal["series"][1]
    d = df.dropna(subset=["n_regs"])
    spill = d["n_spills"].fillna(0) > 0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ns = d[~spill]
    ax.scatter(ns["n_regs"], ns["peak_live_units"], s=14, color=blue, linewidths=0, alpha=0.7)
    hi = float(max(ns["n_regs"].max(), ns["peak_live_units"].max())) * 1.05
    ax.plot([0, hi], [0, hi], color=pal["ink2"], linewidth=1, linestyle="--")
    r = float(np.corrcoef(ns["n_regs"], ns["peak_live_units"])[0, 1]) if len(ns) > 1 else float("nan")
    ax.set_title(f"Non-spilling kernels (n={len(ns)}, r={r:.3f})")
    ax.set_xlabel("ptxas registers / thread")
    ax.set_ylabel("PTX peak live (32-bit units)")
    ax = axes[1]
    ax.scatter(d.loc[~spill, "n_regs"], d.loc[~spill, "peak_live_units"], s=12, color=blue, label="no spill",
               linewidths=0, alpha=0.6)
    ax.scatter(d.loc[spill, "n_regs"], d.loc[spill, "peak_live_units"], s=12, color=orange, label="spills",
               linewidths=0, alpha=0.7)
    ax.axvline(255, color=pal["ink2"], linewidth=1, linestyle=":")
    ax.text(253, ax.get_ylim()[1] * 0.97, "255-register cap ", color=pal["ink2"], fontsize=8, va="top", ha="right")
    ax.set_title("All kernels: spilling kernels pile up at the cap")
    ax.set_xlabel("ptxas registers / thread")
    ax.legend(loc="upper left")
    fig.suptitle(title, color=pal["ink"], fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_tuning_time(integ: pd.DataFrame, shape_id: str, path: str | Path | None = None) -> Any:
    """Live ``triton.autotune`` wall time, unpruned vs ArchTune top-k (log scale), with the
    chosen kernel's runtime next to each bar."""
    plt = style()
    pal = config.PALETTE
    d = integ[integ["shape_id"] == shape_id].copy()
    order = [m for m in ("unpruned", "archtune_top16", "archtune_top8") if m in set(d["mode"])]
    d = d.set_index("mode").loc[order]
    labels = {"unpruned": "Unpruned triton.autotune\n(every config)",
              "archtune_top16": "ArchTune, top-16", "archtune_top8": "ArchTune, top-8"}
    colors = [pal["strategy"]["exhaustive"] if m == "unpruned" else pal["strategy"]["smart"] for m in order]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    y = np.arange(len(order))
    ax.barh(y, d["tuning_s"], color=colors, height=0.56, edgecolor=pal["surface"], linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(y, [labels[m] for m in order])
    ax.invert_yaxis()
    for i, m in enumerate(order):
        r = d.loc[m]
        ax.text(r["tuning_s"] * 1.12, i, f"{r['tuning_s']:,.1f} s   (kernel {r['time_ms_median']:.2f} ms)",
                va="center", fontsize=9, color=pal["ink"])
    un = d.loc["unpruned", "tuning_s"] if "unpruned" in d.index else None
    k8 = d.loc["archtune_top8", "tuning_s"] if "archtune_top8" in d.index else None
    title = f"Live autotuning time, {shape_id.replace('mm_', '').replace('_', ' ')}"
    if un and k8:
        title += f": {un / k8:.0f}x less with ArchTune"
    ax.set_title(title, loc="left")
    ax.set_xlabel("wall-clock tuning time, seconds (log scale)")
    ax.set_xlim(right=d["tuning_s"].max() * 12)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def fig_compile_breakdown(sweep: pd.DataFrame, path: str | Path | None = None) -> Any:
    """Where the exhaustive search's compile time goes: useful configs, configs that spill, and
    configs that exceed shared memory (each config's cold compile counted once)."""
    plt = style()
    pal = config.PALETTE
    d = sweep[sweep["kernel"] == "matmul"]
    cold = d[d["cache_miss"].astype("boolean").fillna(False).astype(bool)].drop_duplicates("cfg_hash")
    spill = cold["n_spills"].fillna(0) > 0
    oor = cold["status"] == "out_of_resources"
    cats = [("Exceeds shared memory\n(fails at load)", oor, pal["series"][3]),
            ("Spills registers\n(slow kernel)", spill & ~oor, pal["series"][4]),
            ("Useful configs", ~spill & ~oor, pal["series"][0])]
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    total = cold["compile_time_s"].sum() / 60
    for i, (lab, mask, col) in enumerate(cats):
        minutes = cold.loc[mask, "compile_time_s"].sum() / 60
        ax.barh(i, minutes, color=col, height=0.56, edgecolor=pal["surface"], linewidth=2)
        ax.text(minutes + total * 0.01, i, f"{minutes:.1f} min  ·  {int(mask.sum())} configs  ·  "
                f"{minutes / total:.0%} of compile time", va="center", fontsize=9, color=pal["ink"])
    ax.set_yticks(range(len(cats)), [c[0] for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.25)
    ax.set_xlabel("minutes of compilation (each matmul config compiled once)")
    ax.set_title(f"Where exhaustive autotuning's compile time goes ({total:.1f} min, {len(cold)} configs)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)


# =========================================================================== pipeline steps


@dataclass
class Run:
    """Where and how a run writes."""

    project: Path
    results: Path
    env: dict[str, Any]
    mini: bool

    @property
    def gpu_dir(self) -> Path:
        return self.results / self.env["gpu_tag"]

    @property
    def figures(self) -> Path:
        return self.results / config.PATHS.figures


def setup(project: str | Path, mini: bool) -> Run:
    """Create the run context (``results_mini/`` for a quick run)."""
    project = Path(project)
    e = gpu.capture()
    if e["gpu_tag"] is None:
        raise RuntimeError("No GPU visible: Runtime > Change runtime type > T4 GPU")
    run = Run(project, project / (config.PATHS.results_mini if mini else config.PATHS.results), e, mini)
    run.gpu_dir.mkdir(parents=True, exist_ok=True)
    run.figures.mkdir(parents=True, exist_ok=True)
    print(f"{'MINI' if mini else 'FULL'} run on {e['gpu_name']} ({e['gpu_tag']}), "
          f"torch {e['torch_version']}, triton {e['triton_version']} -> {run.results}")
    return run


def shapes(run: Run) -> tuple[list[Shape], list[Shape]]:
    """Matmul and RMSNorm shapes (the subset for a quick run)."""
    dims = model.derive_dims(model.load_model_config(cache_dir=run.results))
    mm, rms = model.matmul_shapes(dims), model.rmsnorm_shapes(dims)
    if run.mini:
        sub = model.mini_shapes(mm + rms)
        mm = [s for s in sub if s.kernel == "matmul"]
        rms = [s for s in sub if s.kernel == "rmsnorm"]
    return mm, rms


def matmul_configs(run: Run) -> list[dict[str, Any]]:
    """384 configs (12 for a quick run)."""
    return model.mini_matmul_space() if run.mini else model.matmul_space()


def _large_shape(run: Run) -> Shape:
    return next(s for s in shapes(run)[0] if s.shape_id == config.EXPERIMENTS.large_shape_id)


def phase0_smoke(run: Run) -> dict[str, Any]:
    """Architecture check, API check, one compile with inspection, SASS and the noise check."""
    import torch

    mm = gpu.MATMUL
    b = config.BENCH
    a = gpu.live()
    key = gpu.reference_for_name(a.name)
    diffs = gpu.verify(gpu.reference(key), a) if key else "no reference entry (peaks estimated from clocks)"
    print("arch:", a.name, a.cc, f"SMs={a.sm_count}", f"smem/block={a.smem_per_block_optin}",
          f"peak={a.peak_fp16_tflops:.0f} TFLOP/s", f"bw={a.mem_bw_gbs:.0f} GB/s")
    print("live vs reference:", diffs)
    (run.gpu_dir / "arch.json").write_text(json.dumps(a.to_dict(), indent=2, default=str))
    gpu.pip_freeze(run.gpu_dir / "pip_freeze.txt")
    (run.gpu_dir / "device_properties.json").write_text(json.dumps(gpu.device_properties(), indent=2, default=str))

    api = gpu.api_report()
    (run.gpu_dir / "api_check.json").write_text(json.dumps(api, indent=2, default=str))
    for k in ("JITFunction.warmup", "maxnreg_in_Config", "Autotuner.prune_configs", "do_bench",
              "do_bench_flushes_l2", "cache_dir_env_read_at_init"):
        print(f"  {k:28s} {api.get(k)}")

    gpu.fresh_cache_dir()
    cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    x, y, z = mm.make_inputs(1024, 1024, 1024)
    res = gpu.compile_kernel(mm, (x, y, z), cfg)
    if res.status != "ok":
        raise RuntimeError(f"smoke compile failed: {res.status} {res.error_msg}")
    attrs = gpu.compiled_kernel_attrs(res.compiled)
    (run.gpu_dir / "compiled_kernel_attrs.json").write_text(json.dumps(
        {**attrs, "describe": gpu.describe_compiled(res.compiled)}, indent=2, default=str))
    mm.launcher(x, y, z, cfg)()
    torch.cuda.synchronize()
    ok, err = gpu.check_correct(z, mm.reference(x, y))
    print(f"smoke config: compile {res.compile_time_s:.2f}s, regs={res.n_regs}, spills={res.n_spills}, "
          f"smem={res.shared_bytes}, correct={ok} (max err {err:.3g}); asm keys {attrs['asm_keys']}")
    if not ok or res.n_regs is None or res.shared_bytes is None:
        raise RuntimeError("smoke test did not capture registers/shared memory or was incorrect")
    print("SASS:", "available" if gpu.disassemble(res.asm.get("cubin")) else "unavailable (PTX-only analysis)")

    x, y, z = mm.make_inputs(4096, 4096, 4096)
    big = {**cfg, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}
    noise = gpu.noise_check(mm.launcher(x, y, z, big), n=5 if run.mini else b.noise_runs,
                            warmup=b.sweep_warmup_ms, rep=b.sweep_rep_ms)
    noise.update(shape="4096^3", cfg=big, bench_ms=[b.sweep_warmup_ms, b.sweep_rep_ms],
                 gpu_tag=run.env["gpu_tag"], stack_id=run.env["stack_id"])
    (run.gpu_dir / "noise.json").write_text(json.dumps(noise, indent=2))
    print(f"noise: CV={noise['cv']:.4f} -> differences < {noise['tie_threshold']:.2%} are ties")
    return {"api": api, "noise_cv": noise["cv"], "smoke_correct": ok}


def phase2_sweep(run: Run, run_id: str | None = None) -> pd.DataFrame:
    """Time ten configs to estimate the run, then sweep everything; picks up where it stopped."""
    run_id = run_id or config.EXPERIMENTS.sweep_run_id
    mm, rms = shapes(run)
    cfgs = matmul_configs(run)
    n_rms = config.WORKLOAD.mini_rmsnorm_configs if run.mini else None
    tasks = make_tasks(mm, lambda s: cfgs) + make_tasks(rms, lambda s: model.rmsnorm_space(s.N)[:n_rms])
    print(f"{len(tasks)} tasks ({len(mm)} matmul shapes x {len(cfgs)} configs + RMSNorm)")

    done = load_parts(run.gpu_dir / "sweep")
    if done.empty:  # small first, only on a brand-new sweep
        small = [t for t in tasks if t.shape.shape_id == mm[-1].shape_id][:config.EXPERIMENTS.estimate_sample]
        run_sweep(small, run.results, run.env, run_id=run_id)
        done = load_parts(run.gpu_dir / "sweep")
        sample = done[done.cfg_hash.isin([t.cfg_hash for t in small])]
        bad = sample[sample.status.isin(["compile_error", "launch_error", "incorrect"])]
        if not run.mini and not bad.empty:
            raise RuntimeError(f"small-first: {len(bad)} failing configs, fix before the full sweep: "
                               f"{bad.error_msg.head(3).tolist()}")
        remaining = pending(tasks, keys_of(done), run.env["gpu_tag"], run_id)
        est = estimate_seconds(sample[sample.status == "ok"], len(remaining),
                               len({(t.shape.kernel, t.cfg_hash) for t in remaining}))
        print(f"ESTIMATE for the rest: {est['estimate_s'] / 60:.0f} min "
              f"({est['per_cold_compile_s']:.1f}s per compile, {est['per_task_s']:.2f}s per task)")
        estimate = est["estimate_s"]
    else:
        print(f"resuming: {len(done)} rows already saved")
        estimate = None
    run_sweep(tasks, run.results, run.env, run_id=run_id, estimate_s=estimate)
    df = consolidate(run.gpu_dir)
    df = df[df.run_id == run_id]
    missing = pending(tasks, keys_of(df), run.env["gpu_tag"], run_id)
    counts = df.groupby(["kernel", "status"]).size().unstack(fill_value=0)
    print(counts)
    broken = df.status.isin(["compile_error", "launch_error"]).mean()
    if missing or broken >= 0.02:
        raise RuntimeError(f"sweep unhealthy: {len(missing)} missing rows, {broken:.1%} compile/launch errors: "
                           f"{df.loc[df.status.isin(['compile_error', 'launch_error']), 'error_msg'].head(3).tolist()}")
    return counts


def phase3_ir(run: Run) -> pd.DataFrame:
    """Dump the IR of the fastest and slowest valid config on the large shape; feature table."""
    mm = gpu.MATMUL
    large = config.EXPERIMENTS.large_shape_id
    sw = model.prepare(load_sweep(run.results, run.env["gpu_tag"]))
    t = sw[(sw.shape_id == large) & sw.valid].sort_values("time_ms_median")
    shape = _large_shape(run)
    inputs = mm.make_inputs(shape.M, shape.N, shape.K)
    listings = {}
    for name, row in {"fast": t.iloc[0], "slow": t.iloc[-1]}.items():
        cfg = model.row_config(row)
        r = gpu.compile_kernel(mm, inputs, cfg)
        sass = gpu.disassemble(r.asm.get("cubin")) if r.asm.get("cubin") else None
        compiler.dump_ir(r.asm, run.gpu_dir / "ir_exploration", name, sass)
        listings[name] = {"ptx": r.asm.get("ptx"), "sass": sass}
        print(f"{name}: {row.time_ms_median:.3f} ms, cfg {cfg}, regs {r.n_regs}, spills {r.n_spills}, smem {r.shared_bytes}")
        for pat in (r"mma\.sync", r"ldmatrix", r"cp\.async\.c[ag]", r"(ld|st)\.local"):
            print(f"  {pat}: {compiler.snippets(r.asm.get('ptx', ''), pat, context=0, max_hits=1)}")
    feats = compiler.compare_features(listings)
    feats.to_csv(run.gpu_dir / "ir_exploration" / "features.csv")
    return feats


def phase4_5_analysis(run: Run) -> dict[str, Any]:
    """Fit the predictor on the training shapes, compare it with ptxas and replay the strategies."""
    tag = run.env["gpu_tag"]
    summary = analyze(run.results, tag, refit=True, n_seeds=5 if run.mini else None)
    a_dir = run.gpu_dir / "analysis"
    fig2_predictor(pd.read_parquet(a_dir / "predictor.parquet"),
                   f"Predicted vs measured — {tag} (fitted on train split)", run.figures / f"fig2_predictor_{tag}.png")
    fig1_quality_vs_cost(pd.read_parquet(a_dir / "curves.parquet"), tag, run.figures / f"fig1_quality_vs_cost_{tag}.png")
    for variant, m in summary["predictor"].items():
        print(f"{variant:9s} regs median|err|={m['regs_nonspilling']['median_abs_err']:.3g}  "
              f"smem exact={m['smem']['exact_match']:.2f}  spill P/R={m['spill']['precision']:.2f}/{m['spill']['recall']:.2f}")
    t6 = pd.read_csv(a_dir / "table6.csv")
    print(t6.to_string(index=False))
    print(model.cost_ratio(t6).to_string(index=False))
    return summary


def phase5_spotcheck(run: Run, n_shapes: int | None = None, k: int | None = None) -> pd.DataFrame:
    """Live smart-k (``triton.autotune`` + ArchTune pruning) on held-out shapes vs the sweep's best."""
    n_shapes = config.EXPERIMENTS.spotcheck_shapes if n_shapes is None else n_shapes
    k = config.EXPERIMENTS.spotcheck_k if k is None else k
    tag = run.env["gpu_tag"]
    reg, smem = load_models(run.results, tag)
    summ = json.loads((run.gpu_dir / "analysis" / "summary.json").read_text())
    sw = model.prepare(load_sweep(run.results, tag))
    test_ids = [s for s, v in summ["split"].items() if v == "test" and s.startswith("mm_")][:n_shapes]
    a = gpu.live()
    rows = []
    for sid in test_ids:
        r0 = sw[sw.shape_id == sid].iloc[0]
        live = model.tune_live(matmul_configs(run), int(r0.M), int(r0.N), int(r0.K),
                               prune_by=model.archtune_prune_configs_by(a, k, reg, smem))
        best = model.best_time(sw[sw.shape_id == sid])
        rows.append({"shape_id": sid, "tuning_s": live["tuning_s"], "time_ms": live["time_ms_median"],
                     "sweep_best_ms": best, "quality": best / live["time_ms_median"], "correct": live["correct"]})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_parquet(run.gpu_dir / "spotcheck.parquet")
    return out


def phase6_experiments(run: Run) -> dict[str, pd.DataFrame]:
    """Register caps on three representative configs; pipelining depth on the fastest configs of
    two shapes."""
    ex = config.EXPERIMENTS
    tag = run.env["gpu_tag"]
    sw = load_sweep(run.results, tag)
    by_id = {s.shape_id: s for s in shapes(run)[0]}
    out: dict[str, pd.DataFrame] = {}
    ok, why = gpu.maxnreg_supported()
    print(why)
    if ok:
        reps = representative_configs(sw, ex.large_shape_id)
        tasks = make_tasks([by_id[ex.large_shape_id]], lambda s: with_maxnreg(reps))
        run_sweep(tasks, run.results, run.env, run_id=ex.maxnreg_run_id, table="experiments_maxnreg")
        mx = consolidate(run.gpu_dir, "experiments_maxnreg")
        out["maxnreg"] = maxnreg_frame(mx, gpu.live())
        fig3_maxnreg(out["maxnreg"], f"Register cap sweep — {tag}, {ex.large_shape_id}",
                     run.figures / f"fig3_maxnreg_{tag}.png")
        print(out["maxnreg"].to_string(index=False))
    v = model.prepare(sw)
    v = v[v.valid & (v.kernel == "matmul")]
    tasks = []
    for sid in dict.fromkeys([ex.large_shape_id, sorted(v.shape_id.unique())[0]]):
        for _, r in v[v.shape_id == sid].nsmallest(ex.stages_top_configs, "time_ms_median").iterrows():
            tasks += make_tasks([by_id[sid]], lambda s, c=model.row_config(r): with_num_stages([c]))
    run_sweep(tasks, run.results, run.env, run_id=ex.stages_run_id, table="experiments_stages")
    st = consolidate(run.gpu_dir, "experiments_stages")
    out["stages"] = num_stages_frame(st)
    fig4_num_stages(out["stages"], f"Pipelining depth — {tag}", run.figures / f"fig4_num_stages_{tag}.png")
    print(out["stages"].groupby("num_stages")[["shared_bytes", "ptx_cp_async", "n_regs", "time_rel"]].median())
    return out


def phase8_integration(run: Run, ks: tuple[int, ...] | None = None, unpruned: bool = True,
                       inductor: bool = True) -> pd.DataFrame:
    """Live ``triton.autotune`` with ArchTune pruning on three shapes; the unpruned baseline on the
    large shape only (it compiles and benchmarks every config); optionally Inductor."""
    ex = config.EXPERIMENTS
    ks = ex.integration_ks if ks is None else ks
    tag = run.env["gpu_tag"]
    models = load_models(run.results, tag)
    kw = {"reg_model": models[0], "smem_model": models[1]} if models else {}
    a = gpu.live()
    targets = [s for s in shapes(run)[0] if s.shape_id in ex.integration_shapes]
    cfgs = matmul_configs(run)
    rows = []
    for s in targets:
        for k in ks:
            r = model.tune_live(cfgs, s.M, s.N, s.K, prune_by=model.archtune_prune_configs_by(a, k, **kw))
            rows.append({"mode": f"archtune_top{k}", "shape_id": s.shape_id, **r})
            print(f"{rows[-1]['mode']:16s} {s.shape_id:22s} tune {r['tuning_s']:7.1f}s  {r['time_ms_median']:.4f} ms")
    if unpruned:
        s = next(s for s in targets if s.shape_id == ex.large_shape_id)
        r = model.tune_live(cfgs, s.M, s.N, s.K, prune_by=None)
        rows.append({"mode": "unpruned", "shape_id": s.shape_id, **r})
        print(f"{'unpruned':16s} {s.shape_id:22s} tune {r['tuning_s']:7.1f}s  {r['time_ms_median']:.4f} ms")
    out = pd.DataFrame(rows)
    out["best_cfg"] = out["best_cfg"].astype(str)
    out.to_parquet(run.gpu_dir / "integration.parquet")
    if inductor:
        ind = []
        for s in targets:
            try:
                ind.append({"mode": "inductor", "shape_id": s.shape_id, **model.inductor_compile_and_time(s.M, s.N, s.K)})
                print(f"inductor {s.shape_id}: {ind[-1]['compile_and_tune_s']:.1f}s, {ind[-1]['time_ms_median']:.4f} ms")
            except Exception as exc:  # noqa: BLE001 - optional baseline
                print(f"inductor {s.shape_id} failed (optional): {exc}")
        if ind:
            pd.DataFrame(ind).to_parquet(run.gpu_dir / "inductor.parquet")
    return out


def phase9_liveness(run: Run) -> pd.DataFrame:
    """Peak PTX liveness vs ptxas registers for every compiled matmul config."""
    tag = run.env["gpu_tag"]
    sw = load_sweep(run.results, tag)
    rows = []
    for r in sw[(sw.kernel == "matmul") & sw.artifact_path.notna()].drop_duplicates("cfg_hash").itertuples():
        ptx = gpu.load_artifact(r.artifact_path, "ptx")
        if ptx:
            rows.append({"cfg_hash": r.cfg_hash, "n_regs": r.n_regs, "n_spills": r.n_spills,
                         **compiler.peak_pressure(ptx)})
    lv = pd.DataFrame(rows)
    lv.to_parquet(run.gpu_dir / "ptx_liveness.parquet")
    fig_liveness(lv, f"PTX liveness vs ptxas — {tag}", run.figures / f"fig_liveness_{tag}.png")
    print(lv[["n_regs", "peak_live_units"]].describe().round(1))
    return lv


def make_report(run: Run) -> str:
    """Regenerate RESULTS.md and the README block (a separate report for a quick run)."""
    if run.mini:
        docs = run.results / config.PATHS.docs
        docs.mkdir(exist_ok=True)
        write_report(run.results, docs, run.results / "none.md")
        return (docs / "RESULTS.md").read_text(encoding="utf-8")
    cwd = os.getcwd()
    try:  # relative paths, so the README links to results/figures/... rather than an absolute path
        os.chdir(run.project)
        write_report()
    finally:
        os.chdir(cwd)
    return (run.project / config.PATHS.docs / "RESULTS.md").read_text(encoding="utf-8")


#: The steps in run order. A new step is a function ``step(run, **kw)`` added here.
STEPS: dict[str, Callable[..., Any]] = {
    "phase0_smoke": phase0_smoke, "phase2_sweep": phase2_sweep, "phase3_ir": phase3_ir,
    "phase4_5_analysis": phase4_5_analysis, "phase5_spotcheck": phase5_spotcheck,
    "phase6_experiments": phase6_experiments, "phase8_integration": phase8_integration,
    "phase9_liveness": phase9_liveness, "make_report": make_report,
}


def timed(label: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run one step between banners and print how long it took."""
    print(f"===== {label} =====")
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"===== {label}: done in {(time.perf_counter() - t0) / 60:.1f} min =====")
    return out


def run_pipeline(project: str | Path = PROJECT_ROOT, mini: bool = False, only: Sequence[str] | None = None,
                 step_kwargs: dict[str, dict[str, Any]] | None = None) -> int:
    """Run the steps in order (or just *only*); stops at the first failure. Returns an exit code."""
    run = setup(project, mini)
    t0 = time.perf_counter()
    for name in only or STEPS:
        try:
            out = timed(name, STEPS[name], run, **(step_kwargs or {}).get(name, {}))
            if name == "make_report":
                print(out[:3000])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"!!!!! {name} FAILED after {(time.perf_counter() - t0) / 60:.1f} min total")
            return 1
        sys.stdout.flush()
    print(f"ALL STEPS DONE in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


# =========================================================================== live demo


def demo_autotune(M: int = 4096, N: int = 4096, K: int = 4096, k: int = 8, unpruned: bool = False,
                  results_dir: str | Path | None = None) -> dict[str, Any]:
    """Time ``triton.autotune`` on one matmul with ArchTune pruning, and optionally without.

    Uses the fitted constants in ``results/<gpu_tag>/models.json`` when a full run produced them,
    otherwise the analytic predictor.
    """

    def show(label: str, r: dict[str, Any]) -> None:
        print(f"{label:22s} tuning {r['tuning_s']:8.1f} s | kernel {r['time_ms_median']:.3f} ms | "
              f"correct={r['correct']} | best config {r['best_cfg']}")

    e = gpu.capture()
    if e["gpu_tag"] is None:
        raise RuntimeError("No CUDA GPU visible.")
    print(f"GPU {e['gpu_name']} | torch {e['torch_version']} | triton {e['triton_version']}")
    a = gpu.live()
    models = load_models(results_dir or PROJECT_ROOT / config.PATHS.results, e["gpu_tag"])
    kw = {"reg_model": models[0], "smem_model": models[1]} if models else {}
    print("predictor:", "fitted constants from results/" if models else "analytic (no fitted model found)")
    configs = model.matmul_space()
    print(f"GEMM {M}x{N}x{K}, {len(configs)} candidate configs, keeping top {k}\n")
    print(f"ArchTune: predicting all {len(configs)} configs, compiling only the top {k} ...")
    r = model.tune_live(configs, M, N, K, prune_by=model.archtune_prune_configs_by(a, k, **kw))
    show(f"ArchTune top-{k}", r)
    out = {"archtune": r}
    if unpruned:
        print("running the unpruned baseline (compiles and benchmarks every config; this takes a while)...")
        u = model.tune_live(configs, M, N, K, prune_by=None)
        show("Unpruned", u)
        print(f"\nspeedup {u['tuning_s'] / r['tuning_s']:.0f}x, kernel "
              f"{(r['time_ms_median'] / u['time_ms_median'] - 1) * 100:+.1f}% vs the unpruned choice")
        out["unpruned"] = u
    return out


# =========================================================================== preflight
#
# Every part of the pipeline runs on a tiny workload and gets PASS / WARN / FAIL with its traceback.
# Stages don't stop each other, so one run shows every problem at once. The report goes to
# <out>/<gpu_tag>/preflight.md.

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"


@dataclass
class Stage:
    """Outcome of one preflight stage."""

    name: str
    status: str = PASS
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    error: str | None = None


class Preflight:
    """Runs stages and collects their outcomes; stages call :meth:`note`, :meth:`warn`, :meth:`check`."""

    def __init__(self, out: Path, log: Callable[[str], None] = print) -> None:
        self.out = out
        self.log = log
        self.stages: list[Stage] = []
        self.ctx: dict[str, Any] = {}
        self._cur: Stage | None = None

    def note(self, msg: str) -> None:
        assert self._cur is not None
        self._cur.notes.append(msg)
        self.log(f"    {msg}")

    def warn(self, msg: str) -> None:
        """Downgrade the current stage to WARN without stopping it."""
        assert self._cur is not None
        if self._cur.status == PASS:
            self._cur.status = WARN
        self.note(f"WARNING: {msg}")

    def check(self, cond: bool, msg: str) -> None:
        """Fail the current stage when *cond* is false."""
        if not cond:
            raise AssertionError(msg)

    def run(self, name: str, fn: Callable[["Preflight"], None], requires: tuple[str, ...] = ()) -> None:
        """Run one stage; skip it when a context key it needs is missing."""
        st = Stage(name)
        self.stages.append(st)
        self._cur = st
        missing = [r for r in requires if r not in self.ctx]
        if missing:
            st.status = SKIP
            st.notes.append(f"skipped: needs {missing} from an earlier stage")
            self.log(f"[{SKIP}] {name} ({st.notes[-1]})")
            return
        self.log(f"[....] {name}")
        t0 = time.perf_counter()
        try:
            fn(self)
        except Exception as exc:  # noqa: BLE001 - keep going, that is the point
            st.status = FAIL
            st.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        st.seconds = time.perf_counter() - t0
        self.log(f"[{st.status}] {name} ({st.seconds:.1f}s)" + (f"\n{st.error}" if st.error else ""))

    @property
    def failed(self) -> bool:
        return any(s.status == FAIL for s in self.stages)

    def report(self) -> str:
        """Markdown report."""
        env = self.ctx.get("env", {})
        lines = ["# ArchTune preflight report", "",
                 f"GPU `{env.get('gpu_name')}` ({env.get('gpu_tag')}), torch {env.get('torch_version')}, "
                 f"Triton {env.get('triton_version')}, CUDA {env.get('cuda')}, driver {env.get('driver')}", "",
                 "| stage | status | seconds | notes |", "|---|---|---|---|"]
        for s in self.stages:
            notes = "<br>".join(n.replace("|", "\\|") for n in s.notes[:6])
            lines.append(f"| {s.name} | **{s.status}** | {s.seconds:.1f} | {notes} |")
        for s in self.stages:
            if s.error:
                lines += ["", f"## {s.name} — FAIL", "```", s.error.strip(), "```"]
        verdict = "FAIL — fix before running the full notebooks" if self.failed else \
            "OK — no stage failed (read the WARN notes)"
        lines += ["", f"**Verdict: {verdict}**"]
        return "\n".join(lines) + "\n"


_SMOKE_CFG = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
              "num_warps": 4, "num_stages": 3}


def _mm_compile(cfg: dict[str, Any], M: int = 1024, N: int = 1024, K: int = 1024):
    inputs = gpu.MATMUL.make_inputs(M, N, K)
    return gpu.compile_kernel(gpu.MATMUL, inputs, cfg), inputs


def st_env(p: Preflight) -> None:
    e = gpu.capture()
    p.ctx["env"] = e
    p.note(f"{e['gpu_name']} {e['gpu_tag']} torch={e['torch_version']} triton={e['triton_version']} cuda={e['cuda']}")
    p.check(e["gpu_tag"] is not None, "no CUDA GPU visible to torch")
    if not gpu.pinned_ok(e):
        p.warn(f"stack differs from pinned torch {config.STACK.torch} / triton {config.STACK.triton}; "
               "results would not be comparable")


def st_arch(p: Preflight) -> None:
    a = gpu.live()
    p.ctx["arch"] = a
    p.note(f"{a.name} cc={a.cc} SMs={a.sm_count} smem/block={a.smem_per_block_optin} "
           f"peak={a.peak_fp16_tflops:.1f} TFLOP/s bw={a.mem_bw_gbs:.0f} GB/s")
    p.check(not math.isnan(a.peak_fp16_tflops) and not math.isnan(a.mem_bw_gbs),
            "peak throughput unknown -> predictor score would be NaN")
    key = gpu.reference_for_name(a.name)
    if key:
        diffs = gpu.verify(gpu.reference(key), a)
        if diffs:
            p.warn(f"live vs reference table differs: {diffs}")
    else:
        p.note("no datasheet entry: peaks estimated from clocks (" +
               ", ".join(f"{k}={v}" for k, v in a.provenance.items() if "estimated" in v) + ")")


def st_api(p: Preflight) -> None:
    api = gpu.api_report()
    (p.out / "api_check.json").write_text(json.dumps(api, indent=2, default=str))
    p.ctx["api"] = api
    p.check(api["JITFunction.warmup"] is not None, "JITFunction.warmup missing: compile timing falls back")
    p.check(api["Autotuner.prune_configs"] is not None, "Autotuner.prune_configs missing")
    calls = api.get("prune_configs_calls") or {}
    if not all(calls.values()):
        p.warn(f"prune_configs source differs from assumption: {calls}")
    if not api["maxnreg_in_Config"]:
        p.warn("triton.Config has no maxnreg: the register-cap experiment will be skipped")
    flush = api.get("do_bench_flushes_l2") or {}
    if not any(flush.values()):
        p.warn(f"could not confirm do_bench L2 flush: {flush}")
    cache = api.get("cache_dir_env_read_at_init")
    if isinstance(cache, dict) and not all(cache.values()):
        p.warn("TRITON_CACHE_DIR is not read in FileCacheManager.__init__: cold-compile detection may be wrong")
    p.note(f"do_bench{api['do_bench']}")


def st_compile(p: Preflight) -> None:
    gpu.fresh_cache_dir()
    res, _ = _mm_compile(_SMOKE_CFG)
    p.check(res.status == "ok", f"compile failed: {res.status} {res.error_msg}")
    attrs = gpu.compiled_kernel_attrs(res.compiled)
    (p.out / "compiled_kernel_attrs.json").write_text(json.dumps(
        {**attrs, "describe": gpu.describe_compiled(res.compiled)}, indent=2, default=str))
    p.note(f"compile {res.compile_time_s:.2f}s via {res.method}, cache_miss={res.cache_miss}; "
           f"n_regs={res.n_regs} n_spills={res.n_spills} shared={res.shared_bytes}; asm={attrs['asm_keys']}")
    p.check(isinstance(res.n_regs, int) and res.n_regs > 0, f"n_regs not captured: {res.n_regs!r}")
    p.check(res.n_spills is not None, "n_spills not captured")
    p.check(isinstance(res.shared_bytes, int) and res.shared_bytes > 0, f"shared_bytes not captured: {res.shared_bytes!r}")
    p.check(bool(res.asm.get("ptx")) and bool(res.asm.get("cubin")), "ptx/cubin missing from asm")
    if res.cache_miss is not True:
        p.warn("first compile not detected as a cache miss: check TRITON_CACHE_DIR handling")
    res2, _ = _mm_compile(_SMOKE_CFG)
    p.note(f"second compile {res2.compile_time_s:.3f}s cache_miss={res2.cache_miss}")
    if res2.compile_time_s is not None and res.compile_time_s and res2.compile_time_s > 0.5 * res.compile_time_s:
        p.warn("recompiling the same config was not much faster: in-memory cache not hit?")
    p.ctx["asm"] = res.asm


def st_correct(p: Preflight) -> None:
    import torch

    mm, rn = gpu.MATMUL, gpu.RMSNORM
    for M, N, K in ((1024, 1024, 1024), (16, 1600, 960), (100, 70, 50)):  # the last one is ragged
        a, b, c = mm.make_inputs(M, N, K)
        mm.launcher(a, b, c, _SMOKE_CFG)()
        torch.cuda.synchronize()
        ok, err = gpu.check_correct(c, mm.reference(a, b))
        p.note(f"matmul {M}x{N}x{K}: correct={ok} max_abs_err={err:.3g}")
        p.check(ok, f"matmul {M}x{N}x{K} incorrect (max_abs_err={err})")
    for bs, w in ((128, 4), (1024, 8), (256, 1)):
        x, wt, y = rn.make_inputs(64, 960)
        rn.launcher(x, wt, y, {"BLOCK_SIZE": bs, "num_warps": w})()
        torch.cuda.synchronize()
        ok, err = gpu.check_correct(y, rn.reference(x, wt))
        p.note(f"rmsnorm 64x960 BLOCK={bs} warps={w}: correct={ok} err={err:.3g}")
        p.check(ok, f"rmsnorm BLOCK={bs} incorrect (err={err})")


def st_edge_configs(p: Preflight) -> None:
    a = p.ctx["arch"]
    # A 256x256 tile with BLOCK_K=128 needs 128 KB even with one buffer, so it is too big on every
    # target GPU whether or not Triton multi-buffers (Triton 3.2 single-buffers on sm_75).
    oor = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    res, _ = _mm_compile(oor)
    p.note(f"oversized config -> status={res.status} shared={res.shared_bytes} ({(res.error_msg or '')[:80]})")
    p.check(res.status == "out_of_resources",
            f"expected out_of_resources (limit {a.smem_per_block_optin} B), got {res.status} with "
            f"shared={res.shared_bytes}: {res.error_msg}")
    for stages in (2, 5):  # what num_stages does to shared memory on this GPU
        cfg = {**_SMOKE_CFG, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "num_stages": stages}
        r, _ = _mm_compile(cfg)
        p.note(f"128x128x32 num_stages={stages}: status={r.status} shared={r.shared_bytes}")
        if r.status == "ok":
            p.check(r.shared_bytes <= a.smem_per_block_optin, "a config ran with more shared memory than the limit")
    spill = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 2}
    res, _ = _mm_compile(spill)
    p.note(f"512-acc-reg config -> status={res.status} n_regs={res.n_regs} n_spills={res.n_spills} smem={res.shared_bytes}")
    p.check(res.status in ("ok", "out_of_resources"), f"unexpected status {res.status}: {res.error_msg}")
    if res.status == "ok" and not res.n_spills:
        p.warn("expected register spills for a 256x128 tile on 2 warps but n_spills=0")


def st_sass(p: Preflight) -> None:
    tools = {t: gpu.find_cuda_tool(t) for t in ("cuobjdump", "nvdisasm")}
    p.note(f"tools: {tools}")
    sass = gpu.disassemble(p.ctx["asm"].get("cubin"))
    if not sass:
        p.warn("no SASS (cuobjdump/nvdisasm unavailable): sass_* columns will be null")
        return
    c = compiler.count_sass(sass)
    p.note(f"SASS counts: {c}")
    if c["hmma"] == 0:
        p.warn("no HMMA in SASS: tensor cores unused or the SASS regex does not match this format")
    pc = compiler.count_ptx(p.ctx["asm"]["ptx"])
    p.note(f"PTX counts: mma={pc['mma']} ldmatrix={pc['ldmatrix']} cp_async={pc['cp_async']} ld_local={pc['ld_local']}")
    if pc["mma"] == 0:
        p.warn("no mma.sync in PTX: check the PTX regex against this Triton's output")


def st_noise(p: Preflight) -> None:
    mm = gpu.MATMUL
    a, b, c = mm.make_inputs(2048, 2048, 2048)
    fn = mm.launcher(a, b, c, {**_SMOKE_CFG, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128})
    t = gpu.time_config(fn)
    p.note(f"time_config: {t}")
    p.check(t["time_ms_p10"] <= t["time_ms_median"] <= t["time_ms_p90"], "quantiles out of order")
    n = gpu.noise_check(fn, n=5)
    p.note(f"noise CV over 5 reps = {n['cv']:.4f} (the real noise check uses {config.BENCH.noise_runs}); samples ms "
           f"{[round(x, 4) for x in n['samples_ms']]}, discarded first {n['discarded_first_ms']:.4f}")
    if n["cv"] > 0.05:
        p.warn("timing CV > 5%: thermal/clock noise; ties threshold will be wide")


def _mini_workload(p: Preflight) -> tuple[list[Shape], list[dict[str, Any]]]:
    try:
        cfg = model.load_model_config(cache_dir=p.out)
        p.note("model config fetched from the Hugging Face Hub")
    except Exception as exc:  # noqa: BLE001
        fixture = PROJECT_ROOT / "tests" / "fixtures" / "SmolLM2-360M.config.json"
        p.warn(f"Hub fetch failed ({exc}); using recorded fixture — notebooks need network access")
        cfg = json.loads(fixture.read_text())
    return model.mini_shapes(model.all_shapes(cfg)), model.mini_matmul_space()


def st_sweep(p: Preflight) -> None:
    shapes_, mm_cfgs = _mini_workload(p)
    n_rms = config.WORKLOAD.mini_rmsnorm_configs
    tasks = make_tasks(shapes_, lambda s: mm_cfgs if s.kernel == "matmul" else model.rmsnorm_space(s.N)[:n_rms])
    e = p.ctx["env"]
    logs: list[str] = []
    s1 = run_sweep(tasks, p.out, e, limit=7, log=logs.append, flush_every=3)  # an interrupted session
    s2 = run_sweep(tasks, p.out, e, log=logs.append, flush_every=5)
    s3 = run_sweep(tasks, p.out, e, log=logs.append)
    p.note(f"{len(tasks)} tasks: session1={s1['n_evaluated']} session2={s2['n_evaluated']} "
           f"session3={s3['n_evaluated']} ({s1['elapsed_s'] + s2['elapsed_s']:.0f}s)")
    p.check(s1["n_evaluated"] == 7 and s2["n_evaluated"] == len(tasks) - 7 and s3["n_evaluated"] == 0,
            "resume did not skip already-written rows correctly")
    df = consolidate(p.out / e["gpu_tag"])
    p.check(len(df) == len(tasks), f"{len(df)} rows for {len(tasks)} tasks")
    p.check(not df.duplicated(list(KEY_COLUMNS)).any(), "duplicate rows after resume")
    p.note(f"status counts: {df.groupby(['kernel', 'status']).size().to_dict()}")
    bad = df[df["status"].isin(["compile_error", "launch_error", "incorrect"])]
    for r in bad.head(5).itertuples():
        p.note(f"{r.kernel} {r.shape_id} {r.cfg_hash} {r.status}: {str(r.error_msg)[:200]}")
    p.check(bad.empty, f"{len(bad)} rows with compile_error/launch_error/incorrect (see notes)")
    ok = df[df["status"] == "ok"]
    for col in ("compile_time_s", "n_regs", "n_spills", "shared_bytes", "time_ms_median", "bench_time_s",
                "ptx_mma", "artifact_path"):
        p.check(col in ok and ok[col].notna().all(), f"column {col} missing or null on ok rows")
    if "sass_hmma" not in df or df["sass_hmma"].isna().all():
        p.warn("sass_* columns are null")
    miss = df[df["kernel"] == "matmul"].groupby("cfg_hash")["cache_miss"].max()
    if not miss.astype(bool).all():
        p.warn("some configs never recorded a cache-miss compile: replay would charge ~0 compile cost")
    est = estimate_seconds(df[df["status"] == "ok"].head(10), n_tasks=18 * 384 + 4 * 20, n_cold_compiles=384 + 20)
    p.note(f"extrapolated FULL sweep on this GPU: {est['estimate_s'] / 3600:.2f} h "
           f"({est['per_cold_compile_s']:.2f}s/compile, {est['per_task_s']:.2f}s/task)")
    p.check((p.out / e["gpu_tag"] / "progress.log").exists(), "progress.log not written")
    p.ctx["sweep_done"] = True


def st_ir(p: Preflight) -> None:
    asm = p.ctx["asm"]
    paths = compiler.dump_ir(asm, p.out / p.ctx["env"]["gpu_tag"] / "ir_exploration", "preflight")
    p.check({"ttir", "ttgir", "llir", "ptx"} <= set(paths), f"IR stages missing: {sorted(paths)}")
    lay = compiler.ttgir_layouts(asm.get("ttgir", ""))
    p.note(f"{len(lay)} TTGIR layout attributes")
    if not lay:
        p.warn("no #blocked/#mma layouts parsed from TTGIR (regex may not match this Triton)")


def st_analysis(p: Preflight) -> None:
    tag = p.ctx["env"]["gpu_tag"]
    s = analyze(p.out, tag, n_seeds=3, refit=True)
    pm = pd.read_parquet(p.out / tag / "analysis" / "predictor.parquet")
    p.check(len(pm) > 0, "empty predictor table")
    m = s["predictor"]["fitted"]
    p.note(f"regs median|err|={m['regs_nonspilling']['median_abs_err']}, smem exact={m['smem']['exact_match']}, "
           f"spill P/R={m['spill']['precision']}/{m['spill']['recall']} (12 configs: numbers meaningless)")
    t6 = pd.read_csv(p.out / tag / "analysis" / "table6.csv")
    p.check({"random", "smart", "exhaustive"} <= set(t6["strategy"]), f"strategies missing: {set(t6['strategy'])}")
    curves = pd.read_parquet(p.out / tag / "analysis" / "curves.parquet")
    p.check(curves["cost_s"].notna().all() and curves["quality"].between(0, 1.0 + 1e-9).all(),
            "curves have NaN cost or quality outside [0, 1]")
    fig1_quality_vs_cost(curves, tag, p.out / "figures" / "fig1.png")
    fig2_predictor(pm, "preflight", p.out / "figures" / "fig2.png")


def st_experiments(p: Preflight) -> None:
    ex = config.EXPERIMENTS
    e = p.ctx["env"]
    shapes_, _ = _mini_workload(p)
    shape = [s for s in shapes_ if s.shape_id == ex.large_shape_id]
    base = model.mini_matmul_space()[1:3]
    ok, why = gpu.maxnreg_supported()
    p.note(why)
    if ok:
        tasks = make_tasks(shape, lambda s: with_maxnreg(base, caps=(None, 128, 64)))
        run_sweep(tasks, p.out, e, run_id=ex.maxnreg_run_id, table="experiments_maxnreg", log=lambda m: None)
        mx = consolidate(p.out / e["gpu_tag"], "experiments_maxnreg")
        p.note(f"maxnreg rows: {mx.groupby('status').size().to_dict()}")
        p.check((mx["status"] == "ok").any(), "no maxnreg config ran")
        if "cfg_maxnreg" not in mx:
            p.warn("no cfg_maxnreg column: caps were not recorded")
        else:
            capped = mx[mx["cfg_maxnreg"].notna() & (mx["status"] == "ok")]
            p.note(f"max n_regs by cap: {capped.groupby('cfg_maxnreg')['n_regs'].max().to_dict()}")
            if (capped["n_regs"] > capped["cfg_maxnreg"]).any():
                p.warn("maxnreg was not honoured (n_regs > cap)")
        fig3_maxnreg(maxnreg_frame(mx, p.ctx["arch"]), "preflight", p.out / "figures" / "fig3.png")
    else:
        p.warn("maxnreg unsupported")
    tasks = make_tasks(shape, lambda s: with_num_stages(base[:1], stages=(1, 2, 3)))
    run_sweep(tasks, p.out, e, run_id=ex.stages_run_id, table="experiments_stages", log=lambda m: None)
    st = consolidate(p.out / e["gpu_tag"], "experiments_stages")
    p.note(f"stages: {st[['cfg_num_stages', 'status', 'shared_bytes', 'ptx_cp_async']].to_dict('records')}")
    p.check((st["status"] == "ok").all(), "a num_stages config failed")
    fig4_num_stages(num_stages_frame(st), "preflight", p.out / "figures" / "fig4.png")


def st_hook(p: Preflight) -> None:
    tag = p.ctx["env"]["gpu_tag"]
    models = load_models(p.out, tag)
    kw = {"reg_model": models[0], "smem_model": models[1]} if models else {}
    cfgs = model.mini_matmul_space()
    r_pruned = model.tune_live(cfgs, 1024, 1024, 1024,
                               prune_by=model.archtune_prune_configs_by(p.ctx["arch"], 3, **kw))
    p.note(f"pruned k=3: tuning {r_pruned['tuning_s']:.1f}s, best {r_pruned['best_cfg']}, "
           f"{r_pruned['time_ms_median']:.4f} ms, kept {r_pruned.get('n_after_prune')}")
    p.check(r_pruned["correct"], "pruned autotune result incorrect")
    p.check(r_pruned.get("n_after_prune") == 3, f"pruning kept {r_pruned.get('n_after_prune')} configs, expected 3")
    r_full = model.tune_live(cfgs, 1024, 1024, 1024, prune_by=None)
    p.note(f"unpruned (12 configs incl. an out_of_resources one): tuning {r_full['tuning_s']:.1f}s, "
           f"{r_full['time_ms_median']:.4f} ms")
    p.check(r_full["correct"], "unpruned autotune result incorrect")


def st_inductor(p: Preflight) -> None:
    r = model.inductor_compile_and_time(512, 512, 512)
    p.note(f"inductor max-autotune: {r['compile_and_tune_s']:.1f}s, {r['time_ms_median']:.4f} ms, correct={r['correct']}")
    p.check(r["correct"], "inductor result incorrect")
    os.environ[config.INDUCTOR_PATCH_ENV] = "1"
    try:
        a = p.ctx["arch"]
        ok, msg = model.patch_inductor_mm_configs(
            lambda cfg, m, n, k: model.predict_matmul(cfg, m, n, k, a)["feasible"])
        p.note(f"patch: {ok} ({msg})")
        if not ok:
            p.warn(f"Inductor patch unavailable on this torch: {msg} (optional feature)")
        else:
            r2 = model.inductor_compile_and_time(512, 512, 512)
            p.note(f"patched: {r2['compile_and_tune_s']:.1f}s, correct={r2['correct']}")
    finally:
        model.unpatch_inductor()
        os.environ.pop(config.INDUCTOR_PATCH_ENV, None)


def st_liveness(p: Preflight) -> None:
    sw = load_sweep(p.out, p.ctx["env"]["gpu_tag"])
    rows = sw[(sw.kernel == "matmul") & sw.artifact_path.notna()].drop_duplicates("cfg_hash")
    out = []
    for r in rows.itertuples():
        ptx = gpu.load_artifact(r.artifact_path, "ptx")
        if ptx:
            out.append((r.n_regs, compiler.peak_pressure(ptx)))
    p.check(bool(out), "no PTX artifacts could be read back")
    p.note("ptxas regs vs PTX peak-live: " + ", ".join(f"{a}/{b['peak_live_units']}" for a, b in out[:8]))
    p.check(all(b["n_instructions"] > 50 and b["n_blocks"] >= 2 for _, b in out),
            "parser found too few instructions/blocks in real Triton PTX")


def st_report(p: Preflight) -> None:
    docs = p.out / "docs"
    docs.mkdir(exist_ok=True)
    readme = p.out / "README.md"
    readme.write_text(f"# preflight\n{README_BEGIN}\n{README_END}\n", encoding="utf-8")
    write_report(p.out, docs, readme)
    p.check((docs / "RESULTS.md").exists(), "RESULTS.md not written")


STAGES: list[tuple[str, Callable[[Preflight], None], tuple[str, ...]]] = [
    ("1 environment + pinned stack", st_env, ()),
    ("2 architecture parameters", st_arch, ("env",)),
    ("3 Triton/PyTorch API check", st_api, ("env",)),
    ("4 compile + inspect one config", st_compile, ("env",)),
    ("5 correctness (matmul incl. ragged shape, rmsnorm)", st_correct, ("env",)),
    ("6 edge configs (out_of_resources, spills)", st_edge_configs, ("arch",)),
    ("7 SASS tools + PTX/SASS regexes", st_sass, ("asm",)),
    ("8 timing + noise", st_noise, ("env",)),
    ("9 mini sweep with interruption + resume", st_sweep, ("env",)),
    ("10 IR dump", st_ir, ("asm",)),
    ("11 predictor + replay + figures 1-2", st_analysis, ("sweep_done",)),
    ("12 maxnreg + num_stages + figures 3-4", st_experiments, ("arch",)),
    ("13 triton.autotune hook", st_hook, ("arch",)),
    ("14 Inductor max-autotune (optional)", st_inductor, ("arch",)),
    ("15 PTX liveness on real PTX", st_liveness, ("sweep_done",)),
    ("16 report generation", st_report, ("sweep_done",)),
]


def preflight(out: str | Path | None = None, skip_inductor: bool = False) -> int:
    """Run every preflight stage into *out* (wiped first); returns 1 if any stage failed."""
    out = Path(out or config.PATHS.preflight)
    if out.resolve().name == config.PATHS.results:
        raise SystemExit("refusing to write preflight data into results/")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pf = Preflight(out)
    t0 = time.perf_counter()
    for name, fn, req in STAGES:
        if skip_inductor and fn is st_inductor:
            continue
        pf.run(name, fn, req)
    md = pf.report()
    tag = pf.ctx.get("env", {}).get("gpu_tag") or "unknown"
    (out / tag).mkdir(parents=True, exist_ok=True)
    (out / tag / "preflight.md").write_text(md, encoding="utf-8")
    (out / tag / "preflight.json").write_text(json.dumps([asdict(s) for s in pf.stages], indent=2))
    print("\n" + md)
    print(f"total {time.perf_counter() - t0:.0f}s; report: {out / tag / 'preflight.md'}")
    return 1 if pf.failed else 0


# =========================================================================== report
#
# Figures, docs/RESULTS.md and the README results block, built from what is saved in results/.
# Sections whose data doesn't exist yet are marked as pending.

README_BEGIN = "<!-- RESULTS:BEGIN (generated by archtune.run report; do not edit) -->"
README_END = "<!-- RESULTS:END -->"


def md_table(df: pd.DataFrame | None, floatfmt: str = "{:.4g}") -> str:
    """A DataFrame as a GitHub markdown table."""
    if df is None or df.empty:
        return "_no data_\n"

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isnan(v):
                return "–"
            if v.is_integer() and abs(v) < 1e9:
                return str(int(v))
            return f"{v:.1f}" if abs(v) >= 1000 else floatfmt.format(v)
        return str(v)

    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def gpu_dirs(results: Path) -> list[Path]:
    """GPU result folders (``<tag>_sm<cc>``) in a stable order."""
    dirs = [p for p in results.iterdir() if p.is_dir() and "_sm" in p.name] if results.exists() else []
    order = {t: i for i, t in enumerate(config.PALETTE["gpu_order"])}
    return sorted(dirs, key=lambda p: (order.get(p.name, 99), p.name))


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def collect_results(results: Path) -> dict[str, Any]:
    """Load every result file that exists."""
    data: dict[str, Any] = {"gpus": {}}
    for d in gpu_dirs(results):
        g: dict[str, Any] = {"summary": _read_json(d / "analysis" / "summary.json"),
                             "noise": _read_json(d / "noise.json")}
        p = d / "analysis" / "table6.csv"
        g["table6.csv"] = pd.read_csv(p) if p.exists() else None
        for name in ("curves", "predictor", "targets"):
            p = d / "analysis" / f"{name}.parquet"
            g[name] = pd.read_parquet(p) if p.exists() else None
        for name in ("experiments_maxnreg", "experiments_stages", "integration"):
            p = d / f"{name}.parquet"
            g[name] = pd.read_parquet(p) if p.exists() else None
        g["compile_waste"] = None
        if (d / "sweep.parquet").exists():
            sw = pd.read_parquet(d / "sweep.parquet")
            sw = sw[sw["kernel"] == "matmul"]
            cold = sw[sw["cache_miss"].astype("boolean").fillna(False).astype(bool)].drop_duplicates("cfg_hash")
            if len(cold):
                bad = (cold["n_spills"].fillna(0) > 0) | (cold["status"] == "out_of_resources")
                g["compile_waste"] = float(cold.loc[bad, "compile_time_s"].sum() / cold["compile_time_s"].sum())
        feat = d / "ir_exploration" / "features.csv"
        g["ir_features"] = pd.read_csv(feat, index_col=0) if feat.exists() else None
        sess = d / "sessions.jsonl"
        g["sessions"] = pd.read_json(sess, lines=True) if sess.exists() else None
        data["gpus"][d.name] = g
    return data


def build_figures(results: Path, data: dict[str, Any]) -> dict[str, str]:
    """Render every figure the data allows; returns name -> path."""
    fig_dir = results / config.PATHS.figures
    out: dict[str, str] = {}
    stages = []
    for tag, g in data["gpus"].items():
        if g["curves"] is not None:
            p = fig_dir / f"fig1_quality_vs_cost_{tag}.png"
            fig1_quality_vs_cost(g["curves"], tag, p)
            out[f"fig1_{tag}"] = str(p)
        if g["predictor"] is not None:
            p = fig_dir / f"fig2_predictor_{tag}.png"
            fig2_predictor(g["predictor"], f"Predicted vs measured — {tag} (fitted on train split)", p)
            out[f"fig2_{tag}"] = str(p)
        if g["experiments_maxnreg"] is not None:
            mx = g["experiments_maxnreg"]
            sm = int(mx["sm_count"].dropna().iloc[0]) if "sm_count" in mx and mx["sm_count"].notna().any() else None
            p = fig_dir / f"fig3_maxnreg_{tag}.png"
            fig3_maxnreg(maxnreg_frame(mx, arch_for_tag(tag, sm, results)),
                         f"Register cap sweep — {tag}, shape {mx['shape_id'].iloc[0]}", p)
            out[f"fig3_{tag}"] = str(p)
        if g["integration"] is not None and "unpruned" in set(g["integration"]["mode"]):
            sid = g["integration"].loc[g["integration"]["mode"] == "unpruned", "shape_id"].iloc[0]
            p = fig_dir / f"fig_tuning_time_{tag}.png"
            fig_tuning_time(g["integration"], sid, p)
            out[f"tuning_{tag}"] = str(p)
        sweep_p = results / tag / "sweep.parquet"
        if sweep_p.exists():
            p = fig_dir / f"fig_compile_breakdown_{tag}.png"
            fig_compile_breakdown(pd.read_parquet(sweep_p), p)
            out[f"compile_{tag}"] = str(p)
        if g["experiments_stages"] is not None:
            stages.append(num_stages_frame(g["experiments_stages"]))
    if stages:
        p = fig_dir / "fig4_num_stages.png"
        fig4_num_stages(pd.concat(stages), "Pipelining depth (median over configs and shapes)", p)
        out["fig4"] = str(p)
    return out


def _pending(what: str) -> str:
    return f"_Pending: {what}._\n"


def results_markdown(data: dict[str, Any], figs: dict[str, str], rel: str) -> str:
    """The body of RESULTS.md."""
    L = ["# ArchTune results (auto-generated)\n",
         "Generated by `python -m archtune.run report` from `results/`. Do not edit by hand.\n"]
    if not data["gpus"]:
        L.append(_pending("no GPU results yet — run the pipeline first"))
        return "\n".join(L)
    L.append("## Environment and sweep status\n")
    rows = []
    for tag, g in data["gpus"].items():
        s, noise = g["summary"], g["noise"]
        rows.append({"gpu": tag, "stack_id": s["stack_id"] if s else "–", "rows": s["n_rows"] if s else "–",
                     "noise CV": noise["cv"] if noise else float("nan"),
                     "tie threshold": noise["tie_threshold"] if noise else float("nan")})
    L.append(md_table(pd.DataFrame(rows)))
    for tag, g in data["gpus"].items():
        if g["summary"]:
            sc = pd.DataFrame(g["summary"]["status_counts"])
            L.append(f"\n**Status counts — {tag}**\n\n" + md_table(sc.pivot_table(index="kernel", columns="status",
                                                                               values="n", fill_value=0).reset_index()))
        if g["sessions"] is not None and "estimate_s" in g["sessions"]:
            ses = g["sessions"].dropna(subset=["estimate_s"])
            if not ses.empty:
                L.append(f"\n**Small-first estimate vs actual — {tag}**\n\n" +
                         md_table(ses[["table", "n_evaluated", "estimate_s", "elapsed_s"]]))

    L.append("\n## RQ1 — Predictor accuracy (matmul)\n")
    for tag, g in data["gpus"].items():
        s = g["summary"]
        if not s:
            continue
        rows = [{"variant": variant,
                 "regs median |err| (non-spilling)": m["regs_nonspilling"]["median_abs_err"],
                 "regs exact": m["regs_nonspilling"]["exact_match"],
                 "smem exact match": m["smem"]["exact_match"],
                 "smem median |err| (B)": m["smem"]["median_abs_err"],
                 "spill precision": m["spill"]["precision"], "spill recall": m["spill"]["recall"]}
                for variant, m in s["predictor"].items()]
        L.append(f"**{tag}**\n\n" + md_table(pd.DataFrame(rows)))
        if f"fig2_{tag}" in figs:
            L.append(f"\n![fig2 {tag}]({rel}/{Path(figs[f'fig2_{tag}']).name})\n")

    L.append("\n## RQ2 — Search efficiency (replay over exhaustive sweep)\n")
    L.append("Table 6: cost to reach ≥ 95% of the exhaustive best. `reach_rate` = fraction of "
             "shapes (× seeds for random) that reached the target; costs are medians.\n")
    for tag, g in data["gpus"].items():
        if g["table6.csv"] is not None:
            L.append(f"\n**{tag}**\n\n" + md_table(g["table6.csv"]))
            ratios = pd.DataFrame(g["summary"]["cost_ratio_random_over_smart"]) if g["summary"] else None
            if ratios is not None and not ratios.empty:
                L.append("\nMedian cost ratio random-k / smart-k (> 1 means smart is cheaper):\n\n" + md_table(ratios))
        if g["targets"] is not None:
            ps = model.per_shape(g["targets"])
            wins = int((ps["random_over_smart"] > 1).sum())
            L.append(f"\nPer shape (matmul): smart-k cheaper than random-k on **{wins} of {len(ps)}** shapes.\n\n"
                     + md_table(ps))
        if f"fig1_{tag}" in figs:
            L.append(f"\n![fig1 {tag}]({rel}/{Path(figs[f'fig1_{tag}']).name})\n")

    L.append("\n## RQ3 — Compiler evidence: fastest vs slowest valid config (4096³)\n")
    shown_ir = False
    for tag, g in data["gpus"].items():
        if g.get("ir_features") is not None:
            shown_ir = True
            f = g["ir_features"].reset_index().rename(columns={"index": "static count"})
            L.append(f"**{tag}** (static instruction counts; `ptx_*` before, `sass_*` after ptxas register "
                     f"allocation; IR dumps in `results/{tag}/ir_exploration/`)\n\n" + md_table(f))
    if not shown_ir:
        L.append(_pending("run the IR comparison step"))

    L.append("\n## RQ4 — Compiler experiments\n")
    any_exp = False
    for tag in data["gpus"]:
        if f"fig3_{tag}" in figs:
            any_exp = True
            L.append(f"![fig3 {tag}]({rel}/{Path(figs[f'fig3_{tag}']).name})\n")
    if "fig4" in figs:
        any_exp = True
        L.append(f"![fig4]({rel}/{Path(figs['fig4']).name})\n")
    if not any_exp:
        L.append(_pending("run the compiler experiments step"))

    L.append("\n## Integration — live `triton.autotune` with ArchTune pruning\n")
    shown = False
    for tag, g in data["gpus"].items():
        if g["integration"] is not None:
            shown = True
            cols = [c for c in ("mode", "source", "M", "N", "K", "n_configs_given", "n_after_prune", "tuning_s",
                                "time_ms_median", "correct") if c in g["integration"]]
            L.append(f"**{tag}**\n\n" + md_table(g["integration"][cols]))
    if not shown:
        L.append(_pending("run the integration step"))
    return "\n".join(L)


def headline(data: dict[str, Any]) -> dict[str, Any]:
    """The README numbers, read from held-out (test-split) results."""
    out: dict[str, Any] = {}
    for tag, g in data["gpus"].items():
        s = g["summary"]
        if not s:
            continue
        out.setdefault("regs_median_abs_err", s["predictor"]["fitted"]["regs_nonspilling"]["median_abs_err"])
        for r in s["cost_ratio_random_over_smart"]:
            if r["split"] == "test" and r["kernel"] == "matmul":
                out[f"cost_ratio_test_{tag}"] = r["random_over_smart"]
        if g.get("targets") is not None:
            ps = model.per_shape(g["targets"])
            out.setdefault("shape_wins", (int((ps["random_over_smart"] > 1).sum()), len(ps)))
        integ = g.get("integration")
        if integ is not None and {"unpruned", "archtune_top8"} <= set(integ["mode"]):
            big = integ[integ["shape_id"] == integ.loc[integ["mode"] == "unpruned", "shape_id"].iloc[0]]
            un = big[big["mode"] == "unpruned"].iloc[0]
            k8 = big[big["mode"] == "archtune_top8"].iloc[0]
            out.setdefault("live", {"shape": un["shape_id"], "unpruned_s": float(un["tuning_s"]),
                                    "top8_s": float(k8["tuning_s"]),
                                    "slowdown": float(k8["time_ms_median"] / un["time_ms_median"]), "gpu": tag})
        if g.get("compile_waste") is not None:
            out.setdefault("compile_waste", g["compile_waste"])
    return out


def readme_block(data: dict[str, Any], figs: dict[str, str]) -> str:
    """The generated results block of README.md."""
    h = headline(data)
    if not h:
        return (f"{README_BEGIN}\n_No results yet. Run the pipeline (see Reproduce), then "
                f"`python -m archtune.run report`._\n{README_END}")
    lines = [README_BEGIN, "", "| Result | Value |", "|---|---|"]
    if "live" in h:
        lv = h["live"]
        shape = lv["shape"].replace("mm_", "").replace("_", " ")
        lines.append(f"| Live `triton.autotune` time, GEMM {shape} | **{lv['unpruned_s']:,.0f} s → "
                     f"{lv['top8_s']:.1f} s ({lv['unpruned_s'] / lv['top8_s']:.0f}× less)**, chosen kernel "
                     f"{(lv['slowdown'] - 1) * 100:+.1f}% runtime |")
    if h.get("compile_waste") is not None:
        lines.append(f"| Exhaustive search's compile time spent on configs that spill or overflow shared memory "
                     f"| **{h['compile_waste']:.0%}**, the configs ArchTune prunes |")
    ratio = next((v for k, v in h.items() if k.startswith("cost_ratio_test_")), None)
    if ratio is not None:
        lines.append(f"| Tuning cost to reach 95% of the best kernel, held-out shapes (median) | "
                     f"**{ratio:.3g}× lower** than random search |")
    if "shape_wins" in h:
        lines.append(f"| Shapes where ArchTune beats random search | **{h['shape_wins'][0]} of {h['shape_wins'][1]}** |")
    lines.append(f"| Register prediction, non-spilling configs | median error **{h['regs_median_abs_err']:.3g}** "
                 f"registers/thread |")
    if "live" in h:
        lines.append(f"| GPU | `{h['live']['gpu']}` (see Limitations) |")
    lines.append("")
    for key in ("tuning_", "compile_", "fig1_"):
        for name, p in figs.items():
            if name.startswith(key):
                lines.append(f"![{name}]({Path(p).as_posix()})\n")
    lines.append("\nFull tables: [docs/RESULTS.md](docs/RESULTS.md).")
    lines.append(README_END)
    return "\n".join(lines)


def update_readme(readme: Path, block: str) -> None:
    """Replace the generated block between the markers in README.md."""
    text = readme.read_text(encoding="utf-8")
    start = text.find("<!-- RESULTS:BEGIN")
    end = text.find(README_END)
    if start == -1 or end == -1:
        raise ValueError("README markers not found")
    readme.write_text(text[:start] + block + text[end + len(README_END):], encoding="utf-8")


def write_report(results: str | Path | None = None, docs: str | Path | None = None,
                 readme: str | Path | None = "README.md") -> dict[str, str]:
    """Build the figures, write ``RESULTS.md`` and refresh the README block. Returns the figures.

    Relative paths are kept as given, so run it from the project root; the figure links in
    RESULTS.md point at ``../<results>/figures``.
    """
    results = Path(results or config.PATHS.results)
    docs = Path(docs or config.PATHS.docs)
    data = collect_results(results)
    figs = build_figures(results, data) if data["gpus"] else {}
    rel = Path("..", results.name, config.PATHS.figures).as_posix()
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "RESULTS.md").write_text(results_markdown(data, figs, rel), encoding="utf-8")
    if readme is not None and Path(readme).exists():
        update_readme(Path(readme), readme_block(data, figs))
    print(f"wrote {docs / 'RESULTS.md'}; figures: {len(figs)}")
    return figs


# =========================================================================== write-up facts
#
# scripts/make_project_doc.py fills the project document template with these values, so the
# write-up always matches the data. Every value comes back as a short formatted string.


def _f(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def _cfg_text(r: pd.Series) -> str:
    return (f"{int(r.cfg_BLOCK_SIZE_M)}×{int(r.cfg_BLOCK_SIZE_N)}×{int(r.cfg_BLOCK_SIZE_K)}, "
            f"{int(r.cfg_num_warps)} warps, {int(r.cfg_num_stages)} stages")


def ranking_quality(sweep: pd.DataFrame, arch: gpu.ArchParams, reg: model.RegModel,
                    smem: model.SmemModel) -> pd.DataFrame:
    """Per matmul shape: Spearman(score, runtime), top-1 quality and rank of the true best."""
    from scipy.stats import spearmanr

    sw = model.prepare(sweep)
    sw = sw[sw["kernel"] == "matmul"]
    rows = []
    for sid, t in sw.groupby("shape_id"):
        r0 = t.iloc[0]
        shape = Shape("matmul", sid, int(r0.M), int(r0.N), int(r0.K))
        p = model.predict_frame([model.row_config(r) for _, r in t.iterrows()], [shape], arch, reg, smem)
        order = model.rank_smart(p)
        pi, v = p.set_index("cfg_hash"), t[t["valid"]].set_index("cfg_hash")
        keep = pi.loc[v.index]
        ok = np.isfinite(keep["score"].to_numpy())
        rho = spearmanr(keep["score"][ok], v["time_ms_median"][ok]).statistic
        best_h = v["time_ms_median"].idxmin()
        top1 = v["time_ms_median"].min() / v["time_ms_median"][order[0]] if order[0] in v.index else 0.0
        rows.append({"shape_id": sid, "spearman": rho, "top1_quality": top1,
                     "rank_of_best": order.index(best_h) + 1 if best_h in order else np.nan})
    return pd.DataFrame(rows)


def collect_facts(results: str | Path, tag: str) -> dict[str, str]:
    """Every number the project write-up quotes for one GPU, as formatted strings."""
    large = config.EXPERIMENTS.large_shape_id
    R = Path(results)
    g = R / tag
    a_dir = g / "analysis"
    F: dict[str, str] = {}

    sweep = load_sweep(R, tag)
    arch = arch_for_tag(tag, None, R)
    reg, smem = load_models(R, tag)
    summ = json.loads((a_dir / "summary.json").read_text())
    noise = json.loads((g / "noise.json").read_text())
    r0 = sweep.iloc[0]

    # environment and hardware
    F.update(gpu_name=str(r0.gpu_name), gpu_tag=tag, torch=str(r0.torch_version), triton=str(r0.triton_version),
             cuda=str(r0.cuda), cc=f"sm_{arch.cc[0]}{arch.cc[1]}", sm_count=str(arch.sm_count),
             regs_per_sm=f"{arch.regs_per_sm:,}", max_regs=str(arch.max_regs_per_thread),
             smem_block_kb=_f(arch.smem_per_block_optin / 1024, 0), smem_sm_kb=_f(arch.smem_per_sm / 1024, 0),
             max_threads_sm=f"{arch.max_threads_per_sm:,}", peak_tflops=_f(arch.peak_fp16_tflops),
             bw_gbs=_f(arch.mem_bw_gbs, 0))

    # sweep
    mm = sweep[sweep["kernel"] == "matmul"]
    F.update(n_rows=f"{len(sweep):,}", n_mm_rows=f"{len(mm):,}",
             n_ok=f"{int((sweep.status == 'ok').sum()):,}",
             n_oor_rows=f"{int((sweep.status == 'out_of_resources').sum()):,}",
             n_err=str(int(sweep.status.isin(['compile_error', 'launch_error', 'incorrect']).sum())),
             n_configs=str(mm.cfg_hash.nunique()), n_mm_shapes=str(mm.shape_id.nunique()),
             n_rms_shapes=str(sweep.loc[sweep.kernel == "rmsnorm", "shape_id"].nunique()),
             noise_cv=_f(noise["cv"] * 100, 1), tie_pct=_f(noise["tie_threshold"] * 100, 1))
    cold = mm[mm["cache_miss"].astype("boolean").fillna(False).astype(bool)]
    F.update(compile_median_s=_f(cold.compile_time_s.median(), 2), compile_max_s=_f(cold.compile_time_s.max(), 0),
             bench_median_s=_f(mm.bench_time_s.median(), 2))
    sess = pd.read_json(g / "sessions.jsonl", lines=True)
    full = sess[(sess["table"] == "sweep") & (sess["n_evaluated"] > 1000)]
    if not full.empty:
        F["sweep_minutes"] = _f(full.iloc[-1].elapsed_s / 60, 0)

    # fastest vs slowest valid config on the large shape
    v = model.prepare(sweep)
    v = v[(v.shape_id == large) & v.valid].sort_values("time_ms_median")
    fast, slow = v.iloc[0], v.iloc[-1]
    F.update(fast_cfg=_cfg_text(fast), fast_ms=_f(fast.time_ms_median, 2), fast_regs=str(int(fast.n_regs)),
             fast_smem_kb=_f(fast.shared_bytes / 1024, 0),
             slow_cfg=_cfg_text(slow), slow_ms=_f(slow.time_ms_median, 0), slow_regs=str(int(slow.n_regs)),
             slow_spills=f"{int(slow.n_spills):,}", spread=_f(slow.time_ms_median / fast.time_ms_median, 0))
    feat = pd.read_csv(g / "ir_exploration" / "features.csv", index_col=0)
    F.update(slow_ptx_local=str(int(feat.loc["ptx_ld_local", "slow"]) + int(feat.loc["ptx_st_local", "slow"])),
             slow_sass_ldl=f"{int(feat.loc['sass_ldl', 'slow']):,}", slow_sass_stl=f"{int(feat.loc['sass_stl', 'slow']):,}",
             fast_hmma=str(int(feat.loc["sass_hmma", "fast"])), fast_ldgsts=str(int(feat.loc["sass_ldgsts", "fast"])),
             fast_ptx_lines=f"{int(feat.loc['ptx_lines', 'fast']):,}", slow_ptx_lines=f"{int(feat.loc['ptx_lines', 'slow']):,}")

    # predictor accuracy
    fit, ana = summ["predictor"]["fitted"], summ["predictor"]["analytic"]
    F.update(regs_err_fit=_f(fit["regs_nonspilling"]["median_abs_err"], 0),
             regs_err_ana=_f(ana["regs_nonspilling"]["median_abs_err"], 0),
             smem_exact_fit=_f(fit["smem"]["exact_match"] * 100, 0),
             spill_p=_f(fit["spill"]["precision"], 3), spill_r=_f(fit["spill"]["recall"], 2),
             ana_spill_r=_f(ana["spill"]["recall"], 2))
    tp = fp = fn = 0
    for _, r in mm.drop_duplicates("cfg_hash").iterrows():
        over = model.smem_est(model.row_config(r), smem) > arch.smem_per_block_optin
        oo = r.status == "out_of_resources"
        tp += over and oo
        fp += over and not oo
        fn += (not over) and oo
    F.update(oor_configs=str(tp + fn), oor_caught=str(tp), oor_false=str(fp))
    models = json.loads((g / "models.json").read_text())
    rm, sm_ = models["reg_model"], models["smem_model"]
    F.update(fit_a=_f(rm["a"], 2), fit_b=_f(rm["b"], 2), fit_s=_f(rm["s"], 3), fit_c=_f(rm["c"], 1),
             spill_at=_f(rm["spill_at"], 0), smem_offset=str(sm_["stage_offset"]),
             n_train_cfgs=str(models["n_train_configs"]))

    # ranking quality
    rq = ranking_quality(sweep, arch, reg, smem)
    F.update(spearman_med=_f(rq.spearman.median(), 2), spearman_min=_f(rq.spearman.min(), 2),
             spearman_max=_f(rq.spearman.max(), 2), top1_med=_f(rq.top1_quality.median() * 100, 0))

    # search replay
    t6 = pd.read_csv(a_dir / "table6.csv")

    def t6v(split: str, strat: str, col: str) -> float:
        return float(t6[(t6.kernel == "matmul") & (t6.split == split) & (t6.strategy == strat)][col].iloc[0])

    ratio = model.cost_ratio(t6).set_index(["kernel", "split"])["random_over_smart"]
    F.update(test_random_cfgs=_f(t6v("test", "random", "median_configs"), 0),
             test_smart_cfgs=_f(t6v("test", "smart", "median_configs"), 0),
             test_random_s=_f(t6v("test", "random", "median_cost_s"), 1),
             test_smart_s=_f(t6v("test", "smart", "median_cost_s"), 2),
             exhaustive_s=_f(t6v("test", "exhaustive", "median_cost_s"), 0),
             ratio_test=_f(ratio[("matmul", "test")], 1), ratio_train=_f(ratio[("matmul", "train")], 1),
             ratio_rms_test=_f(ratio[("rmsnorm", "test")], 2))
    ps = model.per_shape(pd.read_parquet(a_dir / "targets.parquet"))
    F.update(shape_wins=str(int((ps.random_over_smart > 1).sum())), shape_total=str(len(ps)),
             shape_losses=str(int((ps.random_over_smart <= 1).sum())))

    # live tuning
    integ = pd.read_parquet(g / "integration.parquet")
    big = integ[integ.shape_id == large]
    un, k8 = big[big["mode"] == "unpruned"].iloc[0], big[big["mode"] == "archtune_top8"].iloc[0]
    F.update(live_unpruned_s=_f(un.tuning_s, 0), live_top8_s=_f(k8.tuning_s, 1),
             live_speedup=_f(un.tuning_s / k8.tuning_s, 0),
             live_delta_pct=f"{(k8.time_ms_median / un.time_ms_median - 1) * 100:+.1f}",
             live_unpruned_min=_f(un.tuning_s / 60, 0))
    spot = pd.read_parquet(g / "spotcheck.parquet")
    F.update(spot_quals=", ".join(_f(q, 2) for q in spot.quality), spot_tune=f"{spot.tuning_s.min():.0f}–{spot.tuning_s.max():.0f}")

    # register caps
    mx = pd.read_parquet(g / "experiments_maxnreg.parquet")
    keys = [c for c in mx.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_maxnreg")]
    capped = mx[mx.cfg_maxnreg.notna() & (mx.cfg_maxnreg <= 128) & (mx.status == "ok")]
    ref = mx[mx.cfg_maxnreg.isna()].set_index(keys)["time_ms_median"]
    slowdowns = capped.apply(lambda r: r.time_ms_median / ref.loc[tuple(r[k] for k in keys)], axis=1)
    F.update(cap_spills=f"{int(capped.n_spills.min())}–{int(capped.n_spills.max())}",
             cap_slowdown=f"{slowdowns.min():.0f}–{slowdowns.max():.0f}")

    # pipelining depth
    st = num_stages_frame(pd.read_parquet(g / "experiments_stages.parquet"))
    med = st.groupby("num_stages")["time_rel"].median()
    F.update(stage_best=f"{int(med.idxmin())}", stage1_rel=_f(med.loc[1], 2), stage_max=str(int(med.index.max())),
             stage_max_rel=_f(med.iloc[-1], 2))

    # PTX liveness vs ptxas
    lv = pd.read_parquet(g / "ptx_liveness.parquet").dropna(subset=["n_regs"])
    ns = lv[lv.n_spills == 0]
    F.update(live_corr=_f(np.corrcoef(ns.peak_live_units, ns.n_regs)[0, 1], 3),
             live_ratio_pct=_f(((ns.peak_live_units / ns.n_regs).median() - 1) * 100, 1),
             live_n=str(len(ns)), live_spill_peak=_f(lv[lv.n_spills > 0].peak_live_units.median(), 0),
             live_instr=str(int(lv.n_instructions.median())), live_blocks=str(int(lv.n_blocks.median())))
    return F


# =========================================================================== headless notebooks
#
# Each notebook gets a fresh Jupyter kernel. The executed copy is saved even when a cell fails, so
# the traceback is in the saved notebook too, and the run stops at the first failing notebook so
# later ones don't run on missing data.

#: Execution order (01 needs the sweep from 02).
NOTEBOOK_ORDER = ("00_setup_smoke", "02_sweep", "01_ir_exploration", "03_predictor", "04_search_eval",
                  "05_compiler_experiments", "07_integration", "08_ptx_liveness")

LOCAL_BOOTSTRAP = '''# --- local bootstrap (replaces the Colab cell when run by archtune.run notebooks) ---
import os, sys
PROJECT = os.environ["ARCHTUNE_PROJECT"]
os.chdir(PROJECT); sys.path.insert(0, PROJECT)
from archtune import config, gpu
ENV = gpu.capture()
MINI = config.mini_mode()
RESULTS = os.environ.get("ARCHTUNE_RESULTS") or os.path.join(PROJECT, "results_mini" if MINI else "results")
print("MINI mode" if MINI else "FULL mode", "->", RESULTS)
print(ENV)'''


def _first_line(src: str) -> str:
    for line in src.splitlines():
        if line.strip() and not line.strip().startswith("#"):
            return line.strip()[:90]
    return src.strip()[:90]


def run_notebooks(project: str | Path, names: Sequence[str] = NOTEBOOK_ORDER, *, mini: bool,
                  out_dir: str | Path, local_bootstrap: bool = False, results_dir: str | Path | None = None,
                  timeout: int | None = None, stop_on_error: bool = True,
                  log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """Execute notebooks in order; one summary dict per notebook.

    ``local_bootstrap`` replaces each notebook's first code cell with :data:`LOCAL_BOOTSTRAP` (for
    machines without ``google.colab``); ``timeout`` is per cell (``None`` for the sweep).
    """
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError, CellTimeoutError, DeadKernelError

    project = Path(project).resolve()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ARCHTUNE_PROJECT"] = str(project)
    os.environ[config.MINI_ENV] = "1" if mini else "0"
    if results_dir is not None:
        os.environ[config.RESULTS_ENV] = str(results_dir)
    else:
        os.environ.pop(config.RESULTS_ENV, None)
    summaries: list[dict[str, Any]] = []
    for name in names:
        nb = nbformat.read(project / "notebooks" / f"{name}.ipynb", as_version=4)
        if local_bootstrap:
            next(c for c in nb.cells if c.cell_type == "code").source = LOCAL_BOOTSTRAP
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
            status, error = "kernel_died", f"{exc} (out of memory?)"
        except Exception as exc:  # noqa: BLE001
            status, error = "runner_error", f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        saved = out_dir / f"{name}.ipynb"
        nbformat.write(nb, saved)
        summaries.append({"notebook": name, "status": status, "seconds": round(elapsed, 1),
                          "failed_cell": state["cell"] if status != "ok" else None,
                          "error": (error or "")[-3000:] or None, "saved": str(saved)})
        log(f"[run] {name}: {status.upper()} in {elapsed:.0f}s -> {saved}")
        if error:
            log(error[-3000:])
        _write_notebook_summary(out_dir, summaries, mini)
        if status != "ok" and stop_on_error:
            log(f"[run] stopping: {name} failed")
            break
    return summaries


def _write_notebook_summary(out_dir: Path, summaries: list[dict[str, Any]], mini: bool) -> None:
    lines = [f"# Notebook run ({'mini' if mini else 'full'}) — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             "", "| notebook | status | seconds | failed cell |", "|---|---|---|---|"]
    lines += [f"| {s['notebook']} | **{s['status']}** | {s['seconds']} | {s['failed_cell'] if s['failed_cell'] is not None else ''} |"
              for s in summaries]
    for s in summaries:
        if s["error"]:
            lines += ["", f"## {s['notebook']}", "```", s["error"], "```"]
    (out_dir / "run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def all_ok(summaries: Sequence[dict[str, Any]], expected: Sequence[str] = NOTEBOOK_ORDER) -> bool:
    """True when every expected notebook ran and succeeded."""
    done = {s["notebook"]: s["status"] for s in summaries}
    return all(done.get(n) == "ok" for n in expected)


# =========================================================================== command line


def main(argv: list[str] | None = None) -> int:
    """``python -m archtune.run <command>``; returns the exit code."""
    ap = argparse.ArgumentParser(prog="python -m archtune.run", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", help="run the project steps in order")
    p.add_argument("--mini", action="store_true", help="tiny workload, written to results_mini/")
    p.add_argument("--only", nargs="*", default=None, choices=list(STEPS), metavar="STEP",
                   help=f"subset of: {' '.join(STEPS)}")
    p.add_argument("--no-inductor", action="store_true", help="skip the Inductor baseline")
    p.add_argument("--no-unpruned", action="store_true", help="skip the slow unpruned triton.autotune baseline")

    p = sub.add_parser("preflight", help="check every GPU code path on a tiny workload")
    p.add_argument("--out", default=config.PATHS.preflight, help="output folder (wiped first; never results/)")
    p.add_argument("--skip-inductor", action="store_true", help="skip the slow Inductor stage")

    p = sub.add_parser("demo", help="time triton.autotune on one matmul with ArchTune pruning")
    for dim in ("M", "N", "K"):
        p.add_argument(f"--{dim}", type=int, default=4096)
    p.add_argument("--k", type=int, default=8, help="configs kept after pruning")
    p.add_argument("--unpruned", action="store_true", help="also time unpruned autotuning of all configs (slow)")

    p = sub.add_parser("report", help="figures, docs/RESULTS.md and the README block from results/")
    p.add_argument("--results", default=config.PATHS.results)
    p.add_argument("--docs", default=config.PATHS.docs)
    p.add_argument("--readme", default="README.md")

    p = sub.add_parser("notebooks", help="execute the notebooks headlessly with a local bootstrap")
    p.add_argument("names", nargs="*", default=list(NOTEBOOK_ORDER))
    p.add_argument("--mini", action="store_true", help="tiny workload (results_mini/, wiped first)")
    p.add_argument("--keep-going", action="store_true", help="do not stop at the first failing notebook")

    args = ap.parse_args(argv)
    if args.command == "pipeline":
        kw = {"phase8_integration": {"inductor": not args.no_inductor, "unpruned": not args.no_unpruned}}
        return run_pipeline(PROJECT_ROOT, args.mini, args.only, kw)
    if args.command == "preflight":
        return preflight(args.out, args.skip_inductor)
    if args.command == "demo":
        demo_autotune(args.M, args.N, args.K, args.k, args.unpruned)
        return 0
    if args.command == "report":
        write_report(args.results, args.docs, args.readme)
        return 0
    if args.command == "notebooks":
        if args.mini and args.names == list(NOTEBOOK_ORDER):
            shutil.rmtree(PROJECT_ROOT / config.PATHS.results_mini, ignore_errors=True)
        out = PROJECT_ROOT / "notebooks" / ("executed_local_mini" if args.mini else "executed_local")
        res = run_notebooks(PROJECT_ROOT, args.names, mini=args.mini, out_dir=out, local_bootstrap=True,
                            stop_on_error=not args.keep_going)
        print((out / "run_summary.md").read_text(encoding="utf-8"))
        return 0 if all_ok(res, args.names) else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
