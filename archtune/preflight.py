"""Preflight: exercise every GPU code path on a tiny workload in a few minutes, before any
multi-hour run, and report PASS / WARN / FAIL per stage with the full traceback.

Run on the target runtime (Colab T4, or a local GPU)::

    python -m archtune.preflight --out results_preflight

Every stage is independent and fail-soft: a failing stage is recorded and the next one runs,
so one execution shows *all* problems. Output (also printed): ``<out>/<gpu_tag>/preflight.md``
and ``preflight.json``. Exit code 1 if any stage FAILED. Results go to a separate directory,
never into ``results/``.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

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
    """Runs stages, collecting outcomes; stages call :meth:`warn` / :meth:`check`."""

    def __init__(self, out: Path, log: Callable[[str], None] = print) -> None:
        self.out = out
        self.log = log
        self.stages: list[Stage] = []
        self.ctx: dict[str, Any] = {}
        self._cur: Stage | None = None

    def note(self, msg: str) -> None:
        """Informational note on the current stage."""
        assert self._cur is not None
        self._cur.notes.append(msg)
        self.log(f"    {msg}")

    def warn(self, msg: str) -> None:
        """Downgrade the current stage to WARN (does not stop it)."""
        assert self._cur is not None
        if self._cur.status == PASS:
            self._cur.status = WARN
        self.note(f"WARNING: {msg}")

    def check(self, cond: bool, msg: str) -> None:
        """Fail the current stage when *cond* is false."""
        if not cond:
            raise AssertionError(msg)

    def run(self, name: str, fn: Callable[["Preflight"], None], requires: tuple[str, ...] = ()) -> None:
        """Run one stage; skipped when a required context key is missing."""
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
        except Exception as exc:  # noqa: BLE001 - the whole point is to keep going
            st.status = FAIL
            st.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        st.seconds = time.perf_counter() - t0
        self.log(f"[{st.status}] {name} ({st.seconds:.1f}s)" + (f"\n{st.error}" if st.error else ""))

    @property
    def failed(self) -> bool:
        """Whether any stage failed."""
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


# --------------------------------------------------------------------------- stages


def st_env(p: Preflight) -> None:
    from archtune import env

    e = env.capture()
    p.ctx["env"] = e
    p.note(f"{e['gpu_name']} {e['gpu_tag']} torch={e['torch_version']} triton={e['triton_version']} cuda={e['cuda']}")
    p.check(e["gpu_tag"] is not None, "no CUDA GPU visible to torch")
    if not env.pinned_ok(e):
        p.warn(f"stack differs from pinned {env.PINNED}; results would not be comparable")


def st_arch(p: Preflight) -> None:
    from archtune import arch

    a = arch.live()
    p.ctx["arch"] = a
    p.note(f"{a.name} cc={a.cc} SMs={a.sm_count} smem/block={a.smem_per_block_optin} "
           f"peak={a.peak_fp16_tflops:.1f} TFLOP/s bw={a.mem_bw_gbs:.0f} GB/s")
    p.check(not math.isnan(a.peak_fp16_tflops) and not math.isnan(a.mem_bw_gbs),
            "peak throughput unknown -> predictor score would be NaN")
    key = arch.reference_for_name(a.name)
    if key:
        diffs = arch.verify(arch.reference(key), a)
        if diffs:
            p.warn(f"live vs reference table differs: {diffs}")
    else:
        p.note("no datasheet entry: peaks estimated from clocks (" +
               ", ".join(f"{k}={v}" for k, v in a.provenance.items() if "estimated" in v) + ")")


def st_api(p: Preflight) -> None:
    from archtune import apicheck

    api = apicheck.report()
    (p.out / "api_check.json").write_text(json.dumps(api, indent=2, default=str))
    p.ctx["api"] = api
    p.check(api["JITFunction.warmup"] is not None, "JITFunction.warmup missing: compile timing falls back")
    p.check(api["Autotuner.prune_configs"] is not None, "Autotuner.prune_configs missing")
    calls = api.get("prune_configs_calls") or {}
    if not all(calls.values()):
        p.warn(f"prune_configs source differs from assumption: {calls}")
    if not api["maxnreg_in_Config"]:
        p.warn("triton.Config has no maxnreg: Phase 6 register-cap sweep will be skipped")
    flush = api.get("do_bench_flushes_l2") or {}
    if not any(flush.values()):
        p.warn(f"could not confirm do_bench L2 flush: {flush}")
    cache = api.get("cache_dir_env_read_at_init")
    if isinstance(cache, dict) and not all(cache.values()):
        p.warn("TRITON_CACHE_DIR is not read in FileCacheManager.__init__: cold-compile detection may be wrong")
    p.note(f"do_bench{api['do_bench']}")


def _mm_compile(cfg: dict[str, Any], M: int = 1024, N: int = 1024, K: int = 1024):
    from archtune import compile_info, space
    from archtune.kernels import matmul as mm

    a, b, c = mm.make_inputs(M, N, K)
    meta, launch = space.split_launch(cfg)
    return compile_info.compile_config(mm.matmul_kernel, mm.kernel_args(a, b, c), mm.grid(M, N, cfg), meta, launch), (a, b, c)


def st_compile(p: Preflight) -> None:
    from archtune import apicheck, compile_info

    compile_info.fresh_cache_dir()
    cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    res, _ = _mm_compile(cfg)
    p.check(res.status == "ok", f"compile failed: {res.status} {res.error_msg}")
    attrs = apicheck.compiled_kernel_attrs(res.compiled)
    (p.out / "compiled_kernel_attrs.json").write_text(json.dumps(
        {**attrs, "describe": compile_info.describe_compiled(res.compiled)}, indent=2, default=str))
    p.note(f"compile {res.compile_time_s:.2f}s via {res.method}, cache_miss={res.cache_miss}; "
           f"n_regs={res.n_regs} n_spills={res.n_spills} shared={res.shared_bytes}; asm={attrs['asm_keys']}")
    p.check(isinstance(res.n_regs, int) and res.n_regs > 0, f"n_regs not captured: {res.n_regs!r}")
    p.check(res.n_spills is not None, "n_spills not captured")
    p.check(isinstance(res.shared_bytes, int) and res.shared_bytes > 0, f"shared_bytes not captured: {res.shared_bytes!r}")
    p.check(bool(res.asm.get("ptx")) and bool(res.asm.get("cubin")), "ptx/cubin missing from asm")
    if res.cache_miss is not True:
        p.warn("first compile not detected as a cache miss: check TRITON_CACHE_DIR handling")
    res2, _ = _mm_compile(cfg)
    p.note(f"second compile {res2.compile_time_s:.3f}s cache_miss={res2.cache_miss}")
    if res2.compile_time_s is not None and res.compile_time_s and res2.compile_time_s > 0.5 * res.compile_time_s:
        p.warn("recompiling the same config was not much faster: in-memory cache not hit?")
    p.ctx["asm"] = res.asm


def st_correct(p: Preflight) -> None:
    import torch

    from archtune import bench
    from archtune.kernels import matmul as mm
    from archtune.kernels import rmsnorm as rn

    for M, N, K in ((1024, 1024, 1024), (16, 1600, 960), (100, 70, 50)):  # last: non-multiples (masking)
        a, b, c = mm.make_inputs(M, N, K)
        mm.launcher(a, b, c, {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
                              "num_warps": 4, "num_stages": 3})()
        torch.cuda.synchronize()
        ok, err = bench.check_correct(c, mm.reference(a, b))
        p.note(f"matmul {M}x{N}x{K}: correct={ok} max_abs_err={err:.3g}")
        p.check(ok, f"matmul {M}x{N}x{K} incorrect (max_abs_err={err})")
    for bs, w in ((128, 4), (1024, 8), (256, 1)):
        x, wt, y = rn.make_inputs(64, 960)
        rn.launcher(x, wt, y, {"BLOCK_SIZE": bs, "num_warps": w})()
        torch.cuda.synchronize()
        ok, err = bench.check_correct(y, rn.reference(x, wt))
        p.note(f"rmsnorm 64x960 BLOCK={bs} warps={w}: correct={ok} err={err:.3g}")
        p.check(ok, f"rmsnorm BLOCK={bs} incorrect (err={err})")


def st_edge_configs(p: Preflight) -> None:
    a = p.ctx["arch"]
    # 256x256 tile with BLOCK_K=128 needs 128 KB even with ONE buffer, so it exceeds every target
    # GPU whether or not Triton multi-buffers (Triton 3.2 single-buffers on sm_75, D-022).
    oor = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    res, _ = _mm_compile(oor)
    p.note(f"oversized config -> status={res.status} shared={res.shared_bytes} ({(res.error_msg or '')[:80]})")
    p.check(res.status == "out_of_resources",
            f"expected out_of_resources (limit {a.smem_per_block_optin} B), got {res.status} with "
            f"shared={res.shared_bytes}: {res.error_msg}")
    for stages in (2, 5):  # what does num_stages do to shared memory on this GPU?
        cfg = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": stages}
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
    from archtune import compile_info, ptx_features

    tools = {t: compile_info.find_cuda_tool(t) for t in ("cuobjdump", "nvdisasm")}
    p.note(f"tools: {tools}")
    sass = compile_info.disassemble(p.ctx["asm"].get("cubin"))
    if not sass:
        p.warn("no SASS (cuobjdump/nvdisasm unavailable): sass_* columns will be null")
        return
    c = ptx_features.count_sass(sass)
    p.note(f"SASS counts: {c}")
    if c["hmma"] == 0:
        p.warn("no HMMA in SASS: tensor cores unused or the SASS regex does not match this format")
    pc = ptx_features.count_ptx(p.ctx["asm"]["ptx"])
    p.note(f"PTX counts: mma={pc['mma']} ldmatrix={pc['ldmatrix']} cp_async={pc['cp_async']} ld_local={pc['ld_local']}")
    if pc["mma"] == 0:
        p.warn("no mma.sync in PTX: check the PTX regex against this Triton's output")


def st_noise(p: Preflight) -> None:
    from archtune import bench
    from archtune.kernels import matmul as mm

    a, b, c = mm.make_inputs(2048, 2048, 2048)
    fn = mm.launcher(a, b, c, {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8,
                               "num_warps": 4, "num_stages": 3})
    t = bench.time_config(fn)
    p.note(f"time_config: {t}")
    p.check(t["time_ms_p10"] <= t["time_ms_median"] <= t["time_ms_p90"], "quantiles out of order")
    n = bench.noise_check(fn, n=5)
    p.note(f"noise CV over 5 reps = {n['cv']:.4f} (the real noise check uses 20); samples ms "
           f"{[round(x, 4) for x in n['samples_ms']]}, discarded first {n['discarded_first_ms']:.4f}")
    if n["cv"] > 0.05:
        p.warn("timing CV > 5%: thermal/clock noise; ties threshold will be wide")


def _mini_workload(p: Preflight):
    from archtune import space, workloads

    try:
        cfg = workloads.load_model_config(cache_dir=p.out)
        p.note("model config fetched from the Hugging Face Hub")
    except Exception as exc:  # noqa: BLE001
        fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "SmolLM2-360M.config.json"
        p.warn(f"Hub fetch failed ({exc}); using recorded fixture — notebooks need network access")
        cfg = json.loads(fixture.read_text())
    shapes = workloads.mini_shapes(workloads.all_shapes(cfg))
    return shapes, space.mini_matmul_space()


def st_sweep(p: Preflight) -> None:
    from archtune import space, sweep

    shapes, mm_cfgs = _mini_workload(p)
    tasks = sweep.make_tasks(shapes, lambda s: mm_cfgs if s.kernel == "matmul" else space.rmsnorm_space(s.N)[:6])
    e = p.ctx["env"]
    logs: list[str] = []
    s1 = sweep.run_sweep(tasks, p.out, e, limit=7, log=logs.append, flush_every=3)  # simulated interruption
    s2 = sweep.run_sweep(tasks, p.out, e, log=logs.append, flush_every=5)
    s3 = sweep.run_sweep(tasks, p.out, e, log=logs.append)
    p.note(f"{len(tasks)} tasks: session1={s1['n_evaluated']} session2={s2['n_evaluated']} "
           f"session3={s3['n_evaluated']} ({s1['elapsed_s'] + s2['elapsed_s']:.0f}s)")
    p.check(s1["n_evaluated"] == 7 and s2["n_evaluated"] == len(tasks) - 7 and s3["n_evaluated"] == 0,
            "resume did not skip already-written rows correctly")
    df = sweep.consolidate(p.out / e["gpu_tag"])
    p.check(len(df) == len(tasks), f"{len(df)} rows for {len(tasks)} tasks")
    p.check(not df.duplicated(list(sweep.KEY_COLUMNS)).any(), "duplicate rows after resume")
    counts = df.groupby(["kernel", "status"]).size().to_dict()
    p.note(f"status counts: {counts}")
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
    est = sweep.estimate_seconds(df[df["status"] == "ok"].head(10), n_tasks=18 * 384 + 4 * 20,
                                 n_cold_compiles=384 + 20)
    p.note(f"extrapolated FULL sweep on this GPU: {est['estimate_s'] / 3600:.2f} h "
           f"({est['per_cold_compile_s']:.2f}s/compile, {est['per_task_s']:.2f}s/task)")
    p.check((p.out / e["gpu_tag"] / "progress.log").exists(), "progress.log not written")
    p.ctx["sweep_done"] = True


def st_ir(p: Preflight) -> None:
    from archtune import explore

    asm = p.ctx["asm"]
    paths = explore.dump_ir(asm, p.out / p.ctx["env"]["gpu_tag"] / "ir_exploration", "preflight")
    p.check({"ttir", "ttgir", "llir", "ptx"} <= set(paths), f"IR stages missing: {sorted(paths)}")
    lay = explore.ttgir_layouts(asm.get("ttgir", ""))
    p.note(f"{len(lay)} TTGIR layout attributes")
    if not lay:
        p.warn("no #blocked/#mma layouts parsed from TTGIR (regex may not match this Triton)")


def st_analysis(p: Preflight) -> None:
    import pandas as pd

    from archtune import analysis

    tag = p.ctx["env"]["gpu_tag"]
    s = analysis.run(p.out, tag, n_seeds=3, refit=True)
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
    from archtune import plots

    plots.fig1_quality_vs_cost(curves, tag, p.out / "figures" / "fig1.png")
    plots.fig2_predictor(pm, "preflight", p.out / "figures" / "fig2.png")


def st_experiments(p: Preflight) -> None:
    from archtune import experiments, plots, sweep
    from archtune.space import mini_matmul_space

    e = p.ctx["env"]
    shapes, _ = _mini_workload(p)
    shape = [s for s in shapes if s.shape_id == "mm_M4096_N4096_K4096"]
    base = mini_matmul_space()[1:3]
    ok, why = experiments.maxnreg_supported()
    p.note(why)
    if ok:
        tasks = sweep.make_tasks(shape, lambda s: experiments.with_maxnreg(base, caps=(None, 128, 64)))
        sweep.run_sweep(tasks, p.out, e, run_id="maxnreg_v1", table="experiments_maxnreg", log=lambda m: None)
        mx = sweep.consolidate(p.out / e["gpu_tag"], "experiments_maxnreg")
        p.note(f"maxnreg rows: {mx.groupby('status').size().to_dict()}")
        p.check((mx["status"] == "ok").any(), "no maxnreg config ran")
        if "cfg_maxnreg" not in mx:
            p.warn("no cfg_maxnreg column: caps were not recorded")
        else:
            capped = mx[mx["cfg_maxnreg"].notna() & (mx["status"] == "ok")]
            p.note(f"max n_regs by cap: {capped.groupby('cfg_maxnreg')['n_regs'].max().to_dict()}")
            if (capped["n_regs"] > capped["cfg_maxnreg"]).any():
                p.warn("maxnreg was not honoured (n_regs > cap)")
        plots.fig3_maxnreg(experiments.maxnreg_frame(mx, p.ctx["arch"]), "preflight", p.out / "figures" / "fig3.png")
    else:
        p.warn("maxnreg unsupported")
    tasks = sweep.make_tasks(shape, lambda s: experiments.with_num_stages(base[:1], stages=(1, 2, 3)))
    sweep.run_sweep(tasks, p.out, e, run_id="stages_v1", table="experiments_stages", log=lambda m: None)
    st = sweep.consolidate(p.out / e["gpu_tag"], "experiments_stages")
    p.note(f"stages: {st[['cfg_num_stages', 'status', 'shared_bytes', 'ptx_cp_async']].to_dict('records')}")
    p.check((st["status"] == "ok").all(), "a num_stages config failed")
    plots.fig4_num_stages(experiments.num_stages_frame(st), "preflight", p.out / "figures" / "fig4.png")


def st_hook(p: Preflight) -> None:
    from archtune import analysis, triton_hook
    from archtune.space import mini_matmul_space

    tag = p.ctx["env"]["gpu_tag"]
    models = analysis.load_models(p.out, tag)
    kw = {"reg_model": models[0], "smem_model": models[1]} if models else {}
    cfgs = mini_matmul_space()
    r_pruned = triton_hook.tune_live(cfgs, 1024, 1024, 1024,
                                     prune_by=triton_hook.archtune_prune_configs_by(p.ctx["arch"], 3, **kw))
    p.note(f"pruned k=3: tuning {r_pruned['tuning_s']:.1f}s, best {r_pruned['best_cfg']}, "
           f"{r_pruned['time_ms_median']:.4f} ms, kept {r_pruned.get('n_after_prune')}")
    p.check(r_pruned["correct"], "pruned autotune result incorrect")
    p.check(r_pruned.get("n_after_prune") == 3, f"pruning kept {r_pruned.get('n_after_prune')} configs, expected 3")
    r_full = triton_hook.tune_live(cfgs, 1024, 1024, 1024, prune_by=None)
    p.note(f"unpruned (12 configs incl. an out_of_resources one): tuning {r_full['tuning_s']:.1f}s, "
           f"{r_full['time_ms_median']:.4f} ms")
    p.check(r_full["correct"], "unpruned autotune result incorrect")


def st_inductor(p: Preflight) -> None:
    import os

    from archtune import inductor_patch, predict

    r = inductor_patch.compile_and_time(512, 512, 512)
    p.note(f"inductor max-autotune: {r['compile_and_tune_s']:.1f}s, {r['time_ms_median']:.4f} ms, correct={r['correct']}")
    p.check(r["correct"], "inductor result incorrect")
    os.environ[inductor_patch.FLAG] = "1"
    try:
        a = p.ctx["arch"]
        ok, msg = inductor_patch.patch_mm_configs(
            lambda cfg, m, n, k: predict.predict_matmul(cfg, m, n, k, a)["feasible"])
        p.note(f"patch: {ok} ({msg})")
        if not ok:
            p.warn(f"Inductor patch unavailable on this torch: {msg} (optional feature)")
        else:
            r2 = inductor_patch.compile_and_time(512, 512, 512)
            p.note(f"patched: {r2['compile_and_tune_s']:.1f}s, correct={r2['correct']}")
    finally:
        inductor_patch.unpatch()
        os.environ.pop(inductor_patch.FLAG, None)


def st_liveness(p: Preflight) -> None:
    from archtune import analysis, compile_info
    from archtune.ptx.liveness import peak_pressure

    sw = analysis.load_sweep(p.out, p.ctx["env"]["gpu_tag"])
    rows = sw[(sw.kernel == "matmul") & sw.artifact_path.notna()].drop_duplicates("cfg_hash")
    out = []
    for r in rows.itertuples():
        ptx = compile_info.load_artifact(r.artifact_path, "ptx")
        if ptx:
            out.append((r.n_regs, peak_pressure(ptx)))
    p.check(bool(out), "no PTX artifacts could be read back")
    p.note("ptxas regs vs PTX peak-live: " + ", ".join(f"{a}/{b['peak_live_units']}" for a, b in out[:8]))
    p.check(all(b["n_instructions"] > 50 and b["n_blocks"] >= 2 for _, b in out),
            "parser found too few instructions/blocks in real Triton PTX")


def st_report(p: Preflight) -> None:
    from archtune import report

    docs = p.out / "docs"
    docs.mkdir(exist_ok=True)
    readme = p.out / "README.md"
    readme.write_text(f"# preflight\n{report.README_BEGIN}\n{report.README_END}\n", encoding="utf-8")
    report.main(["--results", str(p.out), "--docs", str(docs), "--readme", str(readme)])
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
    ("10 IR dump (notebook 01)", st_ir, ("asm",)),
    ("11 predictor + replay + figures 1-2 (notebooks 03-04)", st_analysis, ("sweep_done",)),
    ("12 maxnreg + num_stages + figures 3-4 (notebook 05)", st_experiments, ("arch",)),
    ("13 triton.autotune hook (notebook 07)", st_hook, ("arch",)),
    ("14 Inductor max-autotune (optional, notebook 07)", st_inductor, ("arch",)),
    ("15 PTX liveness on real PTX (notebook 08)", st_liveness, ("sweep_done",)),
    ("16 report generation", st_report, ("sweep_done",)),
]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the exit code."""
    ap = argparse.ArgumentParser(description="ArchTune preflight checks")
    ap.add_argument("--out", default="results_preflight", help="output dir (wiped first; never results/)")
    ap.add_argument("--skip-inductor", action="store_true", help="skip the slow Inductor stage")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.resolve().name == "results":
        raise SystemExit("refusing to write preflight data into results/")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    pf = Preflight(out)
    t0 = time.perf_counter()
    for name, fn, req in STAGES:
        if args.skip_inductor and fn is st_inductor:
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


if __name__ == "__main__":
    sys.exit(main())
