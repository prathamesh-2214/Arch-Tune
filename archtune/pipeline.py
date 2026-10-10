"""One function per project phase, so a single Colab notebook (``ArchTune_Colab.ipynb``) can run
the whole project in one kernel with live output.

Every phase reads what earlier phases saved under ``results/`` (or ``results_mini/``), so any
phase can be re-run on its own after a disconnect. Long phases are resumable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from archtune import env as env_mod
from archtune import space, sweep, workloads

LARGE = "mm_M4096_N4096_K4096"


@dataclass
class Run:
    """Where and how this run writes."""

    project: Path
    results: Path
    env: dict[str, Any]
    mini: bool

    @property
    def gpu_dir(self) -> Path:
        """``results/<gpu_tag>``."""
        return self.results / self.env["gpu_tag"]

    @property
    def figures(self) -> Path:
        """``results/figures``."""
        return self.results / "figures"


def setup(project: str | Path, mini: bool) -> Run:
    """Create the run context (``results_mini/`` in mini mode)."""
    project = Path(project)
    e = env_mod.capture()
    if e["gpu_tag"] is None:
        raise RuntimeError("No GPU visible: Runtime > Change runtime type > T4 GPU")
    run = Run(project, project / ("results_mini" if mini else "results"), e, mini)
    run.gpu_dir.mkdir(parents=True, exist_ok=True)
    run.figures.mkdir(parents=True, exist_ok=True)
    print(f"{'MINI' if mini else 'FULL'} run on {e['gpu_name']} ({e['gpu_tag']}), "
          f"torch {e['torch_version']}, triton {e['triton_version']} -> {run.results}")
    return run


def shapes(run: Run) -> tuple[list[workloads.Shape], list[workloads.Shape]]:
    """Matmul and RMSNorm shapes (subset in mini mode)."""
    cfg = workloads.load_model_config(cache_dir=run.results)
    dims = workloads.derive_dims(cfg)
    mm, rms = workloads.matmul_shapes(dims), workloads.rmsnorm_shapes(dims)
    if run.mini:
        sub = workloads.mini_shapes(mm + rms)
        mm = [s for s in sub if s.kernel == "matmul"]
        rms = [s for s in sub if s.kernel == "rmsnorm"]
    return mm, rms


def matmul_configs(run: Run) -> list[dict[str, Any]]:
    """384 configs (12 in mini mode)."""
    return space.mini_matmul_space() if run.mini else space.matmul_space()


# --------------------------------------------------------------------------- phase 0


def phase0_smoke(run: Run) -> dict[str, Any]:
    """Architecture check, API check, one compile with inspection, SASS, noise check."""
    from archtune import apicheck, arch, bench, compile_info
    from archtune.kernels import matmul as mm

    a = arch.live()
    key = arch.reference_for_name(a.name)
    diffs = arch.verify(arch.reference(key), a) if key else "no reference entry (peaks estimated from clocks)"
    print("arch:", a.name, a.cc, f"SMs={a.sm_count}", f"smem/block={a.smem_per_block_optin}",
          f"peak={a.peak_fp16_tflops:.0f} TFLOP/s", f"bw={a.mem_bw_gbs:.0f} GB/s")
    print("live vs reference:", diffs)
    (run.gpu_dir / "arch.json").write_text(json.dumps(a.to_dict(), indent=2, default=str))
    env_mod.pip_freeze(run.gpu_dir / "pip_freeze.txt")
    (run.gpu_dir / "device_properties.json").write_text(json.dumps(env_mod.device_properties(), indent=2, default=str))

    api = apicheck.report()
    (run.gpu_dir / "api_check.json").write_text(json.dumps(api, indent=2, default=str))
    for k in ("JITFunction.warmup", "maxnreg_in_Config", "Autotuner.prune_configs", "do_bench",
              "do_bench_flushes_l2", "cache_dir_env_read_at_init"):
        print(f"  {k:28s} {api.get(k)}")

    compile_info.fresh_cache_dir()
    cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    M = N = K = 1024
    x, y, z = mm.make_inputs(M, N, K)
    meta, launch = space.split_launch(cfg)
    res = compile_info.compile_config(mm.matmul_kernel, mm.kernel_args(x, y, z), mm.grid(M, N, cfg), meta, launch)
    if res.status != "ok":
        raise RuntimeError(f"smoke compile failed: {res.status} {res.error_msg}")
    attrs = apicheck.compiled_kernel_attrs(res.compiled)
    (run.gpu_dir / "compiled_kernel_attrs.json").write_text(json.dumps(
        {**attrs, "describe": compile_info.describe_compiled(res.compiled)}, indent=2, default=str))
    mm.launcher(x, y, z, cfg)()
    import torch

    torch.cuda.synchronize()
    ok, err = bench.check_correct(z, mm.reference(x, y))
    print(f"smoke config: compile {res.compile_time_s:.2f}s, regs={res.n_regs}, spills={res.n_spills}, "
          f"smem={res.shared_bytes}, correct={ok} (max err {err:.3g}); asm keys {attrs['asm_keys']}")
    if not ok or res.n_regs is None or res.shared_bytes is None:
        raise RuntimeError("smoke test did not capture registers/shared memory or was incorrect")
    sass = compile_info.disassemble(res.asm.get("cubin"))
    print("SASS:", "available" if sass else "unavailable (PTX-only analysis)")

    x, y, z = mm.make_inputs(4096, 4096, 4096)
    big = {**cfg, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}
    noise = bench.noise_check(mm.launcher(x, y, z, big), n=5 if run.mini else 20,
                              warmup=bench.SWEEP_WARMUP_MS, rep=bench.SWEEP_REP_MS)
    noise.update(shape="4096^3", cfg=big, bench_ms=[bench.SWEEP_WARMUP_MS, bench.SWEEP_REP_MS],
                 gpu_tag=run.env["gpu_tag"], stack_id=run.env["stack_id"])
    (run.gpu_dir / "noise.json").write_text(json.dumps(noise, indent=2))
    print(f"noise: CV={noise['cv']:.4f} -> differences < {noise['tie_threshold']:.2%} are ties")
    return {"api": api, "noise_cv": noise["cv"], "smoke_correct": ok}


# --------------------------------------------------------------------------- phase 2


def phase2_sweep(run: Run, run_id: str = sweep.DEFAULT_RUN_ID) -> pd.DataFrame:
    """Small-first (10 configs) with health checks and an estimate, then the full resumable sweep."""
    mm, rms = shapes(run)
    cfgs = matmul_configs(run)
    rms_cfgs = (lambda s: space.rmsnorm_space(s.N)[:6]) if run.mini else (lambda s: space.rmsnorm_space(s.N))
    tasks = sweep.make_tasks(mm, lambda s: cfgs) + sweep.make_tasks(rms, rms_cfgs)
    print(f"{len(tasks)} tasks ({len(mm)} matmul shapes x {len(cfgs)} configs + RMSNorm)")

    done = sweep.load_parts(run.gpu_dir / "sweep")
    if done.empty:  # small first, only on a brand-new sweep
        small = [t for t in tasks if t.shape.shape_id == mm[-1].shape_id][:10]
        sweep.run_sweep(small, run.results, run.env, run_id=run_id)
        done = sweep.load_parts(run.gpu_dir / "sweep")
        sample = done[done.cfg_hash.isin([t.cfg_hash for t in small])]
        bad = sample[sample.status.isin(["compile_error", "launch_error", "incorrect"])]
        if not run.mini and not bad.empty:
            raise RuntimeError(f"small-first: {len(bad)} failing configs, fix before the full sweep: "
                               f"{bad.error_msg.head(3).tolist()}")
        remaining = sweep.pending(tasks, sweep.keys_of(done), run.env["gpu_tag"], run_id)
        est = sweep.estimate_seconds(sample[sample.status == "ok"], len(remaining),
                                     len({(t.shape.kernel, t.cfg_hash) for t in remaining}))
        print(f"ESTIMATE for the rest: {est['estimate_s'] / 60:.0f} min "
              f"({est['per_cold_compile_s']:.1f}s per compile, {est['per_task_s']:.2f}s per task)")
        estimate = est["estimate_s"]
    else:
        print(f"resuming: {len(done)} rows already on Drive")
        estimate = None
    sweep.run_sweep(tasks, run.results, run.env, run_id=run_id, estimate_s=estimate,
                    progress_every=100)
    df = sweep.consolidate(run.gpu_dir)
    df = df[df.run_id == run_id]
    missing = sweep.pending(tasks, sweep.keys_of(df), run.env["gpu_tag"], run_id)
    counts = df.groupby(["kernel", "status"]).size().unstack(fill_value=0)
    print(counts)
    broken = df.status.isin(["compile_error", "launch_error"]).mean()
    if missing or broken >= 0.02:
        raise RuntimeError(f"sweep unhealthy: {len(missing)} missing rows, {broken:.1%} compile/launch errors: "
                           f"{df.loc[df.status.isin(['compile_error', 'launch_error']), 'error_msg'].head(3).tolist()}")
    return counts


# --------------------------------------------------------------------------- phase 3


def phase3_ir(run: Run) -> pd.DataFrame:
    """Dump IR of the fastest and slowest valid config on the large shape; feature table."""
    from archtune import analysis, compile_info, explore, search
    from archtune.experiments import cfg_of
    from archtune.kernels import matmul as mm

    sw = search.prepare(analysis.load_sweep(run.results, run.env["gpu_tag"]))
    t = sw[(sw.shape_id == LARGE) & sw.valid].sort_values("time_ms_median")
    picks = {"fast": t.iloc[0], "slow": t.iloc[-1]}
    shape = next(s for s in shapes(run)[0] if s.shape_id == LARGE)
    x, y, z = mm.make_inputs(shape.M, shape.N, shape.K)
    listings = {}
    for name, row in picks.items():
        cfg = cfg_of(row)
        meta, launch = space.split_launch(cfg)
        r = compile_info.compile_config(mm.matmul_kernel, mm.kernel_args(x, y, z), mm.grid(shape.M, shape.N, cfg), meta, launch)
        sass = compile_info.disassemble(r.asm.get("cubin")) if r.asm.get("cubin") else None
        explore.dump_ir(r.asm, run.gpu_dir / "ir_exploration", name, sass)
        listings[name] = {"ptx": r.asm.get("ptx"), "sass": sass}
        print(f"{name}: {row.time_ms_median:.3f} ms, cfg {cfg}, regs {r.n_regs}, spills {r.n_spills}, smem {r.shared_bytes}")
        for pat in (r"mma\.sync", r"ldmatrix", r"cp\.async\.c[ag]", r"(ld|st)\.local"):
            print(f"  {pat}: {explore.snippets(r.asm.get('ptx', ''), pat, context=0, max_hits=1)}")
    feats = explore.compare_features(listings)
    feats.to_csv(run.gpu_dir / "ir_exploration" / "features.csv")
    return feats


# --------------------------------------------------------------------------- phases 4-5


def phase4_5_analysis(run: Run) -> dict[str, Any]:
    """Fit the predictor (training shapes), replay search strategies, figures 1-2, table 6."""
    from archtune import analysis, plots, search

    tag = run.env["gpu_tag"]
    summary = analysis.run(run.results, tag, refit=True, n_seeds=5 if run.mini else search.N_SEEDS)
    a_dir = run.gpu_dir / "analysis"
    plots.fig2_predictor(pd.read_parquet(a_dir / "predictor.parquet"),
                         f"Predicted vs measured — {tag} (fitted on train split)", run.figures / f"fig2_predictor_{tag}.png")
    plots.fig1_quality_vs_cost(pd.read_parquet(a_dir / "curves.parquet"), tag, run.figures / f"fig1_quality_vs_cost_{tag}.png")
    for variant, m in summary["predictor"].items():
        print(f"{variant:9s} regs median|err|={m['regs_nonspilling']['median_abs_err']:.3g}  "
              f"smem exact={m['smem']['exact_match']:.2f}  spill P/R={m['spill']['precision']:.2f}/{m['spill']['recall']:.2f}")
    t6 = pd.read_csv(a_dir / "table6.csv")
    print(t6.to_string(index=False))
    print(search.cost_ratio(t6).to_string(index=False))
    return summary


def phase5_spotcheck(run: Run, n_shapes: int = 3, k: int = 8) -> pd.DataFrame:
    """Live smart-k (triton.autotune + ArchTune pruning) on held-out shapes vs the sweep's best."""
    from archtune import analysis, arch, search, triton_hook

    tag = run.env["gpu_tag"]
    reg, smem = analysis.load_models(run.results, tag)
    summ = json.loads((run.gpu_dir / "analysis" / "summary.json").read_text())
    sw = search.prepare(analysis.load_sweep(run.results, tag))
    test_ids = [s for s, v in summ["split"].items() if v == "test" and s.startswith("mm_")][:n_shapes]
    a = arch.live()
    rows = []
    for sid in test_ids:
        r0 = sw[sw.shape_id == sid].iloc[0]
        live = triton_hook.tune_live(matmul_configs(run), int(r0.M), int(r0.N), int(r0.K),
                                     prune_by=triton_hook.archtune_prune_configs_by(a, k, reg, smem))
        best = search.best_time(sw[sw.shape_id == sid])
        rows.append({"shape_id": sid, "tuning_s": live["tuning_s"], "time_ms": live["time_ms_median"],
                     "sweep_best_ms": best, "quality": best / live["time_ms_median"], "correct": live["correct"]})
        print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_parquet(run.gpu_dir / "spotcheck.parquet")
    return out


# --------------------------------------------------------------------------- phase 6


def phase6_experiments(run: Run) -> dict[str, pd.DataFrame]:
    """maxnreg sweep on 3 representative configs; num_stages sweep on the 5 fastest configs of 2 shapes."""
    from archtune import analysis, arch, experiments, plots, search

    tag = run.env["gpu_tag"]
    sw = analysis.load_sweep(run.results, tag)
    mm, _ = shapes(run)
    by_id = {s.shape_id: s for s in mm}
    out: dict[str, pd.DataFrame] = {}
    ok, why = experiments.maxnreg_supported()
    print(why)
    if ok:
        reps = experiments.representative_configs(sw, LARGE)
        tasks = sweep.make_tasks([by_id[LARGE]], lambda s: experiments.with_maxnreg(reps))
        sweep.run_sweep(tasks, run.results, run.env, run_id="maxnreg_v1", table="experiments_maxnreg")
        mx = sweep.consolidate(run.gpu_dir, "experiments_maxnreg")
        out["maxnreg"] = experiments.maxnreg_frame(mx, arch.live())
        plots.fig3_maxnreg(out["maxnreg"], f"Register cap sweep — {tag}, {LARGE}", run.figures / f"fig3_maxnreg_{tag}.png")
        print(out["maxnreg"].to_string(index=False))
    v = search.prepare(sw)
    v = v[v.valid & (v.kernel == "matmul")]
    tasks = []
    for sid in dict.fromkeys([LARGE, sorted(v.shape_id.unique())[0]]):
        for _, r in v[v.shape_id == sid].nsmallest(5, "time_ms_median").iterrows():
            tasks += sweep.make_tasks([by_id[sid]], lambda s, c=experiments.cfg_of(r): experiments.with_num_stages([c]))
    sweep.run_sweep(tasks, run.results, run.env, run_id="stages_v1", table="experiments_stages")
    st = sweep.consolidate(run.gpu_dir, "experiments_stages")
    out["stages"] = experiments.num_stages_frame(st)
    plots.fig4_num_stages(out["stages"], f"Pipelining depth — {tag}", run.figures / f"fig4_num_stages_{tag}.png")
    print(out["stages"].groupby("num_stages")[["shared_bytes", "ptx_cp_async", "n_regs", "time_rel"]].median())
    return out


# --------------------------------------------------------------------------- phase 8


def phase8_integration(run: Run, ks: tuple[int, ...] = (8, 16), unpruned: bool = True,
                       inductor: bool = True) -> pd.DataFrame:
    """Live triton.autotune with ArchTune pruning on 3 shapes; unpruned baseline on the large
    shape only (it compiles and benchmarks every config); optional Inductor baseline."""
    from archtune import analysis, arch, triton_hook

    tag = run.env["gpu_tag"]
    models = analysis.load_models(run.results, tag)
    kw = {"reg_model": models[0], "smem_model": models[1]} if models else {}
    a = arch.live()
    mm, _ = shapes(run)
    targets = [s for s in mm if s.shape_id in (LARGE, "mm_M2048_N2560_K960", "mm_M128_N1600_K960")]
    cfgs = matmul_configs(run)
    rows = []
    for s in targets:
        for k in ks:
            r = triton_hook.tune_live(cfgs, s.M, s.N, s.K, prune_by=triton_hook.archtune_prune_configs_by(a, k, **kw))
            rows.append({"mode": f"archtune_top{k}", "shape_id": s.shape_id, **r})
            print(f"{rows[-1]['mode']:16s} {s.shape_id:22s} tune {r['tuning_s']:7.1f}s  {r['time_ms_median']:.4f} ms")
    if unpruned:
        s = next(s for s in targets if s.shape_id == LARGE)
        r = triton_hook.tune_live(cfgs, s.M, s.N, s.K, prune_by=None)
        rows.append({"mode": "unpruned", "shape_id": s.shape_id, **r})
        print(f"{'unpruned':16s} {s.shape_id:22s} tune {r['tuning_s']:7.1f}s  {r['time_ms_median']:.4f} ms")
    out = pd.DataFrame(rows)
    out["best_cfg"] = out["best_cfg"].astype(str)
    out.to_parquet(run.gpu_dir / "integration.parquet")
    if inductor:
        from archtune import inductor_patch

        ind = []
        for s in targets:
            try:
                ind.append({"mode": "inductor", "shape_id": s.shape_id, **inductor_patch.compile_and_time(s.M, s.N, s.K)})
                print(f"inductor {s.shape_id}: {ind[-1]['compile_and_tune_s']:.1f}s, {ind[-1]['time_ms_median']:.4f} ms")
            except Exception as exc:  # noqa: BLE001 - optional baseline
                print(f"inductor {s.shape_id} failed (optional): {exc}")
        if ind:
            pd.DataFrame(ind).to_parquet(run.gpu_dir / "inductor.parquet")
    return out


# --------------------------------------------------------------------------- phase 9 + report


def phase9_liveness(run: Run) -> pd.DataFrame:
    """PTX liveness peak vs ptxas registers for every swept matmul config (stretch)."""
    from archtune import analysis, compile_info, plots
    from archtune.ptx.liveness import peak_pressure

    tag = run.env["gpu_tag"]
    sw = analysis.load_sweep(run.results, tag)
    rows = []
    for r in sw[(sw.kernel == "matmul") & sw.artifact_path.notna()].drop_duplicates("cfg_hash").itertuples():
        ptx = compile_info.load_artifact(r.artifact_path, "ptx")
        if ptx:
            rows.append({"cfg_hash": r.cfg_hash, "n_regs": r.n_regs, "n_spills": r.n_spills, **peak_pressure(ptx)})
    lv = pd.DataFrame(rows)
    lv.to_parquet(run.gpu_dir / "ptx_liveness.parquet")
    plots.fig_liveness(lv, f"PTX liveness vs ptxas — {tag}", run.figures / f"fig_liveness_{tag}.png")
    print(lv[["n_regs", "peak_live_units"]].describe().round(1))
    return lv


def make_report(run: Run) -> str:
    """Regenerate RESULTS.md / README block (full runs) or a mini report (mini runs)."""
    from archtune import report

    if run.mini:
        docs = run.results / "docs"
        docs.mkdir(exist_ok=True)
        report.main(["--results", str(run.results), "--docs", str(docs), "--readme", str(run.results / "none.md")])
        return (docs / "RESULTS.md").read_text(encoding="utf-8")
    cwd = os.getcwd()
    try:
        os.chdir(run.project)
        report.main([])
    finally:
        os.chdir(cwd)
    return (run.project / "docs" / "RESULTS.md").read_text(encoding="utf-8")


def timed(label: str, fn, *args, **kwargs):
    """Run a phase with a start/end banner and elapsed time."""
    print(f"===== {label} =====")
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"===== {label}: done in {(time.perf_counter() - t0) / 60:.1f} min =====")
    return out
