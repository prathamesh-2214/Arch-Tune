"""Every quotable number for the project documentation, computed from ``results/``.

Used by ``scripts/make_project_doc.py`` so the write-up never contains hand-typed results
(CLAUDE.md). Each value is formatted as a short string ready to drop into prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LARGE = "mm_M4096_N4096_K4096"


def _f(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}"


def _cfg_label(r: pd.Series) -> str:
    return (f"{int(r.cfg_BLOCK_SIZE_M)}×{int(r.cfg_BLOCK_SIZE_N)}×{int(r.cfg_BLOCK_SIZE_K)}, "
            f"{int(r.cfg_num_warps)} warps, {int(r.cfg_num_stages)} stages")


def ranking_quality(sweep: pd.DataFrame, arch: Any, reg: Any, smem: Any) -> pd.DataFrame:
    """Per matmul shape: Spearman(score, runtime), top-1 quality, rank of the true best."""
    from scipy.stats import spearmanr

    from archtune import predict, search

    sw = search.prepare(sweep)
    sw = sw[sw["kernel"] == "matmul"]
    rows = []
    for sid, t in sw.groupby("shape_id"):
        r0 = t.iloc[0]
        shape = type("S", (), dict(kernel="matmul", shape_id=sid, M=int(r0.M), N=int(r0.N), K=int(r0.K)))()
        cfgs = [{k: v for k, v in predict.cfg_from_row(r).items() if v is not None} for _, r in t.iterrows()]
        p = predict.predict_frame(cfgs, [shape], arch, reg, smem)
        order = predict.rank_smart(p)
        pi, v = p.set_index("cfg_hash"), t[t["valid"]].set_index("cfg_hash")
        keep = pi.loc[v.index]
        ok = np.isfinite(keep["score"].to_numpy())
        rho = spearmanr(keep["score"][ok], v["time_ms_median"][ok]).statistic
        best_h = v["time_ms_median"].idxmin()
        top1 = v["time_ms_median"].min() / v["time_ms_median"][order[0]] if order[0] in v.index else 0.0
        rows.append({"shape_id": sid, "spearman": rho, "top1_quality": top1,
                     "rank_of_best": order.index(best_h) + 1 if best_h in order else np.nan})
    return pd.DataFrame(rows)


def collect(results: str | Path, tag: str) -> dict[str, str]:
    """All facts for one GPU's results as formatted strings."""
    from archtune import analysis, predict, search
    from archtune.search import per_shape

    R = Path(results)
    g = R / tag
    a_dir = g / "analysis"
    F: dict[str, str] = {}

    sweep = analysis.load_sweep(R, tag)
    arch = analysis.arch_for_tag(tag, None, R)
    reg, smem = analysis.load_models(R, tag)
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
    v = search.prepare(sweep)
    v = v[(v.shape_id == LARGE) & v.valid].sort_values("time_ms_median")
    fast, slow = v.iloc[0], v.iloc[-1]
    F.update(fast_cfg=_cfg_label(fast), fast_ms=_f(fast.time_ms_median, 2), fast_regs=str(int(fast.n_regs)),
             fast_smem_kb=_f(fast.shared_bytes / 1024, 0),
             slow_cfg=_cfg_label(slow), slow_ms=_f(slow.time_ms_median, 0), slow_regs=str(int(slow.n_regs)),
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
        c = {k: x for k, x in predict.cfg_from_row(r).items() if x is not None}
        pi = predict.smem_est(c, smem) > arch.smem_per_block_optin
        oo = r.status == "out_of_resources"
        tp += pi and oo
        fp += pi and not oo
        fn += (not pi) and oo
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

    ratio = search.cost_ratio(t6).set_index(["kernel", "split"])["random_over_smart"]
    F.update(test_random_cfgs=_f(t6v("test", "random", "median_configs"), 0),
             test_smart_cfgs=_f(t6v("test", "smart", "median_configs"), 0),
             test_random_s=_f(t6v("test", "random", "median_cost_s"), 1),
             test_smart_s=_f(t6v("test", "smart", "median_cost_s"), 2),
             exhaustive_s=_f(t6v("test", "exhaustive", "median_cost_s"), 0),
             ratio_test=_f(ratio[("matmul", "test")], 1), ratio_train=_f(ratio[("matmul", "train")], 1),
             ratio_rms_test=_f(ratio[("rmsnorm", "test")], 2))
    ps = per_shape(pd.read_parquet(a_dir / "targets.parquet"))
    F.update(shape_wins=str(int((ps.random_over_smart > 1).sum())), shape_total=str(len(ps)),
             shape_losses=str(int((ps.random_over_smart <= 1).sum())))

    # live tuning
    integ = pd.read_parquet(g / "integration.parquet")
    big = integ[integ.shape_id == LARGE]
    un, k8 = big[big["mode"] == "unpruned"].iloc[0], big[big["mode"] == "archtune_top8"].iloc[0]
    F.update(live_unpruned_s=_f(un.tuning_s, 0), live_top8_s=_f(k8.tuning_s, 1),
             live_speedup=_f(un.tuning_s / k8.tuning_s, 0),
             live_delta_pct=f"{(k8.time_ms_median / un.time_ms_median - 1) * 100:+.1f}",
             live_unpruned_min=_f(un.tuning_s / 60, 0))
    spot = pd.read_parquet(g / "spotcheck.parquet")
    F.update(spot_quals=", ".join(_f(q, 2) for q in spot.quality), spot_tune=f"{spot.tuning_s.min():.0f}–{spot.tuning_s.max():.0f}")

    # register caps
    mx = pd.read_parquet(g / "experiments_maxnreg.parquet")
    capped = mx[mx.cfg_maxnreg.notna() & (mx.cfg_maxnreg <= 128) & (mx.status == "ok")]
    ref = mx[mx.cfg_maxnreg.isna()].set_index(
        [c for c in mx.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_maxnreg")])["time_ms_median"]
    keys = [c for c in mx.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_maxnreg")]
    slowdowns = capped.apply(lambda r: r.time_ms_median / ref.loc[tuple(r[k] for k in keys)], axis=1)
    F.update(cap_spills=f"{int(capped.n_spills.min())}–{int(capped.n_spills.max())}",
             cap_slowdown=f"{slowdowns.min():.0f}–{slowdowns.max():.0f}")

    # pipelining depth
    from archtune.experiments import num_stages_frame

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
