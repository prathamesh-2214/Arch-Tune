"""CPU-only analysis pipeline (Phases 4-5): from saved sweep rows to predictor metrics,
replay curves and the cost-to-target table. Writes everything under
``<results_dir>/<gpu_tag>/analysis/`` so README/RESULTS numbers are generated, never typed.

Predictor constants are fit on the **training-split shapes only** (spec §5) of the GPU being
analyzed, saved to ``<results_dir>/<gpu_tag>/models.json`` and reused unchanged for held-out shapes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from archtune import arch as arch_mod
from archtune import predict, search
from archtune.experiments import check_single_stack
from archtune.sweep import DEFAULT_RUN_ID, consolidate
from archtune.workloads import Shape, train_test_split

def load_sweep(results_dir: str | Path, gpu_tag: str, run_id: str = DEFAULT_RUN_ID,
               table: str = "sweep") -> pd.DataFrame:
    """Consolidated sweep rows for one GPU and run id (consolidating parts if present)."""
    gpu_dir = Path(results_dir) / gpu_tag
    df = consolidate(gpu_dir, table) if (gpu_dir / table).exists() else pd.read_parquet(gpu_dir / f"{table}.parquet")
    return df[df["run_id"] == run_id].reset_index(drop=True)


def arch_for_tag(gpu_tag: str, sm_count: int | None = None,
                 results_dir: str | Path | None = None) -> arch_mod.ArchParams:
    """Architecture parameters for a GPU tag.

    Prefers ``<results_dir>/<gpu_tag>/arch.json`` (written by the sweep from the live device,
    including clock-based peak estimates for GPUs without a datasheet entry); otherwise the
    reference table with the SM count observed in the rows.
    """
    if results_dir is not None and (Path(results_dir) / gpu_tag / "arch.json").exists():
        return arch_mod.ArchParams.from_dict(json.loads((Path(results_dir) / gpu_tag / "arch.json").read_text()))
    prefix, sm = gpu_tag.split("_sm")
    if prefix in arch_mod.GPU_REFERENCE:
        a = arch_mod.reference(prefix)
    else:
        a = arch_mod.from_cc((int(sm[0]), int(sm[1:])), sm_count or 1, name=gpu_tag)
    if sm_count and sm_count != a.sm_count:
        from dataclasses import replace

        a = replace(a, sm_count=int(sm_count))
    return a


def shapes_of(df: pd.DataFrame) -> list[Shape]:
    """Distinct shapes in a sweep frame."""
    u = df[["kernel", "shape_id", "M", "N", "K"]].drop_duplicates("shape_id")
    return [Shape(r.kernel, r.shape_id, int(r.M), int(r.N), int(r.K), ()) for r in u.itertuples()]


def fit_models(sweep: pd.DataFrame, split: dict[str, str]) -> dict[str, Any]:
    """Fit register and shared-memory models on matmul rows of training shapes."""
    d = sweep[(sweep["kernel"] == "matmul") & (sweep["shape_id"].map(split) == "train")]
    per_cfg = predict.configs_from_sweep(d)
    reg = predict.fit_regs(per_cfg)
    smem, rate = predict.fit_smem(per_cfg)
    return {"reg_model": predict.model_to_dict(reg), "smem_model": predict.model_to_dict(smem),
            "smem_train_exact_match": rate, "n_train_configs": int(len(per_cfg)),
            "train_shapes": sorted(k for k, v in split.items() if v == "train")}


def load_models(results_dir: str | Path, gpu_tag: str) -> tuple[predict.RegModel, predict.SmemModel] | None:
    """Fitted models from ``<results_dir>/<gpu_tag>/models.json`` (``None`` if not fitted yet)."""
    p = Path(results_dir) / gpu_tag / "models.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    sm = dict(m["smem_model"])
    return predict.RegModel(**m["reg_model"]), predict.SmemModel(**sm)


def predictor_table(sweep: pd.DataFrame, a: arch_mod.ArchParams, reg: predict.RegModel,
                    smem: predict.SmemModel) -> pd.DataFrame:
    """Per-config predicted vs measured registers / spills / shared memory (matmul)."""
    per_cfg = predict.configs_from_sweep(sweep[sweep["kernel"] == "matmul"])
    rows = []
    for _, r in per_cfg.iterrows():
        cfg = predict.cfg_from_row(r)
        cfg = {k: v for k, v in cfg.items() if v is not None}
        rows.append({
            "cfg_hash": r["cfg_hash"], **{k: v for k, v in r.items() if k.startswith("cfg_")},
            "n_regs": r.get("n_regs"), "n_spills": r.get("n_spills"), "shared_bytes": r.get("shared_bytes"),
            "regs_pred": predict.regs_est(cfg, a, reg, cfg.get("maxnreg")),
            "regs_pred_analytic": predict.regs_est(cfg, a, predict.ANALYTIC_REGS, cfg.get("maxnreg")),
            "spill_pred": predict.spill_pred(cfg, a, reg, cfg.get("maxnreg")),
            "spill_pred_analytic": predict.spill_pred(cfg, a, predict.ANALYTIC_REGS, cfg.get("maxnreg")),
            "smem_pred": predict.smem_est(cfg, smem),
            "smem_pred_analytic": predict.smem_est(cfg, predict.ANALYTIC_SMEM),
        })
    return pd.DataFrame(rows)


def predictor_metrics(pm: pd.DataFrame) -> dict[str, Any]:
    """RQ1 metrics for fitted and analytic variants."""
    d = pm.dropna(subset=["n_regs"])
    nospill = d[d["n_spills"].fillna(0) == 0]
    out: dict[str, Any] = {}
    for variant, sfx in (("fitted", ""), ("analytic", "_analytic")):
        out[variant] = {
            "regs_nonspilling": predict.regression_report(nospill[f"regs_pred{sfx}"], nospill["n_regs"]),
            "smem": predict.regression_report(pm[f"smem_pred{sfx}"], pm["shared_bytes"]),
            "spill": predict.spill_confusion(d[f"spill_pred{sfx}"], d["n_spills"].fillna(0) > 0),
        }
    return out


def top_mispredictions(pm: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The *n* configs with the largest register misprediction (fitted model)."""
    d = pm.dropna(subset=["n_regs"]).copy()
    d["regs_abs_err"] = (d["regs_pred"] - d["n_regs"]).abs()
    d["explanation"] = "unexplained"  # filled in by hand in notebook 03 after PTX/SASS diffing
    return d.sort_values("regs_abs_err", ascending=False).head(n)


def run(results_dir: str | Path, gpu_tag: str, run_id: str = DEFAULT_RUN_ID,
        n_seeds: int = search.N_SEEDS, refit: bool = False) -> dict[str, Any]:
    """Full Phase 4-5 analysis for one GPU; returns a summary dict (also written to JSON)."""
    results_dir = Path(results_dir)
    out_dir = results_dir / gpu_tag / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep = load_sweep(results_dir, gpu_tag, run_id)
    stack = check_single_stack(sweep)
    mm_ids = sorted(sweep.loc[sweep["kernel"] == "matmul", "shape_id"].unique())
    rms_ids = sorted(sweep.loc[sweep["kernel"] == "rmsnorm", "shape_id"].unique())
    split = {**train_test_split(mm_ids), **train_test_split(rms_ids)}

    models_path = results_dir / gpu_tag / "models.json"
    if refit or not models_path.exists():
        models_path.write_text(json.dumps({**fit_models(sweep, split), "stack_id": stack}, indent=2))
    models = load_models(results_dir, gpu_tag)
    reg, smem = models

    sm_count = int(sweep["sm_count"].dropna().iloc[0]) if sweep["sm_count"].notna().any() else None
    a = arch_for_tag(gpu_tag, sm_count, results_dir)

    pm = predictor_table(sweep, a, reg, smem)
    pm.to_parquet(out_dir / "predictor.parquet", index=False)
    metrics = predictor_metrics(pm)
    top_mispredictions(pm).to_csv(out_dir / "top_mispredictions.csv", index=False)

    prepared = search.prepare(sweep)
    curves_all, targets_all = [], []
    for variant, (rm, smm) in {"smart": (reg, smem),
                               "smart_analytic": (predict.ANALYTIC_REGS, predict.ANALYTIC_SMEM)}.items():
        orders, ptime = {}, {}
        for shape in shapes_of(sweep):
            cfgs = [{k: v for k, v in predict.cfg_from_row(r).items() if v is not None}
                    for _, r in prepared[prepared["shape_id"] == shape.shape_id].drop_duplicates("cfg_hash").iterrows()]
            t0 = time.perf_counter()
            pred = predict.predict_frame(cfgs, [shape], a, rm, smm)
            order = predict.rank_smart(pred)
            key = (gpu_tag, shape.kernel, shape.shape_id)
            ptime[key] = time.perf_counter() - t0
            orders[key] = order
        c, t = search.replay(prepared, orders, split=split, predictor_time_s=ptime, n_seeds=n_seeds)
        if variant != "smart":
            c, t = c[c["strategy"] == "smart"], t[t["strategy"] == "smart"]
        c = c.assign(strategy=c["strategy"].replace({"smart": variant}))
        t = t.assign(strategy=t["strategy"].replace({"smart": variant}))
        curves_all.append(c)
        targets_all.append(t)
    curves = pd.concat(curves_all, ignore_index=True)
    targets = pd.concat(targets_all, ignore_index=True)
    curves.to_parquet(out_dir / "curves.parquet", index=False)
    targets.to_parquet(out_dir / "targets.parquet", index=False)
    table6 = search.summarize_targets(targets)
    table6.to_csv(out_dir / "table6.csv", index=False)

    status_counts = sweep.groupby(["kernel", "status"]).size().rename("n").reset_index()
    summary = {"gpu_tag": gpu_tag, "run_id": run_id, "stack_id": stack, "n_rows": int(len(sweep)),
               "status_counts": status_counts.to_dict("records"), "split": split,
               "predictor": metrics, "models": json.loads(models_path.read_text()),
               "cost_ratio_random_over_smart": search.cost_ratio(table6).to_dict("records")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary
