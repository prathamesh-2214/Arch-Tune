"""Analytical pre-compilation predictor (spec §9).

Pure functions of (config, shape, :class:`~archtune.arch.ArchParams`): shared memory,
registers, spills, occupancy, wave efficiency and a roofline score. Two variants are kept
and reported side by side:

* **analytic** — the formulas below with default constants (no data used);
* **fitted** — the same functional forms with constants fitted on the *training split only*
  (:func:`fit_regs`, :func:`fit_smem`).

Every intermediate term is returned by :func:`predict_matmul` so it can be plotted.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from archtune.arch import ArchParams
from archtune.space import config_hash

ELEM_BYTES = 2  # fp16 operands
ACC_BYTES = 4  # fp32 accumulator
#: Occupancy below this many resident warps per SM is penalized in the score.
MIN_ACTIVE_WARPS = 4
LOW_OCCUPANCY_PENALTY = 1.5


def _ceil_to(x: float, unit: int) -> int:
    return int(math.ceil(x / unit) * unit)


def _floor_to(x: float, unit: int) -> int:
    return int(math.floor(x / unit) * unit)


# ----------------------------------------------------------------------------- models


@dataclass(frozen=True)
class RegModel:
    """``regs ≈ a·acc + b·operand + s·operand·(num_stages-1) + c``, clamped to the max.

    ``acc`` = fp32 accumulator registers per thread = BM·BN/threads; ``operand`` = 32-bit
    registers per thread holding one K-step of fp16 A and B tiles = (BM+BN)·BK/(2·threads).
    """

    a: float = 1.0
    b: float = 0.0
    s: float = 0.0
    c: float = 32.0
    name: str = "analytic"
    #: Spill threshold on the *unclamped* estimate, in registers at the default 255 cap
    #: (``None`` = the cap itself, the analytic rule). ptxas squeezes many large tiles to
    #: 252-254 registers without spilling, so the fitted model learns a threshold above 255.
    spill_at: float | None = None


@dataclass(frozen=True)
class SmemModel:
    """``smem = combine(mainloop, epilogue)`` with
    ``mainloop = stages_eff·(BM·BK + BK·BN)·2`` and ``epilogue = BM·BN·epi_bytes``.

    ``stages_eff = min(num_stages + stage_offset, stage_cap)``; ``combine`` is ``max``
    (buffers reused) or ``sum``.
    """

    stage_offset: int = 0
    stage_cap: int | None = None
    epi_bytes: int = 0
    combine: str = "max"
    name: str = "analytic"


ANALYTIC_REGS = RegModel()
ANALYTIC_SMEM = SmemModel()


# ----------------------------------------------------------------------------- terms


def threads(cfg: dict[str, Any]) -> int:
    """Threads per block."""
    return int(cfg["num_warps"]) * 32


def reg_features(cfg: dict[str, Any]) -> dict[str, float]:
    """Register-model features: accumulator, operand and pipelined-operand terms."""
    t = threads(cfg)
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    acc = bm * bn / t * (ACC_BYTES / 4)
    operand = (bm + bn) * bk * ELEM_BYTES / 4 / t
    return {"acc": acc, "operand": operand, "operand_stages": operand * (cfg["num_stages"] - 1)}


def regs_raw(cfg: dict[str, Any], model: RegModel = ANALYTIC_REGS) -> float:
    """Unclamped register estimate (> max means the allocator must spill)."""
    f = reg_features(cfg)
    return model.a * f["acc"] + model.b * f["operand"] + model.s * f["operand_stages"] + model.c


def regs_est(cfg: dict[str, Any], arch: ArchParams, model: RegModel = ANALYTIC_REGS,
             maxnreg: int | None = None) -> int:
    """Predicted registers per thread, clamped at ``maxnreg`` or the architectural max."""
    cap = min(maxnreg or arch.max_regs_per_thread, arch.max_regs_per_thread)
    return int(min(math.ceil(regs_raw(cfg, model)), cap))


def spill_pred(cfg: dict[str, Any], arch: ArchParams, model: RegModel = ANALYTIC_REGS,
               maxnreg: int | None = None) -> bool:
    """Predict register spilling: unclamped estimate exceeds the spill threshold.

    The threshold is the register cap (analytic) or the fitted ``spill_at`` scaled to the cap
    in force (``maxnreg`` or the architectural maximum).
    """
    cap = min(maxnreg or arch.max_regs_per_thread, arch.max_regs_per_thread)
    threshold = cap if model.spill_at is None else model.spill_at * cap / arch.max_regs_per_thread
    return regs_raw(cfg, model) > threshold


def smem_est(cfg: dict[str, Any], model: SmemModel = ANALYTIC_SMEM) -> int:
    """Predicted shared memory per block in bytes (matmul)."""
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    stages = max(cfg["num_stages"] + model.stage_offset, 1)
    if model.stage_cap is not None:
        stages = min(stages, model.stage_cap)
    mainloop = stages * (bm * bk + bk * bn) * ELEM_BYTES
    epilogue = bm * bn * model.epi_bytes
    return int(mainloop + epilogue if model.combine == "sum" else max(mainloop, epilogue))


def occupancy(regs_per_thread: int, smem_bytes: int, num_warps: int, arch: ArchParams) -> dict[str, Any]:
    """Resident blocks per SM and the limiting resource.

    Register rule (cuda_occupancy.h, sm_70+): registers are allocated per warp in units of
    ``reg_alloc_unit``; the register file is split across 4 SM sub-partitions, so warps that
    fit = 4 · floor((regs_per_sm/4) / regs_per_warp). Shared memory is allocated in units of
    ``smem_alloc_unit`` plus ``smem_reserved_per_block`` (1 KB on sm_80+).
    """
    regs_per_warp = _ceil_to(max(regs_per_thread, 1) * 32, arch.reg_alloc_unit)
    warps_by_regs = 4 * ((arch.regs_per_sm // 4) // regs_per_warp)
    by_regs = warps_by_regs // num_warps
    if smem_bytes > arch.smem_per_block_optin:
        by_smem = 0
    elif smem_bytes <= 0:
        by_smem = arch.max_blocks_per_sm
    else:
        per_block = _ceil_to(smem_bytes, arch.smem_alloc_unit) + arch.smem_reserved_per_block
        by_smem = arch.smem_per_sm // per_block
    by_threads = arch.max_threads_per_sm // (num_warps * 32)
    limits = {"regs": by_regs, "smem": by_smem, "threads": by_threads, "blocks": arch.max_blocks_per_sm}
    blocks = min(limits.values())
    limiter = min(limits, key=lambda k: (limits[k], ["smem", "regs", "threads", "blocks"].index(k)))
    active_warps = blocks * num_warps
    return {"blocks_per_sm": blocks, "limiter": limiter, "blocks_by_regs": by_regs,
            "blocks_by_smem": by_smem, "blocks_by_threads": by_threads,
            "active_warps": active_warps, "occupancy": active_warps / arch.max_warps_per_sm}


def waves(n_tiles: int, blocks_per_sm: int, sm_count: int) -> dict[str, float]:
    """Wave quantization: ``efficiency = tiles / (ceil(tiles/slots)·slots)``."""
    slots = blocks_per_sm * sm_count
    if slots <= 0:
        return {"tiles": n_tiles, "slots": 0, "n_waves": float("inf"), "wave_efficiency": 0.0}
    n_waves = math.ceil(n_tiles / slots)
    return {"tiles": n_tiles, "slots": slots, "n_waves": n_waves,
            "wave_efficiency": n_tiles / (n_waves * slots)}


def roofline_matmul(cfg: dict[str, Any], M: int, N: int, K: int, arch: ArchParams) -> dict[str, float]:
    """Roofline time (s) on *padded* tiles with tile-level DRAM traffic.

    * FLOPs count the full tiles the kernel computes: ``2·ceil(M/BM)BM·ceil(N/BN)BN·ceil(K/BK)BK``
      (a 256-row tile on a 16-token decode matmul does 16× the useful work).
    * Bytes count each program streaming its A row-panel and B column-panel, with no L2 reuse:
      ``2·(M·K·ceil(N/BN) + K·N·ceil(M/BM) + M·N)`` — larger tiles move fewer bytes.
    """
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    tm, tn, tk = math.ceil(M / bm), math.ceil(N / bn), math.ceil(K / bk)
    flops = 2.0 * (tm * bm) * (tn * bn) * (tk * bk)
    nbytes = ELEM_BYTES * (M * K * tn + K * N * tm + M * N)
    t_c = flops / (arch.peak_fp16_tflops * 1e12)
    t_m = nbytes / (arch.mem_bw_gbs * 1e9)
    return {"flops_padded": flops, "bytes_tiles": nbytes, "t_compute_s": t_c, "t_memory_s": t_m,
            "t_roofline_s": max(t_c, t_m), "bound": "compute" if t_c >= t_m else "memory",
            "tiles": tm * tn}


def predict_matmul(cfg: dict[str, Any], M: int, N: int, K: int, arch: ArchParams,
                   reg_model: RegModel = ANALYTIC_REGS, smem_model: SmemModel = ANALYTIC_SMEM
                   ) -> dict[str, Any]:
    """All predictor terms for one matmul config and shape. ``score`` is lower-is-better."""
    maxnreg = cfg.get("maxnreg")
    regs = regs_est(cfg, arch, reg_model, maxnreg)
    raw = regs_raw(cfg, reg_model)
    smem = smem_est(cfg, smem_model)
    smem_ok = smem <= arch.smem_per_block_optin
    spill = spill_pred(cfg, arch, reg_model, maxnreg)
    occ = occupancy(regs, smem, cfg["num_warps"], arch)
    roof = roofline_matmul(cfg, M, N, K, arch)
    wv = waves(roof["tiles"], occ["blocks_per_sm"], arch.sm_count)
    penalty = LOW_OCCUPANCY_PENALTY if occ["active_warps"] < MIN_ACTIVE_WARPS else 1.0
    feasible = smem_ok and occ["blocks_per_sm"] > 0
    score = roof["t_roofline_s"] / wv["wave_efficiency"] * penalty if feasible else float("inf")
    return {"regs_pred": regs, "regs_raw": raw, "spill_pred": spill, "smem_pred": smem,
            "smem_feasible": smem_ok, "feasible": feasible, **occ, **roof,
            "slots": wv["slots"], "n_waves": wv["n_waves"], "wave_efficiency": wv["wave_efficiency"],
            "occ_penalty": penalty, "score": score}


def predict_rmsnorm(cfg: dict[str, Any], rows: int, cols: int, arch: ArchParams) -> dict[str, Any]:
    """Simple RMSNorm score: memory roofline / wave efficiency, times the fraction of idle
    lanes when ``BLOCK_SIZE`` < threads and the number of sequential chunks per row."""
    t = threads(cfg)
    bs = cfg["BLOCK_SIZE"]
    occ = occupancy(32, 0, cfg["num_warps"], arch)
    wv = waves(rows, occ["blocks_per_sm"], arch.sm_count)
    t_m = ELEM_BYTES * (2 * rows * cols + cols) / (arch.mem_bw_gbs * 1e9)
    lane_waste = max(1.0, t / bs)
    chunks = math.ceil(cols / bs)
    score = t_m / wv["wave_efficiency"] * lane_waste * (1.0 + 0.05 * (chunks - 1))
    return {"regs_pred": None, "spill_pred": False, "smem_pred": 0, "feasible": True,
            **occ, "t_memory_s": t_m, "lane_waste": lane_waste, "chunks": chunks,
            "wave_efficiency": wv["wave_efficiency"], "score": score}


def predict_frame(configs: Iterable[dict[str, Any]], shapes: Iterable[Any], arch: ArchParams,
                  reg_model: RegModel = ANALYTIC_REGS, smem_model: SmemModel = ANALYTIC_SMEM
                  ) -> pd.DataFrame:
    """Predictions for every (shape, config); one row each, joinable on (shape_id, cfg_hash)."""
    configs = list(configs)
    rows = []
    for s in shapes:
        for cfg in configs:
            if s.kernel == "matmul":
                p = predict_matmul(cfg, s.M, s.N, s.K, arch, reg_model, smem_model)
            else:
                p = predict_rmsnorm(cfg, s.M, s.N, arch)
            rows.append({"kernel": s.kernel, "shape_id": s.shape_id, "cfg_hash": config_hash(cfg),
                         "model": reg_model.name, **p})
    return pd.DataFrame(rows)


def rank_smart(pred: pd.DataFrame, k: int | None = None) -> list[str]:
    """Smart-k policy (spec §9): drop predicted-infeasible and predicted-spilling configs,
    rank by score (ties broken by cfg_hash), return the top *k* cfg hashes.

    If the filter removes *every* config, fall back to ranking all of them by score so the
    policy always returns something to benchmark.
    """
    keep = pred[pred["feasible"].astype(bool) & ~pred["spill_pred"].astype(bool)]
    if keep.empty:
        keep = pred
    ranked = keep.sort_values(["score", "cfg_hash"], kind="mergesort")["cfg_hash"].tolist()
    return ranked if k is None else ranked[:k]


# ----------------------------------------------------------------------------- fitting


def cfg_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Rebuild a config dict from ``cfg_*`` columns of a sweep row."""
    return {k[4:]: (int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None)
            for k, v in dict(row).items() if k.startswith("cfg_") and k != "cfg_hash"}


def configs_from_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct config in a sweep frame (cfg_hash + knob columns + measurements)."""
    cols = ["cfg_hash"] + [c for c in df.columns if c.startswith("cfg_") and c != "cfg_hash"]
    meas = [c for c in ("n_regs", "n_spills", "shared_bytes") if c in df.columns]
    out = df[cols + meas]
    if meas:
        out = out.dropna(subset=[meas[0]])
    return out.drop_duplicates("cfg_hash").reset_index(drop=True)


def fit_regs(df: pd.DataFrame, max_regs: int = 255) -> RegModel:
    """Fit :class:`RegModel` coefficients by non-negative least squares on rows that did not
    spill and did not hit the cap (training split only — the caller filters)."""
    from scipy.optimize import nnls

    d = df[(df["n_spills"].fillna(0) == 0) & (df["n_regs"] < max_regs)]
    if d.empty:
        return ANALYTIC_REGS
    feats = [reg_features(cfg_from_row(r)) for _, r in d.iterrows()]
    X = np.array([[f["acc"], f["operand"], f["operand_stages"], 1.0] for f in feats])
    y = d["n_regs"].to_numpy(dtype=float)
    coef, _ = nnls(X, y)
    model = RegModel(a=float(coef[0]), b=float(coef[1]), s=float(coef[2]), c=float(coef[3]), name="fitted")
    measured = df.dropna(subset=["n_regs"])
    raw = np.array([regs_raw(cfg_from_row(r), model) for _, r in measured.iterrows()])
    spilled = measured["n_spills"].fillna(0).to_numpy() > 0
    return replace(model, spill_at=fit_spill_threshold(raw, spilled, floor=max_regs))


def fit_spill_threshold(raw: np.ndarray, spilled: np.ndarray, floor: float = 255.0,
                        fp_weight: float = 2.0) -> float:
    """Spill threshold on the unclamped register estimate minimizing ``fp_weight·FP + FN``.

    A false positive (predicting a spill for a config that does not spill) removes that config
    from smart-k entirely, possibly the fastest one, while a false negative only wastes one
    compile, so false positives are weighted more. Ties go to the larger threshold. The
    threshold never goes below the register cap (*floor*).
    """
    cands = np.unique(np.concatenate([[floor], raw[raw >= floor]]))
    best_t, best_cost = float(floor), float("inf")
    for t in cands:
        pred = raw > t
        cost = fp_weight * np.sum(pred & ~spilled) + np.sum(~pred & spilled)
        if cost < best_cost or (cost == best_cost and t > best_t):
            best_t, best_cost = float(t), float(cost)
    return best_t


def fit_smem(df: pd.DataFrame) -> tuple[SmemModel, float]:
    """Grid-search :class:`SmemModel` variants for the highest exact-match rate on training rows.

    Returns (best model, its exact-match rate). Variants: stage offset ∈ {-1, 0, 1},
    stage cap ∈ {None, 1, 2}, epilogue bytes/element ∈ {0, 2, 4}, combine ∈ {max, sum}.
    """
    d = df.dropna(subset=["shared_bytes"])
    cfgs = [cfg_from_row(r) for _, r in d.iterrows()]
    meas = d["shared_bytes"].to_numpy()
    best, best_rate = ANALYTIC_SMEM, -1.0
    for off in (-1, 0, 1):
        for cap in (None, 1, 2):
            for epi in (0, 2, 4):
                for comb in ("max", "sum"):
                    m = SmemModel(off, cap, epi, comb, name="fitted")
                    pred = np.array([smem_est(c, m) for c in cfgs])
                    rate = float(np.mean(pred == meas)) if len(meas) else 0.0
                    if rate > best_rate:
                        best, best_rate = m, rate
    return best, best_rate


# ----------------------------------------------------------------------------- evaluation


def regression_report(pred: Sequence[float], meas: Sequence[float]) -> dict[str, float]:
    """Median / mean absolute error, exact-match rate and Pearson r."""
    p, m = np.asarray(pred, dtype=float), np.asarray(meas, dtype=float)
    ok = ~(np.isnan(p) | np.isnan(m))
    p, m = p[ok], m[ok]
    if p.size == 0:
        return {"n": 0, "median_abs_err": float("nan"), "mean_abs_err": float("nan"),
                "exact_match": float("nan"), "pearson_r": float("nan")}
    err = np.abs(p - m)
    r = float(np.corrcoef(p, m)[0, 1]) if p.size > 1 and p.std() > 0 and m.std() > 0 else float("nan")
    return {"n": int(p.size), "median_abs_err": float(np.median(err)), "mean_abs_err": float(err.mean()),
            "exact_match": float(np.mean(err == 0)), "pearson_r": r}


def spill_confusion(pred: Sequence[bool], meas: Sequence[bool]) -> dict[str, float]:
    """Confusion counts plus precision / recall of spill prediction."""
    p, m = np.asarray(pred, dtype=bool), np.asarray(meas, dtype=bool)
    tp, fp = int((p & m).sum()), int((p & ~m).sum())
    fn, tn = int((~p & m).sum()), int((~p & ~m).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def model_to_dict(model: RegModel | SmemModel) -> dict[str, Any]:
    """Serialize a fitted model for JSON."""
    return asdict(model)
