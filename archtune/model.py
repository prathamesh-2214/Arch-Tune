"""The config space, the workloads, the predictor and how it plugs into autotuning.

* **config space**: knob grids, stable config hashes, conversion to and from ``triton.Config``;
* **workloads**: matmul and RMSNorm shapes taken from SmolLM2-360M, and the train/test split;
* **predictor**: shared memory, registers, spills, occupancy, waves and a roofline score for a
  (config, shape, GPU) without compiling anything; constants either analytic or fitted;
* **search replay**: exhaustive, random-k and smart-k replayed over a finished sweep;
* **hooks**: ``prune_configs_by`` for ``triton.autotune``, and an optional filter on Inductor's
  ``max-autotune`` matmul configs.

Nothing here needs a GPU except :func:`tune_live` and :func:`inductor_compile_and_time`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from archtune import config, gpu
from archtune.gpu import ArchParams, split_launch

Config = dict[str, int]

# =========================================================================== config space
#
# A config is a plain dict of knob -> int. Launch options (num_warps, num_stages, maxnreg) sit in
# the same dict as the tl.constexpr tile sizes; split_launch separates them.


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable 12-hex-digit hash: SHA-1 of the sorted, compact JSON, with ``None`` values dropped."""
    clean = {k: v for k, v in cfg.items() if v is not None}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _product(knobs: Mapping[str, Iterable[int]]) -> list[Config]:
    names = list(knobs)
    return [dict(zip(names, vals)) for vals in itertools.product(*knobs.values())]


def matmul_space(knobs: Mapping[str, tuple[int, ...]] | None = None) -> list[Config]:
    """Every combination of the matmul knobs (384 with the defaults).

    Every load is masked, so no divisibility rule removes configs, and resource limits are
    deliberately not used for filtering here: that is what the sweep measures.
    """
    return _product(knobs or config.SPACE.matmul_knobs)


def next_pow2(x: int) -> int:
    """Smallest power of two >= ``x``."""
    return 1 << (int(x) - 1).bit_length()


def rmsnorm_space(cols: int, min_block: int | None = None,
                  warps: tuple[int, ...] | None = None) -> list[Config]:
    """``BLOCK_SIZE`` from ``min_block`` up to ``next_pow2(cols)`` (the single-pass block), times warps."""
    min_block = config.SPACE.rmsnorm_min_block if min_block is None else min_block
    warps = config.SPACE.rmsnorm_warps if warps is None else warps
    top = next_pow2(cols)
    blocks, b = [], min(min_block, top)
    while b <= top:
        blocks.append(b)
        b *= 2
    return [{"BLOCK_SIZE": bs, "num_warps": w} for bs in blocks for w in warps]


def mini_matmul_space() -> list[Config]:
    """The twelve quick-run configs from ``config.SPACE.mini_matmul``."""
    return [{"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 8,
             "num_warps": w, "num_stages": s} for m, n, k, w, s in config.SPACE.mini_matmul]


def to_triton_config(cfg: dict[str, Any]) -> Any:
    """Convert to ``triton.Config``."""
    import triton

    meta, launch = split_launch(cfg)
    return triton.Config(meta, **launch)


def from_triton_config(tcfg: Any) -> Config:
    """Convert a ``triton.Config`` (or anything with ``.kwargs`` and launch attributes) to a dict."""
    cfg = dict(tcfg.kwargs)
    for knob in config.SPACE.launch_knobs:
        val = getattr(tcfg, knob, None)
        if val is not None:
            cfg[knob] = int(val)
    return cfg


def cfg_from_row(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """Config from the ``cfg_*`` columns of a sweep row (missing knobs come back as ``None``)."""
    return {k[4:]: (int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else None)
            for k, v in dict(row).items() if k.startswith("cfg_") and k != "cfg_hash"}


def row_config(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    """:func:`cfg_from_row` without the missing knobs: a config that can be compiled."""
    return {k: v for k, v in cfg_from_row(row).items() if v is not None}


# =========================================================================== workloads
#
# Matmuls follow C[M, N] = A[M, K] @ B[K, N]. The model dimensions come from the model's Hugging
# Face config.json rather than being typed in.


@dataclass(frozen=True)
class ModelDims:
    """Transformer dimensions the projection shapes are derived from."""

    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Shape:
    """One workload shape. For RMSNorm, ``M`` = rows (tokens), ``N`` = cols, ``K`` = 0."""

    kernel: str
    shape_id: str
    M: int
    N: int
    K: int
    roles: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for result rows."""
        d = asdict(self)
        d["roles"] = ",".join(self.roles)
        return d


def load_model_config(model_id: str | None = None, cache_dir: str | os.PathLike[str] | None = None,
                      timeout: float = 30.0) -> dict[str, Any]:
    """Fetch ``config.json`` from the Hugging Face Hub, cached as JSON in *cache_dir*."""
    model_id = model_id or config.WORKLOAD.model_id
    cache = Path(cache_dir) / f"{model_id.replace('/', '__')}.config.json" if cache_dir else None
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        cfg = json.loads(resp.read().decode())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def derive_dims(cfg: dict[str, Any]) -> ModelDims:
    """:class:`ModelDims` from a Llama-style Hugging Face config."""
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    return ModelDims(hidden=hidden, intermediate=int(cfg["intermediate_size"]),
                     heads=heads, kv_heads=kv_heads, head_dim=head_dim)


def projection_nk(dims: ModelDims) -> dict[str, tuple[int, int]]:
    """(N, K) of each linear layer. Q, K and V are one fused matmul; MLP up and gate share a shape."""
    qkv_n = (dims.heads + 2 * dims.kv_heads) * dims.head_dim
    return {"qkv": (qkv_n, dims.hidden),
            "attn_out": (dims.hidden, dims.heads * dims.head_dim),
            "mlp_up_gate": (dims.intermediate, dims.hidden),
            "mlp_down": (dims.hidden, dims.intermediate)}


def shape_id(kernel: str, M: int, N: int, K: int) -> str:
    """Canonical id, e.g. ``mm_M128_N1600_K960`` or ``rms_R128_C960``."""
    return f"mm_M{M}_N{N}_K{K}" if kernel == "matmul" else f"rms_R{M}_C{N}"


def matmul_shapes(dims: ModelDims, tokens: tuple[int, ...] | None = None,
                  large: tuple[tuple[int, int, int], ...] | None = None) -> list[Shape]:
    """Distinct matmul shapes: token counts x projections, plus the large reference shapes."""
    tokens = config.WORKLOAD.token_counts if tokens is None else tokens
    large = config.WORKLOAD.large_shapes if large is None else large
    roles: dict[tuple[int, int, int], list[str]] = {}
    for m in tokens:
        for role, (n, k) in projection_nk(dims).items():
            roles.setdefault((m, n, k), []).append(f"{role}@{m}")
    for mnk in large:
        roles.setdefault(mnk, []).append("large_ref")
    return [Shape("matmul", shape_id("matmul", *mnk), *mnk, roles=tuple(r))
            for mnk, r in sorted(roles.items())]


def rmsnorm_shapes(dims: ModelDims, tokens: tuple[int, ...] | None = None) -> list[Shape]:
    """RMSNorm over (tokens x hidden) for each token count."""
    tokens = config.WORKLOAD.token_counts if tokens is None else tokens
    return [Shape("rmsnorm", shape_id("rmsnorm", m, dims.hidden, 0), m, dims.hidden, 0,
                  roles=(f"rmsnorm@{m}",)) for m in tokens]


def train_test_split(shape_ids: list[str], seed: int | None = None,
                     train_frac: float | None = None) -> dict[str, str]:
    """Seeded split that does not depend on input order: shape_id -> ``"train"`` / ``"test"``."""
    seed = config.WORKLOAD.split_seed if seed is None else seed
    train_frac = config.WORKLOAD.train_frac if train_frac is None else train_frac
    ids = sorted(set(shape_ids))
    random.Random(seed).shuffle(ids)
    n_train = int(round(len(ids) * train_frac))
    return {sid: ("train" if i < n_train else "test") for i, sid in enumerate(ids)}


def all_shapes(model_cfg: dict[str, Any]) -> list[Shape]:
    """Matmul + RMSNorm shapes for a model config."""
    dims = derive_dims(model_cfg)
    return matmul_shapes(dims) + rmsnorm_shapes(dims)


def mini_shapes(shapes: list[Shape]) -> list[Shape]:
    """The quick-run subset of *shapes*."""
    w = config.WORKLOAD
    mm = [s for s in shapes if s.kernel == "matmul" and s.shape_id in w.mini_matmul_shape_ids]
    rms = sorted((s for s in shapes if s.kernel == "rmsnorm"), key=lambda s: s.M)[:w.mini_rmsnorm_shapes]
    return mm + rms


# =========================================================================== predictor
#
# Arithmetic on (config, shape, GPU parameters). The same formulas come with analytic default
# constants or with constants fitted on the training shapes (fit_regs, fit_smem). predict_matmul
# returns every intermediate term, so each one can be checked against what ptxas did.


def _ceil_to(x: float, unit: int) -> int:
    return int(math.ceil(x / unit) * unit)


@dataclass(frozen=True)
class RegModel:
    """``regs = a*acc + b*operand + s*operand*(num_stages-1) + c``, clamped to the register cap.

    ``acc`` = fp32 accumulator registers per thread = BM*BN/threads; ``operand`` = 32-bit registers
    per thread holding one K-step of the fp16 A and B tiles = (BM+BN)*BK/(2*threads).
    """

    a: float = 1.0
    b: float = 0.0
    s: float = 0.0
    c: float = 32.0
    name: str = "analytic"
    #: Spill threshold on the unclamped estimate at the 255 cap (``None`` = the cap itself).
    #: ptxas squeezes many large tiles into 252-254 registers without spilling, so the fitted
    #: threshold ends up above 255.
    spill_at: float | None = None


@dataclass(frozen=True)
class SmemModel:
    """``smem = combine(mainloop, epilogue)``, ``mainloop = stages_eff*(BM*BK + BK*BN)*2``,
    ``epilogue = BM*BN*epi_bytes`` and ``stages_eff = min(num_stages + stage_offset, stage_cap)``.
    ``combine`` is ``max`` (buffers reused) or ``sum``."""

    stage_offset: int = 0
    stage_cap: int | None = None
    epi_bytes: int = 0
    combine: str = "max"
    name: str = "analytic"


#: The unfitted models: one register per accumulator element plus 32 (``config.PREDICTOR``), and
#: one buffer per pipeline stage.
ANALYTIC_REGS = RegModel(*config.PREDICTOR.analytic_regs)
ANALYTIC_SMEM = SmemModel()


def threads(cfg: dict[str, Any]) -> int:
    """Threads per block."""
    return int(cfg["num_warps"]) * 32


def reg_features(cfg: dict[str, Any]) -> dict[str, float]:
    """The accumulator, operand and pipelined-operand terms of the register model."""
    p = config.PREDICTOR
    t = threads(cfg)
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    acc = bm * bn / t * (p.acc_bytes / 4)
    operand = (bm + bn) * bk * p.elem_bytes / 4 / t
    return {"acc": acc, "operand": operand, "operand_stages": operand * (cfg["num_stages"] - 1)}


def regs_raw(cfg: dict[str, Any], model: RegModel = ANALYTIC_REGS) -> float:
    """Unclamped register estimate (above the cap means ptxas has to spill)."""
    f = reg_features(cfg)
    return model.a * f["acc"] + model.b * f["operand"] + model.s * f["operand_stages"] + model.c


def _reg_cap(arch: ArchParams, maxnreg: int | None) -> int:
    return min(maxnreg or arch.max_regs_per_thread, arch.max_regs_per_thread)


def regs_est(cfg: dict[str, Any], arch: ArchParams, model: RegModel = ANALYTIC_REGS,
             maxnreg: int | None = None) -> int:
    """Predicted registers per thread, clamped at ``maxnreg`` or the architectural maximum."""
    return int(min(math.ceil(regs_raw(cfg, model)), _reg_cap(arch, maxnreg)))


def spill_pred(cfg: dict[str, Any], arch: ArchParams, model: RegModel = ANALYTIC_REGS,
               maxnreg: int | None = None) -> bool:
    """Whether ptxas will spill: the unclamped estimate is above the threshold.

    The threshold is the cap (analytic) or the fitted ``spill_at`` scaled to the cap in force.
    """
    cap = _reg_cap(arch, maxnreg)
    threshold = cap if model.spill_at is None else model.spill_at * cap / arch.max_regs_per_thread
    return regs_raw(cfg, model) > threshold


def smem_est(cfg: dict[str, Any], model: SmemModel = ANALYTIC_SMEM) -> int:
    """Predicted shared memory per block, in bytes (matmul)."""
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    stages = max(cfg["num_stages"] + model.stage_offset, 1)
    if model.stage_cap is not None:
        stages = min(stages, model.stage_cap)
    mainloop = stages * (bm * bk + bk * bn) * config.PREDICTOR.elem_bytes
    epilogue = bm * bn * model.epi_bytes
    return int(mainloop + epilogue if model.combine == "sum" else max(mainloop, epilogue))


def occupancy(regs_per_thread: int, smem_bytes: int, num_warps: int, arch: ArchParams) -> dict[str, Any]:
    """Resident blocks per SM and which resource limits them.

    Registers (cuda_occupancy.h, sm_70+) are allocated per warp in units of ``reg_alloc_unit``,
    and the register file is split over 4 sub-partitions, so warps that fit =
    4 * floor((regs_per_sm/4) / regs_per_warp). Shared memory is allocated in units of
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
    """Wave quantization: ``efficiency = tiles / (ceil(tiles/slots) * slots)``."""
    slots = blocks_per_sm * sm_count
    if slots <= 0:
        return {"tiles": n_tiles, "slots": 0, "n_waves": float("inf"), "wave_efficiency": 0.0}
    n_waves = math.ceil(n_tiles / slots)
    return {"tiles": n_tiles, "slots": slots, "n_waves": n_waves,
            "wave_efficiency": n_tiles / (n_waves * slots)}


def roofline_matmul(cfg: dict[str, Any], M: int, N: int, K: int, arch: ArchParams) -> dict[str, float]:
    """Roofline time (s) on padded tiles, with DRAM traffic counted per tile.

    * FLOPs count the full tiles computed, ``2*ceil(M/BM)BM*ceil(N/BN)BN*ceil(K/BK)BK`` (a 256-row
      tile on a 16-token matmul does 16x the useful work).
    * Bytes count every program streaming its A row-panel and B column-panel with no L2 reuse,
      ``2*(M*K*ceil(N/BN) + K*N*ceil(M/BM) + M*N)``, so larger tiles move fewer bytes.
    """
    bm, bn, bk = cfg["BLOCK_SIZE_M"], cfg["BLOCK_SIZE_N"], cfg["BLOCK_SIZE_K"]
    tm, tn, tk = math.ceil(M / bm), math.ceil(N / bn), math.ceil(K / bk)
    flops = 2.0 * (tm * bm) * (tn * bn) * (tk * bk)
    nbytes = config.PREDICTOR.elem_bytes * (M * K * tn + K * N * tm + M * N)
    t_c = flops / (arch.peak_fp16_tflops * 1e12)
    t_m = nbytes / (arch.mem_bw_gbs * 1e9)
    return {"flops_padded": flops, "bytes_tiles": nbytes, "t_compute_s": t_c, "t_memory_s": t_m,
            "t_roofline_s": max(t_c, t_m), "bound": "compute" if t_c >= t_m else "memory",
            "tiles": tm * tn}


def predict_matmul(cfg: dict[str, Any], M: int, N: int, K: int, arch: ArchParams,
                   reg_model: RegModel = ANALYTIC_REGS, smem_model: SmemModel = ANALYTIC_SMEM
                   ) -> dict[str, Any]:
    """Every predictor term for one matmul config and shape. ``score`` is lower-is-better."""
    p = config.PREDICTOR
    maxnreg = cfg.get("maxnreg")
    regs = regs_est(cfg, arch, reg_model, maxnreg)
    smem = smem_est(cfg, smem_model)
    smem_ok = smem <= arch.smem_per_block_optin
    occ = occupancy(regs, smem, cfg["num_warps"], arch)
    roof = roofline_matmul(cfg, M, N, K, arch)
    wv = waves(roof["tiles"], occ["blocks_per_sm"], arch.sm_count)
    penalty = p.low_occupancy_penalty if occ["active_warps"] < p.min_active_warps else 1.0
    feasible = smem_ok and occ["blocks_per_sm"] > 0
    score = roof["t_roofline_s"] / wv["wave_efficiency"] * penalty if feasible else float("inf")
    return {"regs_pred": regs, "regs_raw": regs_raw(cfg, reg_model),
            "spill_pred": spill_pred(cfg, arch, reg_model, maxnreg), "smem_pred": smem,
            "smem_feasible": smem_ok, "feasible": feasible, **occ, **roof,
            "slots": wv["slots"], "n_waves": wv["n_waves"], "wave_efficiency": wv["wave_efficiency"],
            "occ_penalty": penalty, "score": score}


def predict_rmsnorm(cfg: dict[str, Any], rows: int, cols: int, arch: ArchParams) -> dict[str, Any]:
    """RMSNorm score: memory roofline / wave efficiency, times the idle-lane fraction when
    ``BLOCK_SIZE`` < threads and a small cost per sequential chunk of the row."""
    p = config.PREDICTOR
    t = threads(cfg)
    bs = cfg["BLOCK_SIZE"]
    occ = occupancy(32, 0, cfg["num_warps"], arch)
    wv = waves(rows, occ["blocks_per_sm"], arch.sm_count)
    t_m = p.elem_bytes * (2 * rows * cols + cols) / (arch.mem_bw_gbs * 1e9)
    lane_waste = max(1.0, t / bs)
    chunks = math.ceil(cols / bs)
    score = t_m / wv["wave_efficiency"] * lane_waste * (1.0 + p.rmsnorm_chunk_penalty * (chunks - 1))
    return {"regs_pred": None, "spill_pred": False, "smem_pred": 0, "feasible": True,
            **occ, "t_memory_s": t_m, "lane_waste": lane_waste, "chunks": chunks,
            "wave_efficiency": wv["wave_efficiency"], "score": score}


#: Kernel name -> predictor. A new kernel registers its scoring function here.
PREDICTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "matmul": lambda cfg, s, arch, reg, smem: predict_matmul(cfg, s.M, s.N, s.K, arch, reg, smem),
    "rmsnorm": lambda cfg, s, arch, reg, smem: predict_rmsnorm(cfg, s.M, s.N, arch),
}


def predict_frame(configs: Iterable[dict[str, Any]], shapes: Iterable[Any], arch: ArchParams,
                  reg_model: RegModel = ANALYTIC_REGS, smem_model: SmemModel = ANALYTIC_SMEM
                  ) -> pd.DataFrame:
    """Predictions for every (shape, config), joinable with the sweep on (shape_id, cfg_hash)."""
    configs = list(configs)
    rows = []
    for s in shapes:
        for cfg in configs:
            p = PREDICTORS[s.kernel](cfg, s, arch, reg_model, smem_model)
            rows.append({"kernel": s.kernel, "shape_id": s.shape_id, "cfg_hash": config_hash(cfg),
                         "model": reg_model.name, **p})
    return pd.DataFrame(rows)


def rank_smart(pred: pd.DataFrame, k: int | None = None) -> list[str]:
    """Smart-k: drop configs predicted not to fit or to spill, rank the rest by score (ties by
    hash) and return the first *k* hashes. If nothing survives, all configs are ranked."""
    keep = pred[pred["feasible"].astype(bool) & ~pred["spill_pred"].astype(bool)]
    if keep.empty:
        keep = pred
    ranked = keep.sort_values(["score", "cfg_hash"], kind="mergesort")["cfg_hash"].tolist()
    return ranked if k is None else ranked[:k]


# --------------------------------------------------------------------------- fitting


def configs_from_sweep(df: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct config in a sweep frame: hash, knob columns and measurements."""
    cols = ["cfg_hash"] + [c for c in df.columns if c.startswith("cfg_") and c != "cfg_hash"]
    meas = [c for c in ("n_regs", "n_spills", "shared_bytes") if c in df.columns]
    out = df[cols + meas]
    if meas:
        out = out.dropna(subset=[meas[0]])
    return out.drop_duplicates("cfg_hash").reset_index(drop=True)


def fit_regs(df: pd.DataFrame, max_regs: int = 255) -> RegModel:
    """Fit the register model by non-negative least squares on configs that neither spilled nor
    hit the cap, then fit the spill threshold. The caller passes training shapes only."""
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
                        fp_weight: float | None = None) -> float:
    """Threshold on the unclamped estimate minimizing ``fp_weight*FP + FN``.

    A false alarm removes a config from smart-k, possibly the fastest one, while a missed spill
    only wastes one compile, so false alarms weigh more. Ties go to the larger threshold, and the
    threshold never goes below the register cap (*floor*).
    """
    fp_weight = config.PREDICTOR.spill_false_alarm_weight if fp_weight is None else fp_weight
    cands = np.unique(np.concatenate([[floor], raw[raw >= floor]]))
    best_t, best_cost = float(floor), float("inf")
    for t in cands:
        pred = raw > t
        cost = fp_weight * np.sum(pred & ~spilled) + np.sum(~pred & spilled)
        if cost < best_cost or (cost == best_cost and t > best_t):
            best_t, best_cost = float(t), float(cost)
    return best_t


def fit_smem(df: pd.DataFrame) -> tuple[SmemModel, float]:
    """Grid-search :class:`SmemModel` variants (``config.PREDICTOR.smem_*``) for the highest
    exact-match rate on the training rows. Returns (best model, its exact-match rate)."""
    p = config.PREDICTOR
    d = df.dropna(subset=["shared_bytes"])
    cfgs = [cfg_from_row(r) for _, r in d.iterrows()]
    meas = d["shared_bytes"].to_numpy()
    best, best_rate = ANALYTIC_SMEM, -1.0
    for off, cap, epi, comb in itertools.product(p.smem_stage_offsets, p.smem_stage_caps,
                                                 p.smem_epilogue_bytes, p.smem_combines):
        m = SmemModel(off, cap, epi, comb, name="fitted")
        pred = np.array([smem_est(c, m) for c in cfgs])
        rate = float(np.mean(pred == meas)) if len(meas) else 0.0
        if rate > best_rate:
            best, best_rate = m, rate
    return best, best_rate


# --------------------------------------------------------------------------- evaluation


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
    """Confusion counts plus precision and recall of spill prediction."""
    p, m = np.asarray(pred, dtype=bool), np.asarray(meas, dtype=bool)
    tp, fp = int((p & m).sum()), int((p & ~m).sum())
    fn, tn = int((~p & m).sum()), int((~p & ~m).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan")}


def model_to_dict(model: RegModel | SmemModel) -> dict[str, Any]:
    """A fitted model as a JSON-ready dict."""
    return asdict(model)


# =========================================================================== search replay
#
# Every config has been measured, so a strategy is simulated by picking configs from the table and
# charging what they cost: cold compile + benchmark, plus predictor time for smart-k. Failed configs
# still cost their compile but can never be best, nor can incorrect ones.
# quality = best_time / best_found_time.

GROUP = ["gpu_tag", "kernel", "shape_id"]


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """Configs eligible to be "best": status ok, correct and timed."""
    correct = df["correct"].astype("boolean").fillna(False) if "correct" in df else True
    return (df["status"] == "ok") & correct.astype(bool) & df["time_ms_median"].notna()


def cold_compile_times(sweep: pd.DataFrame) -> pd.Series:
    """Cold compile time per (gpu_tag, kernel, cfg_hash), median over cache-miss rows.

    In the sweep a config compiles once and later shapes hit the cache, but an autotuner tuning
    one shape in a fresh process pays the cold compile, so the replay charges it to every shape.
    """
    miss = sweep["cache_miss"].astype("boolean").fillna(False).astype(bool)
    cold = sweep[miss & sweep["compile_time_s"].notna()]
    return cold.groupby(["gpu_tag", "kernel", "cfg_hash"])["compile_time_s"].median()


def prepare(sweep: pd.DataFrame) -> pd.DataFrame:
    """Add ``valid`` and ``cost_s`` (cold compile + benchmark seconds) columns."""
    df = sweep.copy()
    cold = cold_compile_times(df).rename("cold_compile_s")
    df = df.join(cold, on=["gpu_tag", "kernel", "cfg_hash"])
    df["cold_compile_s"] = df["cold_compile_s"].fillna(df["compile_time_s"]).fillna(0.0)
    bench = df["bench_time_s"].fillna(0.0) if "bench_time_s" in df else 0.0
    df["cost_s"] = df["cold_compile_s"] + bench
    df["valid"] = valid_mask(df)
    return df


def best_time(table: pd.DataFrame) -> float:
    """Best median time among valid configs (``nan`` if none)."""
    v = table.loc[table["valid"], "time_ms_median"]
    return float(v.min()) if len(v) else float("nan")


def evaluate_order(table: pd.DataFrame, order: Sequence[str], best: float,
                   extra_cost_s: float = 0.0) -> pd.DataFrame:
    """Cumulative cost and quality after trying the first i configs of *order*."""
    t = table.set_index("cfg_hash").loc[list(order)]
    times = np.where(t["valid"].to_numpy(), t["time_ms_median"].to_numpy(dtype=float), np.inf)
    found = np.minimum.accumulate(times)
    quality = np.where(np.isfinite(found), best / found, 0.0)
    return pd.DataFrame({"n_configs": np.arange(1, len(order) + 1),
                         "cost_s": np.cumsum(t["cost_s"].to_numpy(dtype=float)) + extra_cost_s,
                         "quality": quality, "best_found_ms": found})


def _at_k(curve: pd.DataFrame, k: int | None) -> dict[str, float]:
    i = len(curve) if k is None else min(k, len(curve))
    r = curve.iloc[i - 1]
    return {"n_configs": int(r["n_configs"]), "cost_s": float(r["cost_s"]), "quality": float(r["quality"])}


def _first_reach(curve: pd.DataFrame, target: float) -> dict[str, float]:
    hit = curve[curve["quality"] >= target]
    if hit.empty:
        return {"reached": False, "configs_to_target": np.nan, "cost_to_target_s": np.nan}
    r = hit.iloc[0]
    return {"reached": True, "configs_to_target": int(r["n_configs"]), "cost_to_target_s": float(r["cost_s"])}


def random_orders(cfg_hashes: Sequence[str], n_seeds: int | None = None, base_seed: int = 0) -> list[list[str]]:
    """Seeded random permutations; the first k of each is a uniform random-k sample."""
    n_seeds = config.SEARCH.n_seeds if n_seeds is None else n_seeds
    hs = sorted(cfg_hashes)
    return [list(np.random.default_rng(base_seed + s).permutation(hs)) for s in range(n_seeds)]


def replay_table(table: pd.DataFrame, smart_order: Sequence[str], predictor_time_s: float = 0.0,
                 ks: Sequence[int | None] | None = None, n_seeds: int | None = None,
                 target: float | None = None, base_seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay exhaustive, random-k and smart-k on one (gpu, kernel, shape) table.

    Returns curve rows (strategy, seed, k, n_configs, cost_s, quality) and target rows
    (strategy, seed, reached, configs_to_target, cost_to_target_s).
    """
    s = config.SEARCH
    ks = s.ks if ks is None else ks
    n_seeds = s.n_seeds if n_seeds is None else n_seeds
    target = s.target if target is None else target
    best = best_time(table)
    curve_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    all_hashes = table["cfg_hash"].tolist()

    exh = evaluate_order(table, sorted(all_hashes), best)
    curve_rows.append({"strategy": "exhaustive", "seed": -1, "k": len(all_hashes), **_at_k(exh, None)})
    target_rows.append({"strategy": "exhaustive", "seed": -1,
                        "reached": bool(np.isfinite(best)), "configs_to_target": len(all_hashes),
                        "cost_to_target_s": float(exh["cost_s"].iloc[-1])})

    for seed, order in enumerate(random_orders(all_hashes, n_seeds, base_seed)):
        c = evaluate_order(table, order, best)
        for k in ks:
            curve_rows.append({"strategy": "random", "seed": seed, "k": k or len(order), **_at_k(c, k)})
        target_rows.append({"strategy": "random", "seed": seed, **_first_reach(c, target)})

    smart = [h for h in smart_order if h in set(all_hashes)]
    if smart:
        c = evaluate_order(table, smart, best, extra_cost_s=predictor_time_s)
        for k in ks:
            curve_rows.append({"strategy": "smart", "seed": -1, "k": k or len(smart), **_at_k(c, k)})
        target_rows.append({"strategy": "smart", "seed": -1, **_first_reach(c, target)})
    return pd.DataFrame(curve_rows), pd.DataFrame(target_rows)


def replay(sweep: pd.DataFrame, smart_orders: dict[tuple[str, str, str], Sequence[str]],
           split: dict[str, str] | None = None, predictor_time_s: dict[tuple[str, str, str], float] | None = None,
           **kwargs: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay every (gpu_tag, kernel, shape_id) group of a :func:`prepare`-d sweep.

    *smart_orders* holds the ranked hashes per group, *split* maps shape_id to train/test and
    *predictor_time_s* is the measured predictor time per group, added to smart-k's cost.
    """
    curves, targets = [], []
    for key, table in sweep.groupby(GROUP, sort=True):
        key = tuple(key)
        c, t = replay_table(table, smart_orders.get(key, []),
                            (predictor_time_s or {}).get(key, 0.0), **kwargs)
        for frame in (c, t):
            frame[GROUP] = list(key)
            frame["split"] = (split or {}).get(key[2], "all")
        curves.append(c)
        targets.append(t)
    return pd.concat(curves, ignore_index=True), pd.concat(targets, ignore_index=True)


def summarize_targets(targets: pd.DataFrame) -> pd.DataFrame:
    """Cost to reach the target per strategy x GPU x split.

    Per shape, random is the median over seeds (with the fraction of seeds that got there); then
    shapes are aggregated by median.
    """
    per = (targets.groupby(["gpu_tag", "kernel", "split", "shape_id", "strategy"])
           .agg(reach_rate=("reached", "mean"), configs_to_target=("configs_to_target", "median"),
                cost_to_target_s=("cost_to_target_s", "median"))
           .reset_index())
    return (per.groupby(["gpu_tag", "kernel", "split", "strategy"])
            .agg(n_shapes=("shape_id", "nunique"), reach_rate=("reach_rate", "mean"),
                 median_configs=("configs_to_target", "median"),
                 median_cost_s=("cost_to_target_s", "median"))
            .reset_index())


def per_shape(targets: pd.DataFrame, kernel: str = "matmul") -> pd.DataFrame:
    """Per shape: median configs and seconds to the target, and the random/smart cost ratio."""
    t = targets[targets["kernel"] == kernel]
    g = (t.groupby(["split", "shape_id", "strategy"])
         .agg(configs=("configs_to_target", "median"), cost_s=("cost_to_target_s", "median"))
         .reset_index())
    cfg = g.pivot_table(index=["split", "shape_id"], columns="strategy", values="configs")
    cost = g.pivot_table(index=["split", "shape_id"], columns="strategy", values="cost_s")
    out = pd.DataFrame({"random_configs": cfg.get("random"), "smart_configs": cfg.get("smart"),
                        "random_cost_s": cost.get("random"), "smart_cost_s": cost.get("smart")})
    out["random_over_smart"] = out["random_cost_s"] / out["smart_cost_s"]
    return out.reset_index()


def cost_ratio(summary: pd.DataFrame, a: str = "random", b: str = "smart") -> pd.DataFrame:
    """``median_cost(a) / median_cost(b)`` per GPU x kernel x split."""
    piv = summary.pivot_table(index=["gpu_tag", "kernel", "split"], columns="strategy",
                              values="median_cost_s")
    if a not in piv or b not in piv:
        return pd.DataFrame()
    return (piv[a] / piv[b]).rename(f"{a}_over_{b}").reset_index()


# =========================================================================== triton.autotune hook
#
# Triton 3.2 first calls early_config_prune(configs, named_args, **kwargs) and then, if more than
# top_k configs are left, ranks them with perf_model(**kwargs). ArchTune uses both: the early prune
# drops configs predicted to overflow shared memory or spill and keeps the best-scoring k, and the
# perf model is the same roofline score.


@dataclass
class HookStats:
    """Time spent in the predictor and how many configs went in and out."""

    predictor_time_s: float = 0.0
    calls: int = 0
    n_in: list[int] = field(default_factory=list)
    n_out: list[int] = field(default_factory=list)


def prune_configs(configs: Sequence[Any], M: int, N: int, K: int, arch: ArchParams,
                  reg_model: RegModel = ANALYTIC_REGS, smem_model: SmemModel = ANALYTIC_SMEM,
                  k: int | None = None) -> list[Any]:
    """Smart-k on ``triton.Config`` objects: drop infeasible or spilling ones, rank, keep *k*."""
    scored = []
    for i, c in enumerate(configs):
        p = predict_matmul(from_triton_config(c), M, N, K, arch, reg_model, smem_model)
        scored.append((p["feasible"] and not p["spill_pred"], p["score"], i, c))
    keep = [s for s in scored if s[0]] or scored  # keep everything if everything was dropped
    keep.sort(key=lambda s: (s[1], s[2]))
    out = [s[3] for s in keep]
    return out if k is None else out[:k]


def perf_model_score(arch: ArchParams, reg_model: RegModel = ANALYTIC_REGS,
                     smem_model: SmemModel = ANALYTIC_SMEM) -> Callable[..., float]:
    """A ``perf_model(**kwargs)`` that returns the predictor score (lower is better)."""

    def perf_model(**kw: Any) -> float:
        cfg = {k: kw[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                  "num_warps", "num_stages") if k in kw}
        if kw.get("maxnreg") is not None:
            cfg["maxnreg"] = kw["maxnreg"]
        return predict_matmul(cfg, kw["M"], kw["N"], kw["K"], arch, reg_model, smem_model)["score"]

    return perf_model


def archtune_prune_configs_by(arch: ArchParams, k: int, reg_model: RegModel = ANALYTIC_REGS,
                              smem_model: SmemModel = ANALYTIC_SMEM,
                              stats: HookStats | None = None) -> dict[str, Any]:
    """The ``prune_configs_by`` dict for ``@triton.autotune``.

    ``early_config_prune`` does the whole smart-k policy, so ``perf_model`` only re-sorts when
    Triton applies it; time spent in both goes into ``_stats``.
    """
    stats = stats if stats is not None else HookStats()

    def early_config_prune(configs: list, named_args: dict, **kwargs: Any) -> list:
        t0 = time.perf_counter()
        out = prune_configs(configs, named_args["M"], named_args["N"], named_args["K"], arch,
                            reg_model, smem_model, k)
        stats.predictor_time_s += time.perf_counter() - t0
        stats.calls += 1
        stats.n_in.append(len(configs))
        stats.n_out.append(len(out))
        return out

    base_pm = perf_model_score(arch, reg_model, smem_model)

    def perf_model(**kw: Any) -> float:
        t0 = time.perf_counter()
        s = base_pm(**kw)
        stats.predictor_time_s += time.perf_counter() - t0
        return s

    return {"early_config_prune": early_config_prune, "perf_model": perf_model, "top_k": k,
            "_stats": stats}


def tune_live(configs: Sequence[dict[str, Any]], M: int, N: int, K: int,
              prune_by: dict[str, Any] | None = None, seed: int = 0) -> dict[str, Any]:
    """One live ``triton.autotune`` run with fresh caches, timed.

    Returns the tuning wall time, the chosen config, its runtime and, when *prune_by* came from
    :func:`archtune_prune_configs_by`, the predictor stats.
    """
    import torch

    mm = gpu.MATMUL
    gpu.fresh_cache_dir()
    tcfgs = [to_triton_config(c) for c in configs]
    stats = prune_by.pop("_stats", None) if prune_by else None
    # Triton 3.2's decorator does not forward do_bench, so the autotuner uses its default benchmark.
    kern = mm.make_autotuned(tcfgs, prune_configs_by=prune_by, do_bench=gpu.autotune_do_bench)
    a, b, c = mm.make_inputs(M, N, K, seed=seed)
    args = mm.kernel_args(a, b, c)
    grid = lambda meta: (  # noqa: E731
        ((M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"])
        * ((N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"]),)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kern[grid](*args)
    torch.cuda.synchronize()
    tuning_s = time.perf_counter() - t0
    best = from_triton_config(kern.best_config)
    ok, err = gpu.check_correct(c, mm.reference(a, b))
    # The GPU may have idled during the last compiles, so warm it up before timing the winner.
    timing = gpu.time_config(mm.launcher(a, b, c, best), warmup=config.BENCH.post_idle_warmup_ms,
                             n_repetitions=config.BENCH.final_repetitions)
    out = {"M": M, "N": N, "K": K, "n_configs_given": len(configs), "tuning_s": tuning_s,
           "best_cfg": best, "correct": ok, "max_abs_err": err, **timing}
    if stats is not None:
        out.update(predictor_time_s=stats.predictor_time_s,
                   n_after_prune=stats.n_out[-1] if stats.n_out else None)
    return out


# =========================================================================== Inductor (experimental)
#
# On torch 2.6 Inductor takes its matmul template configs from
# torch._inductor.kernel.mm_common.mm_configs and imports that name into torch._inductor.kernel.mm.
# The patch wraps it and drops the configs the predictor expects to fail. Inductor internals change
# between releases, so it checks for the attribute first, and it only runs with
# ARCHTUNE_INDUCTOR_PATCH=1.

_INDUCTOR_ORIGINAL: dict[str, Any] = {}


def inductor_patch_enabled() -> bool:
    """Whether ``ARCHTUNE_INDUCTOR_PATCH=1`` is set."""
    return os.environ.get(config.INDUCTOR_PATCH_ENV, "0") == "1"


def _cfg_from_inductor(c: Any) -> dict[str, Any] | None:
    """Inductor's ``triton.Config`` objects use BLOCK_M/N/K."""
    kw = getattr(c, "kwargs", None)
    if not kw or not {"BLOCK_M", "BLOCK_N", "BLOCK_K"} <= set(kw):
        return None
    return {"BLOCK_SIZE_M": kw["BLOCK_M"], "BLOCK_SIZE_N": kw["BLOCK_N"], "BLOCK_SIZE_K": kw["BLOCK_K"],
            "GROUP_SIZE_M": kw.get("GROUP_M", 8), "num_warps": c.num_warps, "num_stages": c.num_stages}


def patch_inductor_mm_configs(keep: Callable[[dict[str, Any], int, int, int], bool]) -> tuple[bool, str]:
    """Wrap Inductor's ``mm_configs`` so only configs with ``keep(cfg, m, n, k)`` survive.

    Returns (patched, message).
    """
    if not inductor_patch_enabled():
        return False, f"feature flag {config.INDUCTOR_PATCH_ENV} is not set"
    try:
        from torch._inductor.kernel import mm as mm_mod
        from torch._inductor.kernel import mm_common
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import Inductor mm modules: {exc}"
    orig = getattr(mm_common, "mm_configs", None)
    if orig is None or not hasattr(mm_mod, "mm_configs"):
        return False, "torch._inductor.kernel.mm_common.mm_configs not found on this torch"
    _INDUCTOR_ORIGINAL.setdefault("mm_configs", orig)

    def wrapped(m: Any, n: Any, k: Any, *args: Any, **kwargs: Any):
        configs = list(_INDUCTOR_ORIGINAL["mm_configs"](m, n, k, *args, **kwargs))
        try:
            mi, ni, ki = int(m), int(n), int(k)
        except (TypeError, ValueError):  # symbolic shapes: leave them alone
            return configs
        kept = [c for c in configs if (cfg := _cfg_from_inductor(c)) is None or keep(cfg, mi, ni, ki)]
        return kept or configs

    mm_common.mm_configs = wrapped
    mm_mod.mm_configs = wrapped
    return True, "patched torch._inductor.kernel.mm.mm_configs"


def unpatch_inductor() -> None:
    """Restore Inductor's original ``mm_configs``."""
    if "mm_configs" in _INDUCTOR_ORIGINAL:
        from torch._inductor.kernel import mm as mm_mod
        from torch._inductor.kernel import mm_common

        mm_common.mm_configs = _INDUCTOR_ORIGINAL["mm_configs"]
        mm_mod.mm_configs = _INDUCTOR_ORIGINAL.pop("mm_configs")


def inductor_compile_and_time(M: int, N: int, K: int, seed: int = 0) -> dict[str, Any]:
    """``torch.compile(mode="max-autotune")`` of one fp16 matmul: compile+tune time and runtime.

    Dynamo is reset first so every call tunes again.
    """
    import torch
    import torch._dynamo

    torch._dynamo.reset()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16, generator=gen)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16, generator=gen)
    fn = torch.compile(lambda x, y: x @ y, mode="max-autotune", dynamic=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(a, b)
    torch.cuda.synchronize()
    compile_s = time.perf_counter() - t0
    ok, err = gpu.check_correct(out, torch.matmul(a, b))
    timing = gpu.time_config(lambda: fn(a, b), warmup=config.BENCH.post_idle_warmup_ms,
                             n_repetitions=config.BENCH.final_repetitions)
    return {"M": M, "N": N, "K": K, "compile_and_tune_s": compile_s, "correct": ok,
            "max_abs_err": err, "patched": "mm_configs" in _INDUCTOR_ORIGINAL, **timing}
