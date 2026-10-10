"""Compiler experiments (Phase 6).

* ``maxnreg`` sweep: cap registers per thread on a few representative configs and observe
  spills, occupancy and runtime (requires ``maxnreg`` support in the pinned Triton).
* ``num_stages`` sweep: pipelining depth vs shared memory, ``cp.async`` presence, runtime.

Experiments reuse :func:`archtune.sweep.run_sweep` with their own table name and run id, so
they are resumable and carry the full environment in every row.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from archtune.search import valid_mask

MAXNREG_CAPS: tuple[int | None, ...] = (None, 255, 192, 168, 128, 96, 64)
NUM_STAGES: tuple[int, ...] = (1, 2, 3, 4, 5, 6)


def maxnreg_supported() -> tuple[bool, str]:
    """Check whether ``triton.Config`` accepts ``maxnreg`` on the installed Triton."""
    try:
        import inspect

        import triton

        params = inspect.signature(triton.Config.__init__).parameters
        if "maxnreg" in params:
            return True, f"triton {triton.__version__}: triton.Config(..., maxnreg=...) present"
        return False, f"triton {triton.__version__}: triton.Config has no maxnreg parameter"
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot inspect triton.Config: {exc}"


def with_maxnreg(configs: Sequence[dict[str, Any]], caps: Sequence[int | None] = MAXNREG_CAPS
                 ) -> list[dict[str, Any]]:
    """Cross product of configs and register caps (``None`` = uncapped)."""
    return [{**c, "maxnreg": cap} if cap is not None else dict(c) for c in configs for cap in caps]


def with_num_stages(configs: Sequence[dict[str, Any]], stages: Sequence[int] = NUM_STAGES
                    ) -> list[dict[str, Any]]:
    """Each config at every pipelining depth (duplicates removed, order kept)."""
    out, seen = [], set()
    for c in configs:
        base = {k: v for k, v in c.items() if k != "num_stages"}
        for s in stages:
            key = tuple(sorted({**base, "num_stages": s}.items()))
            if key not in seen:
                seen.add(key)
                out.append({**base, "num_stages": s})
    return out


def _cfg_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("cfg_") and c != "cfg_hash"]


def cfg_of(row: pd.Series) -> dict[str, Any]:
    """Config dict from the ``cfg_*`` columns of a row (NaN knobs dropped)."""
    return {c[4:]: int(row[c]) for c in row.index
            if c.startswith("cfg_") and c != "cfg_hash" and pd.notna(row[c])}


def representative_configs(sweep: pd.DataFrame, shape_id: str) -> list[dict[str, Any]]:
    """Pick up to 3 contrasting configs on one shape for the register-cap study:

    1. the fastest valid config;
    2. the valid config with the most registers that does not spill;
    3. the fastest config that spills (if any) — otherwise the second-highest-register one.
    """
    d = sweep[(sweep["shape_id"] == shape_id)].copy()
    d["valid"] = valid_mask(d)
    v = d[d["valid"]]
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
            out.append(cfg_of(r))
    return out


def check_single_stack(df: pd.DataFrame) -> str:
    """Raise if rows come from more than one software stack (spec §12); return the stack id."""
    ids = df["stack_id"].dropna().unique() if "stack_id" in df else []
    if len(ids) != 1:
        raise ValueError(f"expected exactly one stack_id, found {list(ids)}; never compare across stacks")
    return str(ids[0])


def _label(cfg: dict[str, Any]) -> str:
    return (f"{cfg['BLOCK_SIZE_M']}x{cfg['BLOCK_SIZE_N']}x{cfg['BLOCK_SIZE_K']} "
            f"w{cfg['num_warps']} s{cfg['num_stages']}")


def maxnreg_frame(df: pd.DataFrame, arch: Any, uncapped_as: int = 256) -> pd.DataFrame:
    """Figure-3 frame from ``experiments_maxnreg`` rows of one shape.

    Resident blocks/SM are computed with the occupancy rules from the *measured* registers
    and shared memory, so the occupancy axis reflects what ptxas actually allocated.
    """
    from archtune.predict import occupancy

    d = df.copy()
    if "cfg_maxnreg" not in d:
        d["cfg_maxnreg"] = np.nan
    d["cap"] = d["cfg_maxnreg"].fillna(uncapped_as).astype(int)
    base_cols = [c for c in d.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_maxnreg")]
    d["cfg_label"] = d.apply(lambda r: _label({c[4:]: int(r[c]) for c in base_cols}), axis=1)
    d["blocks_per_sm"] = [
        occupancy(int(r.n_regs), int(r.shared_bytes), int(r.cfg_num_warps), arch)["blocks_per_sm"]
        if pd.notna(r.n_regs) and pd.notna(r.shared_bytes) else np.nan
        for r in d.itertuples()]
    unc = d[d["cap"] == uncapped_as].set_index("cfg_label")["time_ms_median"]
    d["time_rel"] = d["time_ms_median"] / d["cfg_label"].map(unc)
    return d[["cfg_label", "cap", "n_regs", "n_spills", "shared_bytes", "blocks_per_sm",
              "time_ms_median", "time_rel", "status"]].sort_values(["cfg_label", "cap"])


def num_stages_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Figure-4 frame from ``experiments_stages`` rows: runtime relative to each
    (GPU, shape, base config)'s best stage count."""
    d = df.copy()
    base_cols = [c for c in d.columns if c.startswith("cfg_") and c not in ("cfg_hash", "cfg_num_stages")]
    d["base"] = d[base_cols].astype(str).agg("|".join, axis=1)
    d["num_stages"] = d["cfg_num_stages"].astype(int)
    best = d[valid_mask(d)].groupby(["gpu_tag", "shape_id", "base"])["time_ms_median"].min().rename("best")
    d = d.join(best, on=["gpu_tag", "shape_id", "base"])
    d["time_rel"] = d["time_ms_median"] / d["best"]
    return d[["gpu_tag", "shape_id", "base", "num_stages", "shared_bytes", "ptx_cp_async",
              "n_regs", "n_spills", "time_ms_median", "time_rel", "status"]]
