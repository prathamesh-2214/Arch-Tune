"""Search strategies and replay evaluation over exhaustive sweep data (spec §10).

A *table* is the sweep frame for one (gpu_tag, kernel, shape_id). Replaying a strategy means
picking configs from the table and charging their measured cost:

    cost = Σ (cold compile time + benchmark time) over tried configs  (+ predictor time)

Failed configs cost their compile time but can never be "best"; incorrect configs are never
best (spec: correctness before speed). ``quality = best_time / best_found_time``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

KS: tuple[int | None, ...] = (1, 2, 4, 8, 16, 32, 64, None)  # None = all
N_SEEDS = 20
TARGET = 0.95
GROUP = ["gpu_tag", "kernel", "shape_id"]


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """Configs eligible to be "best": status ok, correct, and timed."""
    correct = df["correct"].astype("boolean").fillna(False) if "correct" in df else True
    return (df["status"] == "ok") & correct.astype(bool) & df["time_ms_median"].notna()


def cold_compile_times(sweep: pd.DataFrame) -> pd.Series:
    """Cold compile time per (gpu_tag, kernel, cfg_hash): median over cache-miss rows.

    Within a sweep session a config is compiled once and later shapes hit the in-memory
    cache, but an autotuner tuning one shape in a fresh process pays the cold compile, so the
    replay charges the cold time to every shape.
    """
    miss = sweep["cache_miss"].astype("boolean").fillna(False).astype(bool)
    cold = sweep[miss & sweep["compile_time_s"].notna()]
    return cold.groupby(["gpu_tag", "kernel", "cfg_hash"])["compile_time_s"].median()


def prepare(sweep: pd.DataFrame) -> pd.DataFrame:
    """Add ``valid`` and ``cost_s`` (= cold compile + bench seconds) columns."""
    df = sweep.copy()
    cold = cold_compile_times(df).rename("cold_compile_s")
    df = df.join(cold, on=["gpu_tag", "kernel", "cfg_hash"])
    df["cold_compile_s"] = df["cold_compile_s"].fillna(df["compile_time_s"]).fillna(0.0)
    bench = df["bench_time_s"].fillna(0.0) if "bench_time_s" in df else 0.0
    df["cost_s"] = df["cold_compile_s"] + bench
    df["valid"] = valid_mask(df)
    return df


def best_time(table: pd.DataFrame) -> float:
    """Exhaustive best median time among valid configs (``nan`` if none)."""
    v = table.loc[table["valid"], "time_ms_median"]
    return float(v.min()) if len(v) else float("nan")


def evaluate_order(table: pd.DataFrame, order: Sequence[str], best: float,
                   extra_cost_s: float = 0.0) -> pd.DataFrame:
    """Cumulative cost / quality after trying the first i configs of *order* (i = 1..len)."""
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


def random_orders(cfg_hashes: Sequence[str], n_seeds: int = N_SEEDS, base_seed: int = 0) -> list[list[str]]:
    """Seeded random permutations; the first k of a permutation is a uniform random-k sample."""
    hs = sorted(cfg_hashes)
    return [list(np.random.default_rng(base_seed + s).permutation(hs)) for s in range(n_seeds)]


def replay_table(table: pd.DataFrame, smart_order: Sequence[str], predictor_time_s: float = 0.0,
                 ks: Sequence[int | None] = KS, n_seeds: int = N_SEEDS, target: float = TARGET,
                 base_seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay exhaustive, random-k and smart-k on one (gpu, kernel, shape) table.

    Returns ``(curve_rows, target_rows)``:

    * curve rows: strategy, seed, k, n_configs, cost_s, quality — for figure 1;
    * target rows: strategy, seed, reached, configs_to_target, cost_to_target_s — for table 6.
    """
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
    """Replay every (gpu_tag, kernel, shape_id) group of a prepared sweep frame.

    Args:
        sweep: output of :func:`prepare`.
        smart_orders: ranked cfg hashes per group key (from :func:`archtune.predict.rank_smart`).
        split: shape_id -> ``"train"`` / ``"test"``.
        predictor_time_s: measured predictor wall time per group (added to smart's cost).
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
    """Table 6: cost to reach the target per strategy × GPU × split.

    Per shape, random is summarized by the median over seeds (with the fraction of seeds that
    reached the target); then shapes are aggregated by median.
    """
    per_shape = (targets.groupby(["gpu_tag", "kernel", "split", "shape_id", "strategy"])
                 .agg(reach_rate=("reached", "mean"),
                      configs_to_target=("configs_to_target", "median"),
                      cost_to_target_s=("cost_to_target_s", "median"))
                 .reset_index())
    return (per_shape.groupby(["gpu_tag", "kernel", "split", "strategy"])
            .agg(n_shapes=("shape_id", "nunique"), reach_rate=("reach_rate", "mean"),
                 median_configs=("configs_to_target", "median"),
                 median_cost_s=("cost_to_target_s", "median"))
            .reset_index())


def per_shape(targets: pd.DataFrame, kernel: str = "matmul") -> pd.DataFrame:
    """Per-shape median configs / seconds to reach the target, and the random/smart cost ratio."""
    t = targets[targets["kernel"] == kernel]
    g = (t.groupby(["split", "shape_id", "strategy"])
         .agg(configs=("configs_to_target", "median"), cost_s=("cost_to_target_s", "median"))
         .reset_index())
    cfg = g.pivot_table(index=["split", "shape_id"], columns="strategy", values="configs")
    cost = g.pivot_table(index=["split", "shape_id"], columns="strategy", values="cost_s")
    out = pd.DataFrame({
        "random_configs": cfg.get("random"), "smart_configs": cfg.get("smart"),
        "random_cost_s": cost.get("random"), "smart_cost_s": cost.get("smart"),
    })
    out["random_over_smart"] = out["random_cost_s"] / out["smart_cost_s"]
    return out.reset_index()


def cost_ratio(summary: pd.DataFrame, a: str = "random", b: str = "smart") -> pd.DataFrame:
    """``median_cost(a) / median_cost(b)`` per GPU × kernel × split (e.g. random/smart)."""
    piv = summary.pivot_table(index=["gpu_tag", "kernel", "split"], columns="strategy",
                              values="median_cost_s")
    if a not in piv or b not in piv:
        return pd.DataFrame()
    return (piv[a] / piv[b]).rename(f"{a}_over_{b}").reset_index()
