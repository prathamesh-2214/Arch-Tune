"""All figures (spec §10), one consistent style.

Colors follow a fixed categorical order validated for color-vision deficiency (first three
slots validate on all pairs): series never repaint when the number of series changes.
Every figure has one y-scale per panel; multiple measures are separate panels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
#: Fixed entity -> color mapping (color follows the entity, never its rank).
STRATEGY_COLORS = {"random": SERIES[0], "smart": SERIES[1], "exhaustive": SERIES[2],
                   "smart_analytic": SERIES[1]}  # same entity family; told apart by dash + marker
GPU_ORDER = ["rtx3050_sm86", "t4_sm75", "a100_sm80", "l4_sm89"]


def gpu_color(tag: str) -> str:
    """Stable color per GPU tag."""
    return SERIES[GPU_ORDER.index(tag)] if tag in GPU_ORDER else SERIES[3 + hash(tag) % 4]


def style() -> None:
    """Apply the shared matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2.0, "lines.markersize": 6, "legend.frameon": False,
        "legend.fontsize": 8, "legend.labelcolor": INK, "font.size": 9, "text.color": INK,
    })


def _save(fig: Any, path: str | Path | None) -> Any:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160, bbox_inches="tight")
    return fig


def fig1_quality_vs_cost(curves: pd.DataFrame, gpu_tag: str, path: str | Path | None = None,
                         target: float = 0.95, kernel: str = "matmul") -> Any:
    """Figure 1: quality vs tuning cost, one panel per split.

    Each tuning budget k is a discrete choice, so strategies are drawn as **markers** (one per
    k) joined by a thin dotted connector that only shows the order of increasing k.

    * random-k: mean quality per k over shapes x seeds, vertical bars = IQR (25th-75th pct);
    * smart-k (fitted: filled squares; analytic: hollow triangles): median over shapes,
      fitted points labelled with their k;
    * exhaustive: one point (it has no budget knob: always every config).

    x = median cumulative cost (s), log scale.
    """
    style()
    d = curves[(curves["gpu_tag"] == gpu_tag) & (curves["kernel"] == kernel)]
    splits = [s for s in ("train", "test", "all") if s in set(d["split"])]
    fig, axes = plt.subplots(1, len(splits), figsize=(5.4 * len(splits), 4.0), sharey=True, squeeze=False)
    connector = dict(linestyle=":", linewidth=1.0, alpha=0.6)
    for ax, split in zip(axes[0], splits):
        ds = d[d["split"] == split]
        e = ds[ds["strategy"] == "exhaustive"]
        n_all = int(e["n_configs"].max()) if not e.empty else None
        r = ds[ds["strategy"] == "random"]
        if n_all is not None:  # random with k = all *is* exhaustive: shown once, as the diamond
            r = r[r["n_configs"] < n_all]
        if not r.empty:
            g = r.groupby("n_configs")
            agg = pd.DataFrame({"cost": g["cost_s"].median(), "mean": g["quality"].mean(),
                                "q25": g["quality"].quantile(0.25), "q75": g["quality"].quantile(0.75)})
            col = STRATEGY_COLORS["random"]
            ax.plot(agg["cost"], agg["mean"], color=col, **connector)
            ax.errorbar(agg["cost"], agg["mean"],
                        yerr=[(agg["mean"] - agg["q25"]).clip(lower=0), (agg["q75"] - agg["mean"]).clip(lower=0)],
                        fmt="o", color=col, markersize=6.5, elinewidth=1.2, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=1.2,
                        label="random-k (mean; bar = IQR over seeds × shapes)")
        for strat, marker, filled, label in (
                ("smart", "s", True, "smart-k, fitted (median over shapes)"),
                ("smart_analytic", "^", False, "smart-k, analytic (median over shapes)")):
            s = ds[ds["strategy"] == strat]
            if s.empty:
                continue
            g = s.groupby("n_configs")
            x, y = g["cost_s"].median(), g["quality"].median()
            col = STRATEGY_COLORS[strat]
            ax.plot(x, y, color=col, **connector)
            ax.plot(x, y, marker, linestyle="none", color=col, markersize=6.5, label=label,
                    markerfacecolor=col if filled else SURFACE,
                    markeredgecolor=SURFACE if filled else col, markeredgewidth=1.2 if filled else 1.5)
            if strat == "smart":
                for k, xi, yi in zip(x.index, x.to_numpy(), y.to_numpy()):
                    ax.annotate(f"{k}", (xi, yi), xytext=(0, 6), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color=INK_2)
        if not e.empty:
            ax.plot([e["cost_s"].median()], [1.0], "D", color=STRATEGY_COLORS["exhaustive"], linestyle="none",
                    markersize=9, label=f"exhaustive (all {n_all} configs)", zorder=5,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(target, color=INK_2, linestyle="--", linewidth=1, zorder=0,
                   label=f"{target:.0%} quality target")
        ax.set_ylim(top=1.09)
        ax.set_xscale("log")
        ax.set_xlabel("tuning cost: compile + benchmark seconds (log)")
        n = ds["shape_id"].nunique()
        ax.set_title(f"{gpu_tag} {kernel} — {split} shapes (n={n})", pad=10)
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
    """Figure 2: predicted vs measured registers and shared memory, spill confusion matrix.

    *pm* has columns regs_pred, n_regs, smem_pred, shared_bytes, spill_pred, n_spills.
    """
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    ax.scatter(pm["n_regs"], pm["regs_pred"], s=14, color=SERIES[0], alpha=0.6, linewidths=0)
    lim = [0, 260]
    ax.plot(lim, lim, color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("measured registers / thread (ptxas)")
    ax.set_ylabel("predicted registers / thread")
    ax.set_title("Registers")
    ax = axes[1]
    kb = 1024
    ax.scatter(pm["shared_bytes"] / kb, pm["smem_pred"] / kb, s=14, color=SERIES[0], alpha=0.6, linewidths=0)
    m = float(np.nanmax([pm["shared_bytes"].max(), pm["smem_pred"].max()])) / kb * 1.05
    ax.plot([0, m], [0, m], color=INK_2, linewidth=1, linestyle="--")
    ax.set_xlabel("measured shared memory (KB)")
    ax.set_ylabel("predicted shared memory (KB)")
    ax.set_title("Shared memory")
    ax = axes[2]
    meas = pm["n_spills"].fillna(0) > 0
    pred = pm["spill_pred"].astype(bool)
    cm = np.array([[int((~pred & ~meas).sum()), int((pred & ~meas).sum())],
                   [int((~pred & meas).sum()), int((pred & meas).sum())]])
    ax.imshow(cm, cmap=matplotlib.colors.ListedColormap(SEQ_BLUE), vmin=0)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=12,
                color=SURFACE if v > cm.max() / 2 else INK)
    ax.set_xticks([0, 1], ["no spill", "spill"]), ax.set_yticks([0, 1], ["no spill", "spill"])
    ax.set_xlabel("predicted"), ax.set_ylabel("measured (ptxas n_spills > 0)")
    ax.grid(False)
    ax.set_title("Spill prediction")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig3_maxnreg(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """Figure 3: register cap vs spills, resident blocks/SM and runtime (one line per config).

    *df* columns: cfg_label, cap (int; uncapped plotted as 256), n_spills, blocks_per_sm,
    time_rel (runtime / uncapped runtime).
    """
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    panels = [("n_spills", "spilled registers (ptxas)"), ("blocks_per_sm", "resident blocks / SM"),
              ("time_rel", "runtime / uncapped runtime")]
    labels = list(dict.fromkeys(df["cfg_label"]))
    for ax, (col, ylab) in zip(axes, panels):
        for i, lab in enumerate(labels):
            d = df[df["cfg_label"] == lab].sort_values("cap")
            ax.plot(d["cap"], d[col], "-o", color=SERIES[i % 3], label=lab, markersize=5)
        ax.set_ylabel(ylab)
        ax.set_xlabel("maxnreg cap (256 = uncapped)")
    axes[2].axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
    axes[0].legend(loc="upper right")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig4_num_stages(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """Figure 4: num_stages vs shared memory, static cp.async count and runtime, per GPU.

    *df* columns: gpu_tag, num_stages, shared_bytes, ptx_cp_async, time_rel.
    """
    style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    panels = [("shared_bytes", "shared memory (KB)", 1 / 1024), ("ptx_cp_async", "cp.async in PTX (static)", 1),
              ("time_rel", "runtime / best runtime", 1)]
    for ax, (col, ylab, scale) in zip(axes, panels):
        for tag in [t for t in GPU_ORDER if t in set(df["gpu_tag"])] + \
                   sorted(set(df["gpu_tag"]) - set(GPU_ORDER)):
            d = df[df["gpu_tag"] == tag].groupby("num_stages")[col].median().reset_index()
            ax.plot(d["num_stages"], d[col] * scale, "-o", color=gpu_color(tag), label=tag, markersize=5)
        ax.set_ylabel(ylab)
        ax.set_xlabel("num_stages")
    axes[0].legend(loc="upper left")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_liveness(df: pd.DataFrame, title: str, path: str | Path | None = None) -> Any:
    """Stretch figure: PTX peak-live estimate (32-bit units) vs ptxas registers per thread.

    Left: non-spilling kernels with the identity line (the comparison that matters). Right: all
    kernels, showing how far the PTX peak exceeds the 255-register cap when ptxas must spill.
    """
    style()
    d = df.dropna(subset=["n_regs"])
    spill = d["n_spills"].fillna(0) > 0
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    ns = d[~spill]
    ax.scatter(ns["n_regs"], ns["peak_live_units"], s=14, color=SERIES[0], linewidths=0, alpha=0.7)
    hi = float(max(ns["n_regs"].max(), ns["peak_live_units"].max())) * 1.05
    ax.plot([0, hi], [0, hi], color=INK_2, linewidth=1, linestyle="--")
    r = float(np.corrcoef(ns["n_regs"], ns["peak_live_units"])[0, 1]) if len(ns) > 1 else float("nan")
    ax.set_title(f"Non-spilling kernels (n={len(ns)}, r={r:.3f})")
    ax.set_xlabel("ptxas registers / thread")
    ax.set_ylabel("PTX peak live (32-bit units)")
    ax = axes[1]
    ax.scatter(d.loc[~spill, "n_regs"], d.loc[~spill, "peak_live_units"], s=12, color=SERIES[0], label="no spill",
               linewidths=0, alpha=0.6)
    ax.scatter(d.loc[spill, "n_regs"], d.loc[spill, "peak_live_units"], s=12, color=SERIES[1], label="spills",
               linewidths=0, alpha=0.7)
    ax.axvline(255, color=INK_2, linewidth=1, linestyle=":")
    ax.text(253, ax.get_ylim()[1] * 0.97, "255-register cap ", color=INK_2, fontsize=8, va="top", ha="right")
    ax.set_title("All kernels: spilling kernels pile up at the cap")
    ax.set_xlabel("ptxas registers / thread")
    ax.legend(loc="upper left")
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    return _save(fig, path)


def fig_tuning_time(integ: pd.DataFrame, shape_id: str, path: str | Path | None = None) -> Any:
    """Headline: live triton.autotune wall time, unpruned vs ArchTune top-k (log scale),
    with the chosen kernel's runtime printed next to each bar."""
    style()
    d = integ[integ["shape_id"] == shape_id].copy()
    order = [m for m in ("unpruned", "archtune_top16", "archtune_top8") if m in set(d["mode"])]
    d = d.set_index("mode").loc[order]
    labels = {"unpruned": "Unpruned triton.autotune\n(every config)",
              "archtune_top16": "ArchTune, top-16", "archtune_top8": "ArchTune, top-8"}
    colors = [STRATEGY_COLORS["exhaustive"] if m == "unpruned" else STRATEGY_COLORS["smart"] for m in order]
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    y = np.arange(len(order))
    ax.barh(y, d["tuning_s"], color=colors, height=0.56, edgecolor=SURFACE, linewidth=2)
    ax.set_xscale("log")
    ax.set_yticks(y, [labels[m] for m in order])
    ax.invert_yaxis()
    for i, m in enumerate(order):
        r = d.loc[m]
        ax.text(r["tuning_s"] * 1.12, i, f"{r['tuning_s']:,.1f} s   (kernel {r['time_ms_median']:.2f} ms)",
                va="center", fontsize=9, color=INK)
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
    """Where the exhaustive search's compile time goes: useful configs vs configs that spill
    registers vs configs that exceed shared memory (each config's cold compile counted once)."""
    style()
    d = sweep[sweep["kernel"] == "matmul"]
    cold = d[d["cache_miss"].astype("boolean").fillna(False).astype(bool)].drop_duplicates("cfg_hash")
    spill = cold["n_spills"].fillna(0) > 0
    oor = cold["status"] == "out_of_resources"
    cats = [("Exceeds shared memory\n(fails at load)", oor, SERIES[3]),
            ("Spills registers\n(slow kernel)", spill & ~oor, SERIES[4]),
            ("Useful configs", ~spill & ~oor, SERIES[0])]
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    total = cold["compile_time_s"].sum() / 60
    for i, (lab, mask, col) in enumerate(cats):
        minutes = cold.loc[mask, "compile_time_s"].sum() / 60
        ax.barh(i, minutes, color=col, height=0.56, edgecolor=SURFACE, linewidth=2)
        ax.text(minutes + total * 0.01, i, f"{minutes:.1f} min  ·  {int(mask.sum())} configs  ·  "
                f"{minutes / total:.0%} of compile time", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(cats)), [c[0] for c in cats])
    ax.invert_yaxis()
    ax.set_xlim(0, total * 1.25)
    ax.set_xlabel("minutes of compilation (each matmul config compiled once)")
    ax.set_title(f"Where exhaustive autotuning's compile time goes ({total:.1f} min, {len(cold)} configs)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return _save(fig, path)
