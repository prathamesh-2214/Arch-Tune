"""``triton.autotune(prune_configs_by=...)`` integration (spec §11, Phase 8).

Triton 3.2 ``Autotuner.prune_configs`` calls, in order:

1. ``early_config_prune(configs, named_args, **kwargs)`` -> list of configs;
2. if ``perf_model`` is set and more than ``top_k`` configs remain:
   ``perf_model(**named_args, **kwargs, **config.all_kwargs())`` -> estimated time, keeping
   the ``top_k`` lowest.

ArchTune plugs the predictor into both: the early prune drops configs predicted to exceed
shared memory or spill, and the perf model is the roofline score. The pure parts
(:func:`prune_configs`, :func:`perf_model_score`) take duck-typed configs and are CPU-testable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from archtune import predict
from archtune.arch import ArchParams
from archtune.space import LAUNCH_KNOBS


@dataclass
class HookStats:
    """Wall time spent inside the predictor and counts of pruned configs."""

    predictor_time_s: float = 0.0
    calls: int = 0
    n_in: list[int] = field(default_factory=list)
    n_out: list[int] = field(default_factory=list)


def _cfg_dict(tcfg: Any) -> dict[str, Any]:
    """Flatten a ``triton.Config`` (or any object with ``.kwargs`` and launch attributes)."""
    cfg = dict(tcfg.kwargs)
    for knob in LAUNCH_KNOBS:
        val = getattr(tcfg, knob, None)
        if val is not None:
            cfg[knob] = val
    return cfg


def prune_configs(configs: Sequence[Any], M: int, N: int, K: int, arch: ArchParams,
                  reg_model: predict.RegModel = predict.ANALYTIC_REGS,
                  smem_model: predict.SmemModel = predict.ANALYTIC_SMEM,
                  k: int | None = None) -> list[Any]:
    """Smart-k on Triton configs: drop infeasible / spilling, rank by score, keep top *k*."""
    scored = []
    for i, c in enumerate(configs):
        p = predict.predict_matmul(_cfg_dict(c), M, N, K, arch, reg_model, smem_model)
        scored.append((p["feasible"] and not p["spill_pred"], p["score"], i, c))
    keep = [s for s in scored if s[0]] or scored  # fall back to all if everything is dropped
    keep.sort(key=lambda s: (s[1], s[2]))
    out = [s[3] for s in keep]
    return out if k is None else out[:k]


def perf_model_score(arch: ArchParams, reg_model: predict.RegModel = predict.ANALYTIC_REGS,
                     smem_model: predict.SmemModel = predict.ANALYTIC_SMEM) -> Callable[..., float]:
    """A ``perf_model(**kwargs)`` callable returning the predictor score (lower is better)."""

    def perf_model(**kw: Any) -> float:
        cfg = {k: kw[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                  "num_warps", "num_stages") if k in kw}
        if kw.get("maxnreg") is not None:
            cfg["maxnreg"] = kw["maxnreg"]
        return predict.predict_matmul(cfg, kw["M"], kw["N"], kw["K"], arch, reg_model, smem_model)["score"]

    return perf_model


def archtune_prune_configs_by(arch: ArchParams, k: int,
                              reg_model: predict.RegModel = predict.ANALYTIC_REGS,
                              smem_model: predict.SmemModel = predict.ANALYTIC_SMEM,
                              stats: HookStats | None = None) -> dict[str, Any]:
    """Build the ``prune_configs_by`` dict for ``@triton.autotune``.

    ``early_config_prune`` performs the full smart-k policy (filter + rank + top-k), so
    ``perf_model`` only re-sorts when Triton applies it; both are timed into *stats*.
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
    """Run one live ``triton.autotune`` tuning (fresh caches) and time it.

    Returns tuning wall time, the chosen config and its runtime (median ``do_bench``),
    plus predictor stats when *prune_by* came from :func:`archtune_prune_configs_by`.
    """
    import torch

    from archtune import bench, compile_info
    from archtune.kernels import matmul as mm
    from archtune.space import from_triton_config, to_triton_config

    compile_info.fresh_cache_dir()
    tcfgs = [to_triton_config(c) for c in configs]
    stats = prune_by.pop("_stats", None) if prune_by else None
    kernel = mm.make_autotuned(tcfgs, prune_configs_by=prune_by, do_bench=bench.autotune_do_bench)
    a, b, c = mm.make_inputs(M, N, K, seed=seed)
    args = mm.kernel_args(a, b, c)
    grid = lambda meta: (  # noqa: E731
        ((M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"])
        * ((N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"]),)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    kernel[grid](*args)
    torch.cuda.synchronize()
    tuning_s = time.perf_counter() - t0
    best = from_triton_config(kernel.best_config)
    ok, err = bench.check_correct(c, mm.reference(a, b))
    # Warm the GPU (after tuning it may have idled during the last compiles), then time.
    timing = bench.time_config(mm.launcher(a, b, c, best), warmup=bench.POST_IDLE_WARMUP_MS, n_repetitions=3)
    out = {"M": M, "N": N, "K": K, "n_configs_given": len(configs), "tuning_s": tuning_s,
           "best_cfg": best, "correct": ok, "max_abs_err": err, **timing}
    if stats is not None:
        out.update(predictor_time_s=stats.predictor_time_s,
                   n_after_prune=stats.n_out[-1] if stats.n_out else None)
    return out
