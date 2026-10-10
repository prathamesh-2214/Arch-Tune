"""Correctness check and timing protocol (spec §8).

Pure helpers (:func:`coefficient_of_variation`, :func:`is_tie`, :func:`summarize_times`) are
CPU-testable; :func:`check_correct`, :func:`time_config` and :func:`noise_check` need a GPU.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Sequence

import numpy as np

RTOL = 1e-2
ATOL = 1e-2
#: do_bench budget in ms (Triton defaults: warmup=25, rep=100) — used for final / live timings.
WARMUP_MS = 25
REP_MS = 100
#: Smaller budget for the exhaustive sweep (ranking ~7,000 configs). do_bench repeats the kernel
#: ~rep/estimate times, so even a 5 ms kernel gets ~8 timed iterations. The noise check is run
#: with the same budget so ties are judged at the precision actually used (DECISIONS D-021).
SWEEP_WARMUP_MS = 10
SWEEP_REP_MS = 40
#: Warm-up used instead when the GPU has just been idle (e.g. during a cold compile, which can
#: take seconds). Laptop/T4 GPUs drop their clocks when idle; the first measurement after idle
#: read ~25-33% slow in our runs (DECISIONS D-025), so the kernel is run for this long first.
POST_IDLE_WARMUP_MS = 200


def coefficient_of_variation(samples: Sequence[float]) -> float:
    """Sample standard deviation / mean."""
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2 or arr.mean() == 0:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean())


def is_tie(t_a: float, t_b: float, cv: float, factor: float = 2.0) -> bool:
    """True when two times differ by less than ``factor x cv`` (relative to the faster one)."""
    lo, hi = sorted((t_a, t_b))
    return (hi - lo) / lo < factor * cv


def summarize_times(samples: Sequence[float]) -> dict[str, float]:
    """Median and p10/p90 of a list of per-repetition times (ms)."""
    arr = np.asarray(samples, dtype=float)
    return {"time_ms_median": float(np.median(arr)),
            "time_ms_p10": float(np.quantile(arr, 0.1)),
            "time_ms_p90": float(np.quantile(arr, 0.9)),
            "n_reps": int(arr.size)}


def check_correct(out: Any, ref: Any, rtol: float = RTOL, atol: float = ATOL) -> tuple[bool, float]:
    """Compare a kernel output with the PyTorch reference; returns (correct, max_abs_err)."""
    import torch

    diff = (out.float() - ref.float()).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)) and bool(
        torch.isfinite(out.float()).all())
    return ok, max_err


def _median_ms(fn: Callable[[], Any], warmup: int, rep: int) -> float:
    """Median of one ``do_bench`` call (handles scalar or list return for one quantile)."""
    from triton.testing import do_bench

    r = do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5])
    return float(r[0] if isinstance(r, (list, tuple)) else r)


def autotune_do_bench(kernel_call: Callable[[], Any], quantiles: Sequence[float] | None = None) -> Any:
    """``do_bench`` hook for ``triton.autotune`` (Triton 3.2 ``do_bench=`` argument).

    The autotuner compiles each config right before benchmarking it, and the GPU drops its clocks
    during the compile. A 200 ms warm-up per config (D-026) makes its choice robust; it is used
    for pruned and unpruned tuning alike, so the comparison stays fair.
    """
    from triton.testing import do_bench

    return do_bench(kernel_call, warmup=POST_IDLE_WARMUP_MS, rep=REP_MS, quantiles=quantiles)


def time_config(fn: Callable[[], Any], warmup: int = WARMUP_MS, rep: int = REP_MS,
                n_repetitions: int = 1) -> dict[str, float]:
    """Time *fn* with ``triton.testing.do_bench``.

    ``do_bench`` (Triton 3.2) clears L2 between iterations by zeroing a 256 MB buffer and
    returns the requested quantiles of per-iteration times. With ``n_repetitions > 1`` the
    medians of independent ``do_bench`` calls are summarized instead (final evaluation set).
    """
    import torch
    from triton.testing import do_bench

    t0 = time.perf_counter()
    torch.cuda.synchronize()
    if n_repetitions == 1:
        med, p10, p90 = do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.1, 0.9])
        out = {"time_ms_median": float(med), "time_ms_p10": float(p10),
               "time_ms_p90": float(p90), "n_reps": 1}
    else:
        out = summarize_times([_median_ms(fn, warmup, rep) for _ in range(n_repetitions)])
    torch.cuda.synchronize()
    out["bench_time_s"] = time.perf_counter() - t0
    return out


def noise_check(fn: Callable[[], Any], n: int = 20, warmup: int = WARMUP_MS,
                rep: int = REP_MS) -> dict[str, Any]:
    """Time one config *n* times (independent ``do_bench`` medians) and report the CV.

    Differences smaller than ``2 x cv`` are reported as ties (:func:`is_tie`). One extra
    measurement is taken first and discarded: on a T4 the first one after idle runs while the
    GPU is still raising its clocks (seen as a 35% CV in the first Colab preflight).
    """
    discarded = _median_ms(fn, max(warmup, POST_IDLE_WARMUP_MS), rep)  # ramp clocks up first
    meds = [_median_ms(fn, warmup, rep) for _ in range(n)]
    return {"samples_ms": meds, "discarded_first_ms": discarded, "mean_ms": statistics.fmean(meds),
            "cv": coefficient_of_variation(meds), "tie_threshold": 2 * coefficient_of_variation(meds)}
