"""OPTIONAL, feature-flagged: feed ArchTune pruning into Inductor ``max-autotune`` (Phase 8).

Inductor internals are version-fragile. On torch 2.6 the matmul template configs come from
``torch._inductor.kernel.mm_common.mm_configs`` (a ``functools.partial`` over
``filtered_configs``) and are imported by name into ``torch._inductor.kernel.mm``.
:func:`patch_mm_configs` wraps that function and filters its output with the predictor.
If the attribute is missing on the installed torch, it returns ``False`` with a reason and
does nothing — document that in ``DECISIONS.md`` and stop (spec §11).

Enabled only when the environment variable ``ARCHTUNE_INDUCTOR_PATCH=1`` is set.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

FLAG = "ARCHTUNE_INDUCTOR_PATCH"
_ORIGINAL: dict[str, Any] = {}


def enabled() -> bool:
    """Whether the feature flag is on."""
    return os.environ.get(FLAG, "0") == "1"


def _cfg_from_inductor(c: Any) -> dict[str, Any] | None:
    """Inductor yields ``triton.Config`` objects with BLOCK_M/N/K kwargs."""
    kw = getattr(c, "kwargs", None)
    if not kw or not {"BLOCK_M", "BLOCK_N", "BLOCK_K"} <= set(kw):
        return None
    return {"BLOCK_SIZE_M": kw["BLOCK_M"], "BLOCK_SIZE_N": kw["BLOCK_N"], "BLOCK_SIZE_K": kw["BLOCK_K"],
            "GROUP_SIZE_M": kw.get("GROUP_M", 8), "num_warps": c.num_warps, "num_stages": c.num_stages}


def patch_mm_configs(keep: Callable[[dict[str, Any], int, int, int], bool]) -> tuple[bool, str]:
    """Wrap Inductor's ``mm_configs`` so only configs with ``keep(cfg, m, n, k)`` survive.

    Returns (patched, message).
    """
    if not enabled():
        return False, f"feature flag {FLAG} is not set"
    try:
        from torch._inductor.kernel import mm as mm_mod
        from torch._inductor.kernel import mm_common
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot import Inductor mm modules: {exc}"
    orig = getattr(mm_common, "mm_configs", None)
    if orig is None or not hasattr(mm_mod, "mm_configs"):
        return False, "torch._inductor.kernel.mm_common.mm_configs not found on this torch"
    _ORIGINAL.setdefault("mm_configs", orig)

    def wrapped(m: Any, n: Any, k: Any, *args: Any, **kwargs: Any):
        configs = list(_ORIGINAL["mm_configs"](m, n, k, *args, **kwargs))
        try:
            mi, ni, ki = int(m), int(n), int(k)
        except (TypeError, ValueError):  # symbolic shapes: do not prune
            return configs
        kept = [c for c in configs if (cfg := _cfg_from_inductor(c)) is None or keep(cfg, mi, ni, ki)]
        return kept or configs

    mm_common.mm_configs = wrapped
    mm_mod.mm_configs = wrapped
    return True, "patched torch._inductor.kernel.mm.mm_configs"


def unpatch() -> None:
    """Restore Inductor's original ``mm_configs``."""
    if "mm_configs" in _ORIGINAL:
        from torch._inductor.kernel import mm as mm_mod
        from torch._inductor.kernel import mm_common

        mm_common.mm_configs = _ORIGINAL["mm_configs"]
        mm_mod.mm_configs = _ORIGINAL.pop("mm_configs")


def compile_and_time(M: int, N: int, K: int, seed: int = 0) -> dict[str, Any]:
    """``torch.compile(mode="max-autotune")`` one fp16 matmul; time compile+tune and runtime.

    Clears Dynamo state first so each call re-tunes. Inductor's chosen kernel is logged by
    Inductor itself (``TORCH_LOGS=+inductor`` / autotune logs); here we record wall times.
    """
    import torch
    import torch._dynamo

    from archtune import bench

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
    ok, err = bench.check_correct(out, torch.matmul(a, b))
    timing = bench.time_config(lambda: fn(a, b), warmup=bench.POST_IDLE_WARMUP_MS, n_repetitions=3)
    return {"M": M, "N": N, "K": K, "compile_and_tune_s": compile_s, "correct": ok,
            "max_abs_err": err, "patched": "mm_configs" in _ORIGINAL, **timing}
