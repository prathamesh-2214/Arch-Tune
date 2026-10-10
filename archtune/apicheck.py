"""Verify, on the installed stack, every Triton/PyTorch API the project relies on (spec §12).

:func:`report` returns a JSON-serializable dict; notebook 00 saves it to
``results/<gpu_tag>/api_check.json`` and its facts are copied into ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import inspect
from typing import Any


def _sig(obj: Any) -> str | None:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def _src_contains(obj: Any, *needles: str) -> dict[str, bool] | None:
    try:
        src = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return {n: n in src for n in needles}


def report() -> dict[str, Any]:
    """Collect signatures and feature probes for the pinned Triton/PyTorch."""
    import torch
    import triton
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction
    from triton.testing import do_bench

    out: dict[str, Any] = {"torch": torch.__version__, "triton": triton.__version__,
                           "cuda": torch.version.cuda}
    out["JITFunction.warmup"] = _sig(getattr(JITFunction, "warmup", None)) if hasattr(JITFunction, "warmup") else None
    out["triton.Config.__init__"] = _sig(triton.Config.__init__)
    out["maxnreg_in_Config"] = "maxnreg" in (out["triton.Config.__init__"] or "")
    out["triton.autotune"] = _sig(triton.autotune)
    out["Autotuner.prune_configs"] = _sig(getattr(Autotuner, "prune_configs", None))
    out["prune_configs_calls"] = _src_contains(Autotuner.prune_configs, "early_config_prune(self.configs, self.nargs",
                                               "perf_model(", "all_kwargs()")
    out["do_bench"] = _sig(do_bench)
    out["do_bench_flushes_l2"] = _src_contains(do_bench, "cache.zero_()", "get_empty_cache_for_benchmark",
                                               "clear_cache")
    try:
        from triton.runtime.cache import FileCacheManager

        out["cache_dir_env_read_at_init"] = _src_contains(FileCacheManager.__init__, "TRITON_CACHE_DIR")
    except Exception as exc:  # noqa: BLE001
        out["cache_dir_env_read_at_init"] = f"error: {exc}"
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(torch.cuda.current_device())
        out["triton_device_properties_keys"] = sorted(props)
    except Exception as exc:  # noqa: BLE001
        out["triton_device_properties_keys"] = f"error: {exc}"
    props = torch.cuda.get_device_properties(0)
    out["torch_device_properties_attrs"] = sorted(a for a in dir(props) if not a.startswith("_"))
    try:
        from torch._inductor.kernel import mm_common

        out["inductor_mm_common.mm_configs"] = hasattr(mm_common, "mm_configs")
    except Exception as exc:  # noqa: BLE001
        out["inductor_mm_common.mm_configs"] = f"error: {exc}"
    return out


def compiled_kernel_attrs(ck: Any) -> dict[str, Any]:
    """Locate registers / spills / shared memory / asm on a compiled kernel object."""
    meta = getattr(ck, "metadata", None)
    return {
        "type": type(ck).__name__,
        "has_n_regs": hasattr(ck, "n_regs"), "has_n_spills": hasattr(ck, "n_spills"),
        "metadata_type": type(meta).__name__,
        "metadata_fields": list(getattr(meta, "_fields", [])) or (sorted(meta) if isinstance(meta, dict) else None),
        "asm_keys": sorted((getattr(ck, "asm", None) or {}).keys()),
    }
