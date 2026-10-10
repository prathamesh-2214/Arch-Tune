"""Compile one config and extract registers, spills, shared memory, IR/assembly and compile time.

The attribute names below are those of Triton 3.2 (pinned, see ``docs/DECISIONS.md``):
``CompiledKernel.n_regs``, ``.n_spills`` (filled by ``_init_handles()``),
``.metadata.shared`` and ``.asm[{"ttir","ttgir","llir","ptx","cubin"}]``. Every access goes
through ``getattr`` fallbacks so a renamed attribute yields ``None`` instead of a crash.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATUSES = ("ok", "compile_error", "launch_error", "out_of_resources", "incorrect")


@dataclass
class CompileResult:
    """Outcome of compiling (and loading) one config."""

    status: str = "ok"
    error_msg: str | None = None
    compile_time_s: float | None = None
    load_time_s: float | None = None
    cache_miss: bool | None = None
    method: str | None = None
    n_regs: int | None = None
    n_spills: int | None = None
    shared_bytes: int | None = None
    asm: dict[str, Any] = field(default_factory=dict)
    compiled: Any = None


def fresh_cache_dir(base: str | os.PathLike[str] | None = None) -> str:
    """Point ``TRITON_CACHE_DIR`` at a new empty directory and return it (spec §8)."""
    path = tempfile.mkdtemp(prefix="archtune_triton_cache_", dir=base)
    os.environ["TRITON_CACHE_DIR"] = path
    return path


def _count_files(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(len(files) for _, _, files in os.walk(path))


def classify_error(exc: BaseException) -> str:
    """Map an exception raised while compiling/loading to a sweep ``status``."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if (name == "OutOfResources" or "out of resource" in msg
            or "too many resources requested for launch" in msg
            or ("shared memory" in msg and "exceed" in msg)):
        return "out_of_resources"
    return "compile_error"


def truncate(msg: str, limit: int = 500) -> str:
    """Truncate an error message for storage."""
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 3] + "..."


def _metadata_get(meta: Any, key: str) -> Any:
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(key)
    return getattr(meta, key, None)


def inspect_compiled(ck: Any) -> dict[str, Any]:
    """Extract n_regs / n_spills / shared_bytes / asm from a compiled kernel object.

    Calls ``_init_handles()`` first (loads the cubin), which populates ``n_regs`` and
    ``n_spills`` on Triton 3.x and raises ``OutOfResources`` when shared memory exceeds the
    device limit.
    """
    init = getattr(ck, "_init_handles", None)
    if callable(init):
        init()
    shared = getattr(ck, "shared", None)
    if shared is None:
        shared = _metadata_get(getattr(ck, "metadata", None), "shared")
    asm = getattr(ck, "asm", None) or {}
    return {
        "n_regs": getattr(ck, "n_regs", None),
        "n_spills": getattr(ck, "n_spills", None),
        "shared_bytes": int(shared) if shared is not None else None,
        "asm": dict(asm),
    }


def compile_config(jit_fn: Any, args: tuple, grid: tuple, cfg_meta: dict[str, Any],
                   cfg_launch: dict[str, Any]) -> CompileResult:
    """Compile one config without launching it and inspect the result (fail-soft).

    Compile time is the wall time of ``JITFunction.warmup`` (compile only, no launch). On a
    Triton without ``warmup`` the first launch is timed instead and a second, cached launch is
    subtracted (``method="first_launch_minus_cached"``).
    """
    res = CompileResult()
    cache = os.environ.get("TRITON_CACHE_DIR")
    before = _count_files(cache)
    try:
        if hasattr(jit_fn, "warmup"):
            res.method = "warmup"
            t0 = time.perf_counter()
            ck = jit_fn.warmup(*args, grid=grid, **cfg_meta, **cfg_launch)
            res.compile_time_s = time.perf_counter() - t0
        else:
            import torch

            res.method = "first_launch_minus_cached"
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            ck = jit_fn[grid](*args, **cfg_meta, **cfg_launch)
            torch.cuda.synchronize()
            first = time.perf_counter() - t0
            t0 = time.perf_counter()
            jit_fn[grid](*args, **cfg_meta, **cfg_launch)
            torch.cuda.synchronize()
            res.compile_time_s = max(first - (time.perf_counter() - t0), 0.0)
    except Exception as exc:  # noqa: BLE001 - fail soft
        res.status, res.error_msg = classify_error(exc), truncate(f"{type(exc).__name__}: {exc}")
        return res
    res.cache_miss = _count_files(cache) > before if cache else None
    res.compiled = ck
    try:
        t0 = time.perf_counter()
        info = inspect_compiled(ck)
        res.load_time_s = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        res.status, res.error_msg = classify_error(exc), truncate(f"{type(exc).__name__}: {exc}")
        # Shared memory is known from metadata even when loading failed.
        meta_shared = _metadata_get(getattr(ck, "metadata", None), "shared")
        res.shared_bytes = int(meta_shared) if meta_shared is not None else None
        res.asm = dict(getattr(ck, "asm", None) or {})
        return res
    res.n_regs, res.n_spills = info["n_regs"], info["n_spills"]
    res.shared_bytes, res.asm = info["shared_bytes"], info["asm"]
    return res


def describe_compiled(ck: Any) -> dict[str, str]:
    """Every public attribute of a compiled kernel (type + short repr), for Phase 0 discovery."""
    out = {}
    for name in sorted(dir(ck)):
        if name.startswith("__"):
            continue
        try:
            val = getattr(ck, name)
        except Exception as exc:  # noqa: BLE001
            out[name] = f"<error {exc}>"
            continue
        r = repr(val)
        out[name] = f"{type(val).__name__}: {r[:160]}"
    return out


def find_cuda_tool(name: str) -> str | None:
    """Locate ``cuobjdump`` / ``nvdisasm`` / ``ptxas``: Triton's bundled binaries first,
    then ``/usr/local/cuda/bin``, then ``PATH``."""
    candidates = []
    try:
        import triton

        candidates.append(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / name)
    except ImportError:
        pass
    candidates.append(Path("/usr/local/cuda/bin") / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which(name)


def disassemble(cubin: bytes, tool: str | None = None) -> str | None:
    """SASS listing of a cubin via ``cuobjdump -sass`` (or ``nvdisasm``); ``None`` if unavailable."""
    tool = tool or find_cuda_tool("cuobjdump") or find_cuda_tool("nvdisasm")
    if tool is None or not cubin:
        return None
    with tempfile.NamedTemporaryFile(suffix=".cubin", delete=False) as fh:
        fh.write(cubin)
        path = fh.name
    try:
        cmd = [tool, "-sass", path] if "cuobjdump" in os.path.basename(tool) else [tool, path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return out.stdout if out.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None
    finally:
        os.unlink(path)


def save_artifacts(asm: dict[str, Any], out_dir: str | os.PathLike[str], cfg_hash: str,
                   keys: tuple[str, ...] = ("ptx", "cubin"), sass: str | None = None) -> str:
    """Write gzip-compressed PTX / cubin (and SASS) for one config; return the directory.

    Artifacts depend only on (GPU arch, kernel, config, specialization), so they are stored by
    config hash and written once.
    """
    d = Path(out_dir) / cfg_hash
    d.mkdir(parents=True, exist_ok=True)
    items = {k: asm.get(k) for k in keys}
    if sass:
        items["sass"] = sass
    for key, val in items.items():
        if val is None:
            continue
        data = val if isinstance(val, (bytes, bytearray)) else str(val).encode()
        target = d / f"{key}.gz"
        if not target.exists():
            with gzip.open(target, "wb") as fh:
                fh.write(data)
    return str(d)


def load_artifact(artifact_dir: str | os.PathLike[str], key: str) -> str | bytes | None:
    """Read back an artifact written by :func:`save_artifacts` (text for IR, bytes for cubin)."""
    path = Path(artifact_dir) / f"{key}.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    return data if key == "cubin" else data.decode()
