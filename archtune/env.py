"""Environment capture: GPU, compute capability, driver, CUDA, torch and Triton versions.

Every result row carries the dict returned by :func:`capture` (spec §7, §12). Rows from
different *stacks* (see :func:`stack_id`) must never be compared with each other.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from typing import Any

#: Map from a substring of ``torch.cuda.get_device_name()`` to a short GPU tag prefix.
_GPU_TAGS = {
    "T4": "t4",
    "A100": "a100",
    "L4": "l4",
    "H100": "h100",
    "V100": "v100",
    "A10G": "a10g",
    "RTX 3050": "rtx3050",
}

#: Columns that identify a software stack. Rows are comparable only within one stack.
STACK_COLUMNS = ("torch_version", "triton_version", "cuda")


#: Set ARCHTUNE_MINI=1 to run every notebook on a tiny workload (debug / smoke mode).
MINI_ENV = "ARCHTUNE_MINI"
PINNED = {"torch": "2.6.0", "triton": "3.2.0"}


def mini_mode() -> bool:
    """True when notebooks should use the mini workload (``ARCHTUNE_MINI=1``)."""
    return os.environ.get(MINI_ENV, "0") == "1"


def pinned_ok(e: dict[str, Any]) -> bool:
    """Whether a captured environment matches the pinned torch/Triton versions."""
    return (str(e.get("torch_version") or "").split("+")[0] == PINNED["torch"]
            and e.get("triton_version") == PINNED["triton"])


def gpu_tag(gpu_name: str, cc_major: int, cc_minor: int) -> str:
    """Return a short, filesystem-safe tag such as ``t4_sm75``.

    Args:
        gpu_name: Device name as reported by the driver.
        cc_major: Compute capability major version.
        cc_minor: Compute capability minor version.
    """
    prefix = None
    # Longest key first so "A100" wins over a hypothetical shorter overlapping key.
    for key in sorted(_GPU_TAGS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(key)}\b", gpu_name):
            prefix = _GPU_TAGS[key]
            break
    if prefix is None:
        prefix = re.sub(r"[^a-z0-9]+", "", gpu_name.lower().replace("nvidia", "")) or "gpu"
    return f"{prefix}_sm{cc_major}{cc_minor}"


def stack_id(row: dict[str, Any]) -> str:
    """Short hash identifying the pinned software stack of a result row."""
    payload = json.dumps({k: str(row.get(k)) for k in STACK_COLUMNS}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _driver_version() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - environment probing must never fail
        return None


def capture() -> dict[str, Any]:
    """Capture the environment as a flat dict suitable for result rows.

    Works without torch/triton installed (fields are then ``None``).
    """
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "hostname": platform.node(),
        "gpu_name": None, "gpu_tag": None, "cc_major": None, "cc_minor": None,
        "sm_count": None, "driver": _driver_version(), "cuda": None,
        "torch_version": None, "triton_version": None,
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            env["gpu_name"] = props.name
            env["cc_major"], env["cc_minor"] = props.major, props.minor
            env["sm_count"] = props.multi_processor_count
            env["gpu_tag"] = gpu_tag(props.name, props.major, props.minor)
    except ImportError:
        pass
    try:
        import triton

        env["triton_version"] = triton.__version__
    except ImportError:
        pass
    env["stack_id"] = stack_id(env)
    return env


def device_properties() -> dict[str, Any]:
    """Every public attribute of ``torch.cuda.get_device_properties(0)`` as a dict."""
    import torch

    props = torch.cuda.get_device_properties(0)
    out: dict[str, Any] = {}
    for name in dir(props):
        if name.startswith("_"):
            continue
        try:
            value = getattr(props, name)
        except Exception:  # noqa: BLE001
            continue
        if not callable(value):
            out[name] = value if isinstance(value, (int, float, str, bool)) else str(value)
    return out


def pip_freeze(path: str | os.PathLike[str]) -> None:
    """Write ``pip freeze`` output to *path* (records the exact numerics stack)."""
    import sys

    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out.stdout)
