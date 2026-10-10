"""Config spaces (spec §6) and stable config hashing.

A *config* is a plain ``dict`` of knob name -> int. Launch-only knobs (``num_warps``,
``num_stages``, ``maxnreg``) live in the same dict as the ``tl.constexpr`` meta-parameters;
:func:`split_launch` separates them when launching.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterable

Config = dict[str, int]

#: Knobs passed to the launcher rather than to the kernel as constexprs.
LAUNCH_KNOBS = ("num_warps", "num_stages", "maxnreg")

MATMUL_KNOBS: dict[str, tuple[int, ...]] = {
    "BLOCK_SIZE_M": (32, 64, 128, 256),
    "BLOCK_SIZE_N": (32, 64, 128, 256),
    "BLOCK_SIZE_K": (32, 64),
    "GROUP_SIZE_M": (8,),
    "num_warps": (2, 4, 8),
    "num_stages": (2, 3, 4, 5),
}

RMSNORM_WARPS: tuple[int, ...] = (1, 2, 4, 8, 16)
RMSNORM_MIN_BLOCK = 128


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable 12-hex-digit hash of a config: SHA-1 of sorted-key compact JSON.

    ``None`` values (e.g. ``maxnreg=None``) are dropped so that an uncapped config hashes the
    same with or without the key.
    """
    clean = {k: v for k, v in cfg.items() if v is not None}
    payload = json.dumps(clean, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _product(knobs: dict[str, Iterable[int]]) -> list[Config]:
    names = list(knobs)
    return [dict(zip(names, vals)) for vals in itertools.product(*knobs.values())]


def matmul_space(knobs: dict[str, tuple[int, ...]] | None = None) -> list[Config]:
    """Enumerate the matmul config space (384 configs with the default knobs).

    The tutorial kernel masks every load, so no divisibility rule removes configs; resource
    limits are deliberately *not* used for filtering (that is what the study measures).
    """
    return _product(knobs or MATMUL_KNOBS)


def next_pow2(x: int) -> int:
    """Smallest power of two >= ``x`` (``x >= 1``)."""
    return 1 << (int(x) - 1).bit_length()


def rmsnorm_space(cols: int, min_block: int = RMSNORM_MIN_BLOCK,
                  warps: tuple[int, ...] = RMSNORM_WARPS) -> list[Config]:
    """RMSNorm configs: ``BLOCK_SIZE`` = powers of two from ``min_block`` to ``next_pow2(cols)``.

    The kernel loops over the row in ``BLOCK_SIZE`` chunks, so blocks smaller than the row
    are valid; ``next_pow2(cols)`` is the single-pass block.
    """
    top = next_pow2(cols)
    blocks, b = [], min(min_block, top)
    while b <= top:
        blocks.append(b)
        b *= 2
    return [{"BLOCK_SIZE": bs, "num_warps": w} for bs in blocks for w in warps]


def split_launch(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a config into (constexpr meta-parameters, launch options)."""
    meta = {k: v for k, v in cfg.items() if k not in LAUNCH_KNOBS}
    launch = {k: v for k, v in cfg.items() if k in LAUNCH_KNOBS and v is not None}
    return meta, launch


def to_triton_config(cfg: dict[str, Any]):
    """Convert to ``triton.Config`` (imports Triton lazily)."""
    import triton

    meta, launch = split_launch(cfg)
    return triton.Config(meta, **launch)


def from_triton_config(tcfg: Any) -> Config:
    """Convert a ``triton.Config`` back to a flat config dict."""
    cfg = dict(tcfg.kwargs)
    for knob in LAUNCH_KNOBS:
        val = getattr(tcfg, knob, None)
        if val is not None:
            cfg[knob] = int(val)
    return cfg


def mini_matmul_space() -> list[Config]:
    """12 deterministic matmul configs for mini (smoke) runs.

    Deliberately includes the edge cases a full sweep must survive: a config that exceeds
    shared memory when multi-buffered (out_of_resources on sm_80+), a config that spills registers,
    small and large tiles, and every ``num_warps`` value.
    """
    picks = [
        (32, 32, 32, 2, 2), (64, 64, 32, 4, 3), (128, 128, 32, 4, 3), (128, 64, 64, 4, 4),
        (64, 128, 32, 8, 2), (128, 256, 64, 8, 3), (256, 128, 32, 8, 4), (32, 64, 64, 2, 5),
        (128, 128, 64, 8, 5), (64, 32, 32, 4, 2),
        (256, 128, 32, 2, 2),   # 512 accumulator regs/thread -> spills
        (256, 256, 64, 4, 5),   # 5 x 64 KB buffers: out_of_resources on sm_80+; fits on sm_75,
                                # where Triton 3.2 single-buffers (64 KB, DECISIONS D-022)
    ]
    return [{"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n, "BLOCK_SIZE_K": k, "GROUP_SIZE_M": 8,
             "num_warps": w, "num_stages": s} for m, n, k, w, s in picks]
