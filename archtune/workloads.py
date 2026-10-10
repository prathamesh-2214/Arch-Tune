"""Workloads (spec §5): SmolLM2-360M projection shapes, large reference shapes, RMSNorm shapes.

Model dimensions are derived programmatically from the Hugging Face ``config.json``; nothing
about the model is hardcoded here. Matmul convention: ``C[M, N] = A[M, K] @ B[K, N]``.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
TOKEN_COUNTS: tuple[int, ...] = (16, 128, 512, 2048)
LARGE_SHAPES: tuple[tuple[int, int, int], ...] = ((4096, 4096, 4096), (8192, 2048, 2048))
SPLIT_SEED = 0


@dataclass(frozen=True)
class ModelDims:
    """Transformer dimensions needed to derive projection shapes."""

    hidden: int
    intermediate: int
    heads: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class Shape:
    """One workload shape. For RMSNorm, ``M`` = rows (tokens), ``N`` = cols, ``K`` = 0."""

    kernel: str
    shape_id: str
    M: int
    N: int
    K: int
    roles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain dict for result rows."""
        d = asdict(self)
        d["roles"] = ",".join(self.roles)
        return d


def load_model_config(model_id: str = MODEL_ID, cache_dir: str | os.PathLike[str] | None = None,
                      timeout: float = 30.0) -> dict[str, Any]:
    """Fetch ``config.json`` from the Hugging Face Hub (cached as JSON in *cache_dir*)."""
    cache = Path(cache_dir) / f"{model_id.replace('/', '__')}.config.json" if cache_dir else None
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https URL
        cfg = json.loads(resp.read().decode())
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def derive_dims(cfg: dict[str, Any]) -> ModelDims:
    """Derive :class:`ModelDims` from a Llama-style HF config dict."""
    hidden = int(cfg["hidden_size"])
    heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg.get("num_key_value_heads", heads))
    head_dim = int(cfg.get("head_dim") or hidden // heads)
    return ModelDims(hidden=hidden, intermediate=int(cfg["intermediate_size"]),
                     heads=heads, kv_heads=kv_heads, head_dim=head_dim)


def projection_nk(dims: ModelDims) -> dict[str, tuple[int, int]]:
    """(N, K) of each linear projection. Q, K and V are fused into one QKV matmul;
    MLP up and gate have identical shapes and are one entry."""
    qkv_n = (dims.heads + 2 * dims.kv_heads) * dims.head_dim
    return {
        "qkv": (qkv_n, dims.hidden),
        "attn_out": (dims.hidden, dims.heads * dims.head_dim),
        "mlp_up_gate": (dims.intermediate, dims.hidden),
        "mlp_down": (dims.hidden, dims.intermediate),
    }


def shape_id(kernel: str, M: int, N: int, K: int) -> str:
    """Canonical shape id, e.g. ``mm_M128_N1600_K960`` or ``rms_R128_C960``."""
    return f"mm_M{M}_N{N}_K{K}" if kernel == "matmul" else f"rms_R{M}_C{N}"


def matmul_shapes(dims: ModelDims, tokens: tuple[int, ...] = TOKEN_COUNTS,
                  large: tuple[tuple[int, int, int], ...] = LARGE_SHAPES) -> list[Shape]:
    """Deduplicated matmul shapes: tokens x projections, plus large reference shapes."""
    roles: dict[tuple[int, int, int], list[str]] = {}
    for m in tokens:
        for role, (n, k) in projection_nk(dims).items():
            roles.setdefault((m, n, k), []).append(f"{role}@{m}")
    for mnk in large:
        roles.setdefault(mnk, []).append("large_ref")
    return [Shape("matmul", shape_id("matmul", *mnk), *mnk, roles=tuple(r))
            for mnk, r in sorted(roles.items())]


def rmsnorm_shapes(dims: ModelDims, tokens: tuple[int, ...] = TOKEN_COUNTS) -> list[Shape]:
    """RMSNorm over (tokens x hidden) for each token count."""
    return [Shape("rmsnorm", shape_id("rmsnorm", m, dims.hidden, 0), m, dims.hidden, 0,
                  roles=(f"rmsnorm@{m}",)) for m in tokens]


def train_test_split(shape_ids: list[str], seed: int = SPLIT_SEED,
                     train_frac: float = 0.5) -> dict[str, str]:
    """Seeded, order-independent split: maps shape_id -> ``"train"`` or ``"test"``."""
    ids = sorted(set(shape_ids))
    random.Random(seed).shuffle(ids)
    n_train = int(round(len(ids) * train_frac))
    return {sid: ("train" if i < n_train else "test") for i, sid in enumerate(ids)}


def all_shapes(model_cfg: dict[str, Any]) -> list[Shape]:
    """Matmul + RMSNorm shapes for a model config."""
    dims = derive_dims(model_cfg)
    return matmul_shapes(dims) + rmsnorm_shapes(dims)


#: Shapes kept in mini (smoke) runs: the large reference shape (used by notebooks 01/05/07)
#: and one decode-ish projection, plus the smallest RMSNorm shape.
MINI_MATMUL_SHAPE_IDS = ("mm_M4096_N4096_K4096", "mm_M128_N1600_K960")


def mini_shapes(shapes: list[Shape]) -> list[Shape]:
    """Subset of *shapes* for mini runs (keeps notebooks' hard-coded shape ids valid)."""
    mm = [s for s in shapes if s.kernel == "matmul" and s.shape_id in MINI_MATMUL_SHAPE_IDS]
    rms = sorted((s for s in shapes if s.kernel == "rmsnorm"), key=lambda s: s.M)[:1]
    return mm + rms
