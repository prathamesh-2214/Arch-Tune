"""Fused RMSNorm Triton kernel, config-parameterized by ``BLOCK_SIZE`` and ``num_warps``.

``y = x / sqrt(mean(x^2) + eps) * w`` per row. One program per row; the row is processed in
``BLOCK_SIZE`` chunks (two passes: sum of squares, then normalize), so any power-of-two
``BLOCK_SIZE`` is valid. fp16 in/out, fp32 reduction.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import triton
import triton.language as tl

from archtune.space import split_launch

EPS = 1e-5


@triton.jit
def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, n_cols, eps,
                   BLOCK_SIZE: tl.constexpr):
    """RMSNorm of one row per program."""
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_x
    y_row = y_ptr + row * stride_y
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + cols, mask=cols < n_cols, other=0.0).to(tl.float32)
        acc += x * x
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / n_cols + eps)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(y_row + cols, (x * rstd * w).to(tl.float16), mask=mask)


def make_inputs(rows: int, cols: int, seed: int = 0, device: str = "cuda"
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed, seeded fp16 ``x[rows, cols]``, weight ``w[cols]`` and output ``y``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((rows, cols), device=device, dtype=torch.float16, generator=gen)
    w = torch.rand((cols,), device=device, dtype=torch.float16, generator=gen) + 0.5
    y = torch.empty_like(x)
    return x, w, y


def kernel_args(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor) -> tuple:
    """Positional (non-constexpr) arguments of :func:`rmsnorm_kernel`."""
    return (x, w, y, x.stride(0), y.stride(0), x.shape[1], EPS)


def grid(rows: int, cols: int, cfg: dict[str, Any]) -> tuple[int]:
    """One program per row."""
    return (rows,)


def launcher(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor,
             cfg: dict[str, Any]) -> Callable[[], Any]:
    """Zero-arg callable launching the kernel with config *cfg*."""
    meta, launch = split_launch(cfg)
    args = kernel_args(x, w, y)
    g = grid(x.shape[0], x.shape[1], cfg)
    return lambda: rmsnorm_kernel[g](*args, **meta, **launch)


def reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """PyTorch reference RMSNorm (fp32 math, fp16 output)."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS) * w.float()).half()


def bytes_moved(rows: int, cols: int) -> int:
    """Minimum unique DRAM bytes (fp16): read x once, write y, read w. The second pass over
    x usually hits L2 and is not counted."""
    return 2 * (2 * rows * cols + cols)
