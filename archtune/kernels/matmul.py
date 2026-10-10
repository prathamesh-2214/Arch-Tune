"""Triton matmul from the official tutorial (03-matrix-multiplication), config-parameterized.

https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

fp16 inputs, fp32 accumulation, fp16 output. Grouped ("super-grouping") tile order with
``GROUP_SIZE_M`` for L2 reuse. All loads are masked, so any M/N/K is valid for any config.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import triton
import triton.language as tl

from archtune.space import split_launch


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """C = A @ B for one (BLOCK_SIZE_M x BLOCK_SIZE_N) tile of C per program."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def grid(M: int, N: int, cfg: dict[str, Any]) -> tuple[int]:
    """1-D launch grid: one program per output tile."""
    return (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)


def make_inputs(M: int, N: int, K: int, seed: int = 0, device: str = "cuda"
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed, seeded fp16 inputs ``a[M,K]``, ``b[K,N]`` and an output buffer ``c[M,N]``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    a = torch.randn((M, K), device=device, dtype=torch.float16, generator=gen)
    b = torch.randn((K, N), device=device, dtype=torch.float16, generator=gen)
    c = torch.empty((M, N), device=device, dtype=torch.float16)
    return a, b, c


def kernel_args(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> tuple:
    """Positional (non-constexpr) arguments of :func:`matmul_kernel`."""
    M, K = a.shape
    _, N = b.shape
    return (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1))


def launcher(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
             cfg: dict[str, Any]) -> Callable[[], Any]:
    """Return a zero-arg callable that launches the kernel with config *cfg*."""
    meta, launch = split_launch(cfg)
    args = kernel_args(a, b, c)
    g = grid(a.shape[0], b.shape[1], cfg)
    return lambda: matmul_kernel[g](*args, **meta, **launch)


def matmul(a: torch.Tensor, b: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    """Compute ``a @ b`` with a specific config."""
    c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
    launcher(a, b, c, cfg)()
    return c


def reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch reference result, computed in **fp32** (returned as fp32).

    An fp16 ``torch.matmul`` is not a trustworthy reference: cuBLAS may use reduced-precision
    (fp16) split-K reductions (``allow_fp16_reduced_precision_reduction`` is on by default), which
    on skinny shapes such as 128x1600x960 on a T4 deviates from the exact result by more than the
    tolerance, while our kernel accumulates in fp32 (DECISIONS D-022). fp32 ``torch.matmul`` does
    not use TF32 unless ``torch.backends.cuda.matmul.allow_tf32`` is enabled (off by default).
    """
    return torch.matmul(a.float(), b.float())


def flops(M: int, N: int, K: int) -> int:
    """Floating-point operations of one matmul."""
    return 2 * M * N * K


def make_autotuned(configs: list, prune_configs_by: dict | None = None,
                   key: tuple[str, ...] = ("M", "N", "K"), do_bench=None):
    """Wrap the kernel in ``triton.autotune`` (used by :mod:`archtune.triton_hook`).

    A fresh ``JITFunction`` wrapped in a fresh ``triton.autotune`` is returned on every call,
    so timing a tuning run starts from empty in-memory kernel and autotuner caches (the
    on-disk cache is controlled by ``TRITON_CACHE_DIR``; see :mod:`archtune.compile_info`).
    """
    fresh = triton.jit(matmul_kernel.fn)
    return triton.autotune(configs=configs, key=list(key), prune_configs_by=prune_configs_by,
                           do_bench=do_bench)(fresh)
