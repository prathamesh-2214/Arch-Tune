"""Every tunable parameter of the project, in one place.

Settings are grouped into small frozen dataclasses (``STACK``, ``WORKLOAD``, ``SPACE``, ``BENCH``,
``PREDICTOR``, ``SEARCH``, ``EXPERIMENTS``, ``PATHS``) plus the hardware tables. Other modules read
them when a function runs, not when it is imported, so a change made here, or at runtime with
:func:`configure`, applies everywhere::

    from archtune import config
    config.configure(bench={"rep_ms": 50}, search={"n_seeds": 5})
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping

# --------------------------------------------------------------------------- software stack


@dataclass(frozen=True)
class StackConfig:
    """The pinned torch / Triton pair. Results are only compared within one stack."""

    torch: str = "2.6.0"
    triton: str = "3.2.0"
    stack_columns: tuple[str, ...] = ("torch_version", "triton_version", "cuda")


# --------------------------------------------------------------------------- workloads


@dataclass(frozen=True)
class WorkloadConfig:
    """Which shapes are measured. Matmuls follow ``C[M, N] = A[M, K] @ B[K, N]``."""

    model_id: str = "HuggingFaceTB/SmolLM2-360M"
    token_counts: tuple[int, ...] = (16, 128, 512, 2048)
    large_shapes: tuple[tuple[int, int, int], ...] = ((4096, 4096, 4096), (8192, 2048, 2048))
    split_seed: int = 0
    train_frac: float = 0.5
    # A mini run keeps the large reference matmul, one decode-sized one and the smallest RMSNorm.
    mini_matmul_shape_ids: tuple[str, ...] = ("mm_M4096_N4096_K4096", "mm_M128_N1600_K960")
    mini_rmsnorm_shapes: int = 1
    mini_rmsnorm_configs: int = 6


# --------------------------------------------------------------------------- config space


@dataclass(frozen=True)
class SpaceConfig:
    """The knobs autotuning chooses between."""

    matmul_knobs: Mapping[str, tuple[int, ...]] = field(default_factory=lambda: MappingProxyType({
        "BLOCK_SIZE_M": (32, 64, 128, 256),
        "BLOCK_SIZE_N": (32, 64, 128, 256),
        "BLOCK_SIZE_K": (32, 64),
        "GROUP_SIZE_M": (8,),
        "num_warps": (2, 4, 8),
        "num_stages": (2, 3, 4, 5),
    }))
    rmsnorm_warps: tuple[int, ...] = (1, 2, 4, 8, 16)
    rmsnorm_min_block: int = 128
    # Knobs passed to the launcher rather than to the kernel as constexprs.
    launch_knobs: tuple[str, ...] = ("num_warps", "num_stages", "maxnreg")
    # Twelve matmul configs for quick runs: (BM, BN, BK, warps, stages). The last two are there on
    # purpose: a 256x128 tile on 2 warps spills, and 5 x 64 KB stage buffers do not fit on Ampere.
    mini_matmul: tuple[tuple[int, int, int, int, int], ...] = (
        (32, 32, 32, 2, 2), (64, 64, 32, 4, 3), (128, 128, 32, 4, 3), (128, 64, 64, 4, 4),
        (64, 128, 32, 8, 2), (128, 256, 64, 8, 3), (256, 128, 32, 8, 4), (32, 64, 64, 2, 5),
        (128, 128, 64, 8, 5), (64, 32, 32, 4, 2), (256, 128, 32, 2, 2), (256, 256, 64, 4, 5),
    )


# --------------------------------------------------------------------------- measurement


@dataclass(frozen=True)
class BenchConfig:
    """Correctness tolerances and timing budgets (milliseconds for ``do_bench``)."""

    rtol: float = 1e-2
    atol: float = 1e-2
    # Triton's own defaults, used for final and live timings.
    warmup_ms: int = 25
    rep_ms: int = 100
    # The sweep only has to rank configs, so it uses a smaller budget; even a 5 ms kernel still
    # gets ~8 timed runs. The noise check uses the same budget.
    sweep_warmup_ms: int = 10
    sweep_rep_ms: int = 40
    # A laptop GPU drops its clocks while the CPU compiles; the first timing after that read
    # 25-33% slow, so freshly compiled kernels run this long before they are timed.
    post_idle_warmup_ms: int = 200
    slow_compile_s: float = 0.5
    noise_runs: int = 20
    tie_factor: float = 2.0
    final_repetitions: int = 3
    flush_every: int = 25
    progress_every: int = 100


# --------------------------------------------------------------------------- predictor


@dataclass(frozen=True)
class PredictorConfig:
    """Constants of the cost model. The fitted values live in ``results/<gpu>/models.json``."""

    elem_bytes: int = 2          # fp16 operands
    acc_bytes: int = 4           # fp32 accumulator
    min_active_warps: int = 4    # below this, latency is hard to hide ...
    low_occupancy_penalty: float = 1.5   # ... so the time score is multiplied by this
    # Analytic register model: regs = a*acc + b*operand + s*operand*(stages-1) + c
    analytic_regs: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 32.0)
    # Dropping a config that would not have spilled can lose the best kernel, while missing a
    # spill only wastes one compile, so false alarms count double when fitting the threshold.
    spill_false_alarm_weight: float = 2.0
    # Shared-memory formula variants tried when fitting.
    smem_stage_offsets: tuple[int, ...] = (-1, 0, 1)
    smem_stage_caps: tuple[int | None, ...] = (None, 1, 2)
    smem_epilogue_bytes: tuple[int, ...] = (0, 2, 4)
    smem_combines: tuple[str, ...] = ("max", "sum")
    rmsnorm_chunk_penalty: float = 0.05
    rmsnorm_eps: float = 1e-5


# --------------------------------------------------------------------------- search replay


@dataclass(frozen=True)
class SearchConfig:
    """How search strategies are replayed and judged."""

    ks: tuple[int | None, ...] = (1, 2, 4, 8, 16, 32, 64, None)   # None = every config
    n_seeds: int = 20
    target: float = 0.95


# --------------------------------------------------------------------------- experiments


@dataclass(frozen=True)
class ExperimentConfig:
    """Run ids, shapes and sizes of the individual experiments."""

    sweep_run_id: str = "sweep_v1"
    maxnreg_run_id: str = "maxnreg_v1"
    stages_run_id: str = "stages_v1"
    large_shape_id: str = "mm_M4096_N4096_K4096"
    maxnreg_caps: tuple[int | None, ...] = (None, 255, 192, 168, 128, 96, 64)
    num_stages: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    stages_top_configs: int = 5
    spotcheck_shapes: int = 3
    spotcheck_k: int = 8
    integration_shapes: tuple[str, ...] = ("mm_M4096_N4096_K4096", "mm_M2048_N2560_K960", "mm_M128_N1600_K960")
    integration_ks: tuple[int, ...] = (8, 16)
    estimate_sample: int = 10


# --------------------------------------------------------------------------- paths and environment


@dataclass(frozen=True)
class PathsConfig:
    """Folder names, relative to the project root."""

    results: str = "results"
    results_mini: str = "results_mini"
    preflight: str = "results_preflight"
    figures: str = "figures"
    docs: str = "docs"


MINI_ENV = "ARCHTUNE_MINI"              # 1 = tiny workload
RESULTS_ENV = "ARCHTUNE_RESULTS"        # override the results folder
INDUCTOR_PATCH_ENV = "ARCHTUNE_INDUCTOR_PATCH"   # 1 = allow patching Inductor


# --------------------------------------------------------------------------- hardware tables

#: Device-name fragment -> short tag prefix (results go to results/<prefix>_sm<cc>).
GPU_TAGS: Mapping[str, str] = MappingProxyType({
    "T4": "t4", "A100": "a100", "L4": "l4", "H100": "h100", "V100": "v100", "A10G": "a10g",
    "RTX 3050": "rtx3050",
})

HW_SOURCES: Mapping[str, str] = MappingProxyType({
    "cuda_guide": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications",
    "occupancy": "CUDA Occupancy Calculator / cuda_occupancy.h (register and shared-memory allocation rules)",
    "t4": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
    "a100": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
    "l4": "https://www.nvidia.com/en-us/data-center/l4/",
})

_SM70_PLUS = dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                  reg_alloc_unit=256, warp_alloc_granularity=4, max_threads_per_block=1024)

#: Limits per compute capability (CUDA programming guide + occupancy calculator). Bytes.
CC_LIMITS: Mapping[tuple[int, int], Mapping[str, Any]] = MappingProxyType({
    (7, 5): {**_SM70_PLUS, "smem_per_sm": 65536, "smem_per_block_optin": 65536, "smem_reserved_per_block": 0,
             "smem_alloc_unit": 256, "max_threads_per_sm": 1024, "max_blocks_per_sm": 16, "has_cp_async": False},
    (8, 0): {**_SM70_PLUS, "smem_per_sm": 167936, "smem_per_block_optin": 166912, "smem_reserved_per_block": 1024,
             "smem_alloc_unit": 128, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "has_cp_async": True},
    (8, 6): {**_SM70_PLUS, "smem_per_sm": 102400, "smem_per_block_optin": 101376, "smem_reserved_per_block": 1024,
             "smem_alloc_unit": 128, "max_threads_per_sm": 1536, "max_blocks_per_sm": 16, "has_cp_async": True},
    (8, 9): {**_SM70_PLUS, "smem_per_sm": 102400, "smem_per_block_optin": 101376, "smem_reserved_per_block": 1024,
             "smem_alloc_unit": 128, "max_threads_per_sm": 1536, "max_blocks_per_sm": 24, "has_cp_async": True},
    (9, 0): {**_SM70_PLUS, "smem_per_sm": 233472, "smem_per_block_optin": 232448, "smem_reserved_per_block": 1024,
             "smem_alloc_unit": 128, "max_threads_per_sm": 2048, "max_blocks_per_sm": 32, "has_cp_async": True},
})

#: Datasheet numbers: SMs, dense FP16 tensor TFLOP/s (FP32 accumulate), DRAM GB/s. The L4 sheet
#: quotes 242 TFLOP/s with sparsity, so dense is half; the A100 is the 40 GB part Colab offers.
GPU_REFERENCE: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    "t4": dict(name="Tesla T4", cc=(7, 5), sm_count=40, peak_fp16_tflops=65.0, mem_bw_gbs=320.0, source="t4"),
    "a100": dict(name="A100", cc=(8, 0), sm_count=108, peak_fp16_tflops=312.0, mem_bw_gbs=1555.0, source="a100"),
    "l4": dict(name="L4", cc=(8, 9), sm_count=58, peak_fp16_tflops=121.0, mem_bw_gbs=300.0, source="l4"),
})

#: FP16 tensor FLOPs per clock per SM, used to estimate peak throughput for GPUs without a
#: datasheet entry. Back-calculated from the datasheets above; GeForce Ampere (sm_86) runs FP32-
#: accumulate tensor math at half rate (RTX 3090: 71e12 / (82 SMs x 1.70 GHz) ~ 512).
FP16_FLOPS_PER_CLK_PER_SM: Mapping[tuple[int, int], int] = MappingProxyType({
    (7, 5): 1024, (8, 0): 2048, (8, 6): 512, (8, 9): 1024, (9, 0): 4096,
})

# --------------------------------------------------------------------------- plot style

PALETTE: Mapping[str, Any] = MappingProxyType({
    "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3df",
    "series": ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"),
    "sequential": ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"),
    # A series keeps its colour in every figure.
    "strategy": MappingProxyType({"random": "#2a78d6", "smart": "#eb6834", "smart_analytic": "#eb6834",
                                  "exhaustive": "#1baf7a"}),
    "gpu_order": ("rtx3050_sm86", "t4_sm75", "a100_sm80", "l4_sm89"),
})

# --------------------------------------------------------------------------- the live settings

STACK = StackConfig()
WORKLOAD = WorkloadConfig()
SPACE = SpaceConfig()
BENCH = BenchConfig()
PREDICTOR = PredictorConfig()
SEARCH = SearchConfig()
EXPERIMENTS = ExperimentConfig()
PATHS = PathsConfig()

_SECTIONS = {"stack": "STACK", "workload": "WORKLOAD", "space": "SPACE", "bench": "BENCH",
             "predictor": "PREDICTOR", "search": "SEARCH", "experiments": "EXPERIMENTS", "paths": "PATHS"}


def configure(**sections: Mapping[str, Any]) -> None:
    """Change settings at runtime, e.g. ``configure(bench={"rep_ms": 50})``."""
    for name, changes in sections.items():
        attr = _SECTIONS[name]
        globals()[attr] = replace(globals()[attr], **changes)


def mini_mode() -> bool:
    """True when the tiny workload is requested (``ARCHTUNE_MINI=1``)."""
    return os.environ.get(MINI_ENV, "0") == "1"


__all__ = [
    "StackConfig", "WorkloadConfig", "SpaceConfig", "BenchConfig", "PredictorConfig", "SearchConfig",
    "ExperimentConfig", "PathsConfig", "STACK", "WORKLOAD", "SPACE", "BENCH", "PREDICTOR", "SEARCH",
    "EXPERIMENTS", "PATHS", "GPU_TAGS", "HW_SOURCES", "CC_LIMITS", "GPU_REFERENCE",
    "FP16_FLOPS_PER_CLK_PER_SM", "PALETTE", "MINI_ENV", "RESULTS_ENV", "INDUCTOR_PATCH_ENV",
    "configure", "mini_mode", "replace",
]
