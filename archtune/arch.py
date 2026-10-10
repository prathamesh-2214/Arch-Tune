"""Architecture parameters: reference table, live device query and verification.

Two layers:

* :data:`CC_LIMITS` — per-compute-capability limits from the CUDA C Programming Guide,
  "Technical Specifications per Compute Capability" table, and the occupancy-calculator
  allocation rules (register allocation unit, warp allocation granularity, shared-memory
  allocation unit, reserved shared memory per block).
* :data:`GPU_REFERENCE` — per-product numbers (SM count, peak dense FP16 tensor throughput,
  DRAM bandwidth) from the NVIDIA datasheets cited in :data:`SOURCES`.

:func:`live` overlays whatever the driver reports on top of the reference values, and
:func:`verify` lists every disagreement so it can be recorded in ``docs/DECISIONS.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

SOURCES = {
    "cuda_guide": "https://docs.nvidia.com/cuda/cuda-c-programming-guide/#features-and-technical-specifications",
    "occupancy": "CUDA Occupancy Calculator / cuda_occupancy.h (register and shared-memory allocation rules)",
    "t4": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf",
    "a100": "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf",
    "l4": "https://www.nvidia.com/en-us/data-center/l4/",
}

#: Per compute capability limits. Shared memory in bytes.
CC_LIMITS: dict[tuple[int, int], dict[str, Any]] = {
    (7, 5): dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                 reg_alloc_unit=256, warp_alloc_granularity=4,
                 smem_per_sm=65536, smem_per_block_optin=65536, smem_reserved_per_block=0,
                 smem_alloc_unit=256, max_threads_per_sm=1024, max_blocks_per_sm=16,
                 max_threads_per_block=1024, has_cp_async=False),
    (8, 0): dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                 reg_alloc_unit=256, warp_alloc_granularity=4,
                 smem_per_sm=167936, smem_per_block_optin=166912, smem_reserved_per_block=1024,
                 smem_alloc_unit=128, max_threads_per_sm=2048, max_blocks_per_sm=32,
                 max_threads_per_block=1024, has_cp_async=True),
    (8, 6): dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                 reg_alloc_unit=256, warp_alloc_granularity=4,
                 smem_per_sm=102400, smem_per_block_optin=101376, smem_reserved_per_block=1024,
                 smem_alloc_unit=128, max_threads_per_sm=1536, max_blocks_per_sm=16,
                 max_threads_per_block=1024, has_cp_async=True),
    (8, 9): dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                 reg_alloc_unit=256, warp_alloc_granularity=4,
                 smem_per_sm=102400, smem_per_block_optin=101376, smem_reserved_per_block=1024,
                 smem_alloc_unit=128, max_threads_per_sm=1536, max_blocks_per_sm=24,
                 max_threads_per_block=1024, has_cp_async=True),
    (9, 0): dict(regs_per_sm=65536, max_regs_per_thread=255, max_regs_per_block=65536,
                 reg_alloc_unit=256, warp_alloc_granularity=4,
                 smem_per_sm=233472, smem_per_block_optin=232448, smem_reserved_per_block=1024,
                 smem_alloc_unit=128, max_threads_per_sm=2048, max_blocks_per_sm=32,
                 max_threads_per_block=1024, has_cp_async=True),
}

#: Per product: SM count, peak dense FP16 tensor-core TFLOP/s (FP32 accumulate), DRAM GB/s.
#: L4 datasheet quotes 242 TFLOP/s FP16 *with sparsity*; dense is half (121).
#: A100 is the 40 GB SXM/PCIe part Colab provides (1,555 GB/s); the 80 GB part has 2,039 GB/s.
GPU_REFERENCE: dict[str, dict[str, Any]] = {
    "t4": dict(name="Tesla T4", cc=(7, 5), sm_count=40, peak_fp16_tflops=65.0, mem_bw_gbs=320.0, source="t4"),
    "a100": dict(name="A100", cc=(8, 0), sm_count=108, peak_fp16_tflops=312.0, mem_bw_gbs=1555.0, source="a100"),
    "l4": dict(name="L4", cc=(8, 9), sm_count=58, peak_fp16_tflops=121.0, mem_bw_gbs=300.0, source="l4"),
}


#: Dense FP16 tensor-core FLOPs per clock per SM with FP32 accumulation, used only to *estimate*
#: peak throughput for GPUs without a datasheet entry. Derived from the datasheet entries above
#: (T4: 65e12 / (40 SM x 1.59 GHz) ~ 1024; A100: 312e12 / (108 x 1.41 GHz) ~ 2048;
#: L4: 121e12 / (58 x 2.04 GHz) ~ 1024). GeForce Ampere (sm_86) runs FP32-accumulate HMMA at
#: half rate (RTX 3090: 71e12 / (82 x 1.70 GHz) ~ 512).
FP16_FLOPS_PER_CLK_PER_SM: dict[tuple[int, int], int] = {
    (7, 5): 1024, (8, 0): 2048, (8, 6): 512, (8, 9): 1024, (9, 0): 4096,
}


def estimate_peaks(cc: tuple[int, int], sm_count: int, sm_clock_khz: float | None,
                   mem_clock_khz: float | None, mem_bus_bits: int | None) -> dict[str, float]:
    """Estimate peak dense FP16 TFLOP/s and DRAM GB/s from clocks (``nan`` when unknown).

    Bandwidth = 2 (double data rate) x memory clock x bus width / 8, which is the formula the
    CUDA ``deviceQuery`` sample uses. Both values are estimates; datasheet values are preferred.
    """
    per_clk = FP16_FLOPS_PER_CLK_PER_SM.get(_nearest_cc(cc))
    tflops = (per_clk * sm_count * sm_clock_khz * 1e3 / 1e12) if per_clk and sm_clock_khz else float("nan")
    gbs = (2 * mem_clock_khz * 1e3 * mem_bus_bits / 8 / 1e9) if mem_clock_khz and mem_bus_bits else float("nan")
    return {"peak_fp16_tflops": tflops, "mem_bw_gbs": gbs}


@dataclass(frozen=True)
class ArchParams:
    """Architecture parameters consumed by :mod:`archtune.predict`.

    All shared-memory quantities are in bytes; throughput in TFLOP/s and GB/s.
    """

    name: str
    cc: tuple[int, int]
    sm_count: int
    regs_per_sm: int
    max_regs_per_thread: int
    max_regs_per_block: int
    reg_alloc_unit: int
    warp_alloc_granularity: int
    smem_per_sm: int
    smem_per_block_optin: int
    smem_reserved_per_block: int
    smem_alloc_unit: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    max_threads_per_block: int
    has_cp_async: bool
    peak_fp16_tflops: float
    mem_bw_gbs: float
    provenance: dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def max_warps_per_sm(self) -> int:
        """Maximum resident warps per SM."""
        return self.max_threads_per_sm // 32

    @property
    def sm_tag(self) -> str:
        """``sm_75``-style tag."""
        return f"sm_{self.cc[0]}{self.cc[1]}"

    def to_dict(self) -> dict[str, Any]:
        """Plain dict (for result rows / JSON)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchParams":
        """Inverse of :meth:`to_dict` (JSON turns the ``cc`` tuple into a list)."""
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["cc"] = tuple(d["cc"])
        return cls(**d)


def _nearest_cc(cc: tuple[int, int]) -> tuple[int, int]:
    """Closest known compute capability not newer than *cc* (falls back to the oldest)."""
    known = sorted(CC_LIMITS)
    older = [k for k in known if k <= cc]
    return older[-1] if older else known[0]


def reference(key: str) -> ArchParams:
    """Reference parameters for a product key in :data:`GPU_REFERENCE` (``t4``, ``a100``, ``l4``)."""
    gpu = GPU_REFERENCE[key]
    limits = CC_LIMITS[gpu["cc"]]
    prov = {k: "cuda_guide/occupancy" for k in limits}
    prov.update({k: gpu["source"] for k in ("sm_count", "peak_fp16_tflops", "mem_bw_gbs")})
    return ArchParams(name=gpu["name"], cc=gpu["cc"], sm_count=gpu["sm_count"],
                      peak_fp16_tflops=gpu["peak_fp16_tflops"], mem_bw_gbs=gpu["mem_bw_gbs"],
                      provenance=prov, **limits)


def from_cc(cc: tuple[int, int], sm_count: int, name: str = "unknown",
            peak_fp16_tflops: float = float("nan"), mem_bw_gbs: float = float("nan")) -> ArchParams:
    """Build parameters for an arbitrary GPU from its compute capability."""
    limits = CC_LIMITS[_nearest_cc(cc)]
    return ArchParams(name=name, cc=cc, sm_count=sm_count, peak_fp16_tflops=peak_fp16_tflops,
                      mem_bw_gbs=mem_bw_gbs, provenance={k: "cuda_guide/occupancy" for k in limits},
                      **limits)


def reference_for_name(gpu_name: str) -> str | None:
    """Return the :data:`GPU_REFERENCE` key matching a device name, if any."""
    import re

    for key, gpu in GPU_REFERENCE.items():
        token = gpu["name"].split()[-1]
        if re.search(rf"\b{re.escape(token)}\b", gpu_name):
            return key
    return None


def _live_values() -> dict[str, Any]:
    """Raw values reported by torch and Triton for device 0 (keys use ArchParams names)."""
    import torch

    props = torch.cuda.get_device_properties(0)
    vals: dict[str, Any] = {"name": props.name, "cc": (props.major, props.minor),
                            "sm_count": props.multi_processor_count}
    torch_map = {
        "regs_per_multiprocessor": "regs_per_sm",
        "max_threads_per_multi_processor": "max_threads_per_sm",
        "shared_memory_per_block_optin": "smem_per_block_optin",
        "shared_memory_per_multiprocessor": "smem_per_sm",
    }
    for attr, key in torch_map.items():
        if hasattr(props, attr):
            vals[key] = int(getattr(props, attr))
    try:  # Triton driver utilities (names verified per pinned version in DECISIONS.md)
        from triton.runtime import driver

        tprops = driver.active.utils.get_device_properties(torch.cuda.current_device())
        if "max_shared_mem" in tprops:
            vals.setdefault("smem_per_block_optin", int(tprops["max_shared_mem"]))
        if "max_num_regs" in tprops:
            vals.setdefault("regs_per_sm", int(tprops["max_num_regs"]))
        if "multiprocessor_count" in tprops:
            vals.setdefault("sm_count", int(tprops["multiprocessor_count"]))
        for k in ("sm_clock_rate", "mem_clock_rate", "mem_bus_width"):
            if k in tprops:
                vals["_" + k] = tprops[k]
    except Exception:  # noqa: BLE001 - optional source
        pass
    return vals


def live() -> ArchParams:
    """Parameters for the current device: live query overlaid on the reference table."""
    vals = _live_values()
    key = reference_for_name(vals["name"])
    base = reference(key) if key is not None and GPU_REFERENCE[key]["cc"] == vals["cc"] \
        else from_cc(vals["cc"], vals["sm_count"], name=vals["name"])
    overrides = {k: v for k, v in vals.items() if k in ArchParams.__dataclass_fields__}
    prov = dict(base.provenance)
    prov.update({k: "live" for k in overrides if k not in ("name", "cc")})
    if math.isnan(base.peak_fp16_tflops) or math.isnan(base.mem_bw_gbs):
        est = estimate_peaks(vals["cc"], vals["sm_count"], vals.get("_sm_clock_rate"),
                             vals.get("_mem_clock_rate"), vals.get("_mem_bus_width"))
        for k, v in est.items():
            if math.isnan(getattr(base, k)):
                overrides[k] = v
                prov[k] = "estimated from clocks (arch.estimate_peaks)"
    return replace(base, provenance=prov, **overrides)


def verify(ref: ArchParams, observed: ArchParams) -> list[dict[str, Any]]:
    """List every field where *observed* (usually :func:`live`) differs from *ref*."""
    diffs = []
    for name in ArchParams.__dataclass_fields__:
        if name in ("provenance", "name"):
            continue
        a, b = getattr(ref, name), getattr(observed, name)
        if a != b and not (isinstance(a, float) and a != a and b != b):
            diffs.append({"field": name, "reference": a, "observed": b})
    return diffs
