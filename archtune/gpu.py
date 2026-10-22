"""Everything that touches the GPU or the installed toolchain.

* environment capture (GPU, driver, CUDA, torch, Triton) and the stack id results are grouped by;
* architecture parameters: limits per compute capability, datasheet numbers, the live device;
* the Triton kernels, behind a small registry (``KERNELS["matmul"]``, ``KERNELS["rmsnorm"]``);
* compiling one config and reading what the compiler decided (registers, spills, shared memory,
  PTX / cubin / SASS);
* correctness checks and timing;
* a record of which Triton / PyTorch APIs exist on the installed versions.

torch and Triton are imported lazily or guarded, so the rest of the package (predictor, search
replay, PTX analysis) imports and runs on a machine without a GPU.
"""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from archtune import config

try:  # the kernels need Triton; everything else in this module does not
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only machines
    triton = tl = None


# =========================================================================== environment


def mini_mode() -> bool:
    """True when the tiny workload is requested (``ARCHTUNE_MINI=1``)."""
    return config.mini_mode()


def pinned_ok(e: dict[str, Any]) -> bool:
    """Whether a captured environment matches the pinned torch/Triton versions."""
    return (str(e.get("torch_version") or "").split("+")[0] == config.STACK.torch
            and e.get("triton_version") == config.STACK.triton)


def gpu_tag(gpu_name: str, cc_major: int, cc_minor: int) -> str:
    """Short, filesystem-safe tag such as ``t4_sm75`` or ``rtx3050_sm86``."""
    prefix = None
    for key in sorted(config.GPU_TAGS, key=len, reverse=True):  # longest match wins
        if re.search(rf"\b{re.escape(key)}\b", gpu_name):
            prefix = config.GPU_TAGS[key]
            break
    if prefix is None:
        prefix = re.sub(r"[^a-z0-9]+", "", gpu_name.lower().replace("nvidia", "")) or "gpu"
    return f"{prefix}_sm{cc_major}{cc_minor}"


def stack_id(row: dict[str, Any]) -> str:
    """Short hash of the software stack of a result row."""
    payload = json.dumps({k: str(row.get(k)) for k in config.STACK.stack_columns}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _driver_version() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=driver_version", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True)
        return out.stdout.strip().splitlines()[0]
    except Exception:  # noqa: BLE001 - probing the environment must never fail
        return None


def capture() -> dict[str, Any]:
    """The environment as a flat dict for result rows (fields are ``None`` without torch/Triton)."""
    env: dict[str, Any] = {
        "python": platform.python_version(), "hostname": platform.node(),
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
    if triton is not None:
        env["triton_version"] = triton.__version__
    env["stack_id"] = stack_id(env)
    return env


def device_properties() -> dict[str, Any]:
    """Every public attribute of ``torch.cuda.get_device_properties(0)``."""
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
    """Write ``pip freeze`` to *path*."""
    out = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    Path(path).write_text(out.stdout, encoding="utf-8")


# =========================================================================== architecture


@dataclass(frozen=True)
class ArchParams:
    """What the predictor needs to know about a GPU. Bytes, TFLOP/s and GB/s."""

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
        """Plain dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ArchParams":
        """Inverse of :meth:`to_dict` (JSON turns the ``cc`` tuple into a list)."""
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        d["cc"] = tuple(d["cc"])
        return cls(**d)


def _nearest_cc(cc: tuple[int, int]) -> tuple[int, int]:
    """Closest known compute capability not newer than *cc* (or the oldest one)."""
    known = sorted(config.CC_LIMITS)
    older = [k for k in known if k <= tuple(cc)]
    return older[-1] if older else known[0]


def estimate_peaks(cc: tuple[int, int], sm_count: int, sm_clock_khz: float | None,
                   mem_clock_khz: float | None, mem_bus_bits: int | None) -> dict[str, float]:
    """Peak FP16 TFLOP/s and DRAM GB/s estimated from clocks (``nan`` when unknown).

    Bandwidth = 2 (double data rate) x memory clock x bus width / 8, as in CUDA's ``deviceQuery``.
    """
    per_clk = config.FP16_FLOPS_PER_CLK_PER_SM.get(_nearest_cc(cc))
    tflops = (per_clk * sm_count * sm_clock_khz * 1e3 / 1e12) if per_clk and sm_clock_khz else float("nan")
    gbs = (2 * mem_clock_khz * 1e3 * mem_bus_bits / 8 / 1e9) if mem_clock_khz and mem_bus_bits else float("nan")
    return {"peak_fp16_tflops": tflops, "mem_bw_gbs": gbs}


def reference(key: str) -> ArchParams:
    """Datasheet parameters for ``t4``, ``a100`` or ``l4``."""
    gpu = config.GPU_REFERENCE[key]
    limits = dict(config.CC_LIMITS[gpu["cc"]])
    prov = {k: "cuda_guide/occupancy" for k in limits}
    prov.update({k: gpu["source"] for k in ("sm_count", "peak_fp16_tflops", "mem_bw_gbs")})
    return ArchParams(name=gpu["name"], cc=gpu["cc"], sm_count=gpu["sm_count"],
                      peak_fp16_tflops=gpu["peak_fp16_tflops"], mem_bw_gbs=gpu["mem_bw_gbs"],
                      provenance=prov, **limits)


def from_cc(cc: tuple[int, int], sm_count: int, name: str = "unknown",
            peak_fp16_tflops: float = float("nan"), mem_bw_gbs: float = float("nan")) -> ArchParams:
    """Parameters for any GPU from its compute capability."""
    limits = dict(config.CC_LIMITS[_nearest_cc(cc)])
    return ArchParams(name=name, cc=tuple(cc), sm_count=sm_count, peak_fp16_tflops=peak_fp16_tflops,
                      mem_bw_gbs=mem_bw_gbs, provenance={k: "cuda_guide/occupancy" for k in limits},
                      **limits)


def reference_for_name(gpu_name: str) -> str | None:
    """The datasheet key matching a device name, if there is one."""
    for key, gpu in config.GPU_REFERENCE.items():
        token = gpu["name"].split()[-1]
        if re.search(rf"\b{re.escape(token)}\b", gpu_name):
            return key
    return None


def _live_values() -> dict[str, Any]:
    """Raw values torch and Triton report for device 0, under ArchParams names."""
    import torch

    props = torch.cuda.get_device_properties(0)
    vals: dict[str, Any] = {"name": props.name, "cc": (props.major, props.minor),
                            "sm_count": props.multi_processor_count}
    torch_map = {"regs_per_multiprocessor": "regs_per_sm",
                 "max_threads_per_multi_processor": "max_threads_per_sm",
                 "shared_memory_per_block_optin": "smem_per_block_optin",
                 "shared_memory_per_multiprocessor": "smem_per_sm"}
    for attr, key in torch_map.items():
        if hasattr(props, attr):
            vals[key] = int(getattr(props, attr))
    try:
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
    """Parameters of the current device: what it reports, filled in from the tables."""
    vals = _live_values()
    key = reference_for_name(vals["name"])
    base = reference(key) if key is not None and config.GPU_REFERENCE[key]["cc"] == vals["cc"] \
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
                prov[k] = "estimated from clocks (gpu.estimate_peaks)"
    return replace(base, provenance=prov, **overrides)


def verify(ref: ArchParams, observed: ArchParams) -> list[dict[str, Any]]:
    """Every field where *observed* (usually :func:`live`) differs from *ref*."""
    diffs = []
    for name in ArchParams.__dataclass_fields__:
        if name in ("provenance", "name"):
            continue
        a, b = getattr(ref, name), getattr(observed, name)
        if a != b and not (isinstance(a, float) and a != a and b != b):
            diffs.append({"field": name, "reference": a, "observed": b})
    return diffs


# =========================================================================== kernels


def split_launch(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split a config into (constexpr meta-parameters, launch options)."""
    knobs = config.SPACE.launch_knobs
    meta = {k: v for k, v in cfg.items() if k not in knobs}
    launch = {k: v for k, v in cfg.items() if k in knobs and v is not None}
    return meta, launch


if triton is not None:

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
        """C = A @ B, one (BLOCK_SIZE_M x BLOCK_SIZE_N) tile of C per program.

        The matmul from Triton's tutorial 03: fp16 in, fp32 accumulate, grouped tile order for
        L2 reuse, every load masked so any M/N/K works with any config.
        """
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

    @triton.jit
    def rmsnorm_kernel(x_ptr, w_ptr, y_ptr, stride_x, stride_y, n_cols, eps,
                       BLOCK_SIZE: tl.constexpr):
        """RMSNorm of one row per program, ``y = x / sqrt(mean(x^2) + eps) * w``.

        The row is read in ``BLOCK_SIZE`` chunks (sum of squares, then normalize), so any
        power-of-two block works. fp16 in and out, fp32 reduction.
        """
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

else:  # pragma: no cover
    matmul_kernel = rmsnorm_kernel = None


class Kernel:
    """One tunable Triton kernel: inputs, launch grid, launcher and a PyTorch reference.

    Subclasses fill in the kernel-specific parts. ``inputs(shape)`` / ``reference_of(inputs)`` /
    ``shape_grid(shape, cfg)`` take a workload :class:`~archtune.model.Shape`, so the sweep can
    treat every kernel the same way. A new kernel is a subclass plus an entry in :data:`KERNELS`.
    """

    name = "kernel"
    jit_fn: Any = None

    def make_inputs(self, *dims: int, seed: int = 0, device: str = "cuda") -> tuple:
        raise NotImplementedError

    def kernel_args(self, *tensors: Any) -> tuple:
        raise NotImplementedError

    def grid(self, d0: int, d1: int, cfg: dict[str, Any]) -> tuple[int, ...]:
        raise NotImplementedError

    def reference(self, *tensors: Any) -> Any:
        raise NotImplementedError

    def shape_dims(self, shape: Any) -> tuple[int, ...]:
        """The arguments of :meth:`make_inputs` for a workload shape."""
        raise NotImplementedError

    def launcher(self, *tensors_and_cfg: Any, jit_fn: Any = None) -> Callable[[], Any]:
        """A zero-argument callable launching the kernel with a config (the last argument)."""
        *tensors, cfg = tensors_and_cfg
        meta, launch = split_launch(cfg)
        args = self.kernel_args(*tensors)
        g = self.grid(tensors[0].shape[0], self._grid_dim1(tensors), cfg)
        fn = jit_fn or self.jit_fn
        return lambda: fn[g](*args, **meta, **launch)

    def _grid_dim1(self, tensors: Sequence[Any]) -> int:
        return tensors[0].shape[1]

    def fresh_jit(self) -> Any:
        """A new ``JITFunction`` of the same kernel, with an empty in-memory compile cache."""
        return triton.jit(self.jit_fn.fn)

    def inputs(self, shape: Any, seed: int = 0) -> tuple:
        return self.make_inputs(*self.shape_dims(shape), seed=seed)

    def reference_of(self, inputs: tuple) -> Any:
        return self.reference(inputs[0], inputs[1])

    def shape_grid(self, shape: Any, cfg: dict[str, Any]) -> tuple[int, ...]:
        return self.grid(shape.M, shape.N, cfg)


class MatmulKernel(Kernel):
    """fp16 ``C[M, N] = A[M, K] @ B[K, N]`` with fp32 accumulation."""

    name = "matmul"
    jit_fn = matmul_kernel

    def make_inputs(self, M: int, N: int, K: int, seed: int = 0, device: str = "cuda") -> tuple:
        """Seeded fp16 ``a[M,K]``, ``b[K,N]`` and an output buffer ``c[M,N]``."""
        import torch

        gen = torch.Generator(device=device).manual_seed(seed)
        a = torch.randn((M, K), device=device, dtype=torch.float16, generator=gen)
        b = torch.randn((K, N), device=device, dtype=torch.float16, generator=gen)
        c = torch.empty((M, N), device=device, dtype=torch.float16)
        return a, b, c

    def kernel_args(self, a: Any, b: Any, c: Any) -> tuple:
        M, K = a.shape
        _, N = b.shape
        return (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                c.stride(0), c.stride(1))

    def grid(self, M: int, N: int, cfg: dict[str, Any]) -> tuple[int]:
        """One program per output tile."""
        return (triton.cdiv(M, cfg["BLOCK_SIZE_M"]) * triton.cdiv(N, cfg["BLOCK_SIZE_N"]),)

    def _grid_dim1(self, tensors: Sequence[Any]) -> int:
        return tensors[1].shape[1]

    def reference(self, a: Any, b: Any) -> Any:
        """``a @ b`` computed in fp32.

        An fp16 ``torch.matmul`` is not a safe reference: cuBLAS may reduce in fp16 on skinny
        shapes and land further from the exact result than the tolerance, while this kernel
        accumulates in fp32. fp32 ``torch.matmul`` does not use TF32 unless that is switched on.
        """
        return a.float() @ b.float()

    def shape_dims(self, shape: Any) -> tuple[int, int, int]:
        return shape.M, shape.N, shape.K

    def matmul(self, a: Any, b: Any, cfg: dict[str, Any]) -> Any:
        """``a @ b`` with one specific config."""
        import torch

        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=torch.float16)
        self.launcher(a, b, c, cfg)()
        return c

    @staticmethod
    def flops(M: int, N: int, K: int) -> int:
        return 2 * M * N * K

    def make_autotuned(self, configs: list, prune_configs_by: dict | None = None,
                       key: tuple[str, ...] = ("M", "N", "K"), do_bench: Any = None) -> Any:
        """The kernel wrapped in ``triton.autotune``.

        Built from a fresh ``JITFunction`` every time, so a timed tuning run starts with empty
        in-memory kernel and autotuner caches; the on-disk cache is ``TRITON_CACHE_DIR``.
        """
        return triton.autotune(configs=configs, key=list(key), prune_configs_by=prune_configs_by,
                               do_bench=do_bench)(self.fresh_jit())


class RMSNormKernel(Kernel):
    """Fused RMSNorm over rows of ``x[rows, cols]``."""

    name = "rmsnorm"
    jit_fn = rmsnorm_kernel

    @property
    def eps(self) -> float:
        return config.PREDICTOR.rmsnorm_eps

    def make_inputs(self, rows: int, cols: int, seed: int = 0, device: str = "cuda") -> tuple:
        """Seeded fp16 ``x[rows, cols]``, weight ``w[cols]`` and output ``y``."""
        import torch

        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn((rows, cols), device=device, dtype=torch.float16, generator=gen)
        w = torch.rand((cols,), device=device, dtype=torch.float16, generator=gen) + 0.5
        return x, w, torch.empty_like(x)

    def kernel_args(self, x: Any, w: Any, y: Any) -> tuple:
        return (x, w, y, x.stride(0), y.stride(0), x.shape[1], self.eps)

    def grid(self, rows: int, cols: int, cfg: dict[str, Any]) -> tuple[int]:
        """One program per row."""
        return (rows,)

    def reference(self, x: Any, w: Any) -> Any:
        """RMSNorm in fp32, returned as fp16."""
        import torch

        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * w.float()).half()

    def shape_dims(self, shape: Any) -> tuple[int, int]:
        return shape.M, shape.N

    @staticmethod
    def bytes_moved(rows: int, cols: int) -> int:
        """DRAM bytes: x read once, y written, w read (the second pass over x is an L2 hit)."""
        return 2 * (2 * rows * cols + cols)


MATMUL = MatmulKernel()
RMSNORM = RMSNormKernel()
#: Kernel name -> implementation. Sweep rows store the name in their ``kernel`` column.
KERNELS: dict[str, Kernel] = {k.name: k for k in (MATMUL, RMSNORM)}


def kernel(name: str) -> Kernel:
    """Look up a registered kernel."""
    try:
        return KERNELS[name]
    except KeyError:
        raise ValueError(f"unknown kernel {name!r}; registered: {sorted(KERNELS)}") from None


# =========================================================================== compiling

STATUSES = ("ok", "compile_error", "launch_error", "out_of_resources", "incorrect")


@dataclass
class CompileResult:
    """Outcome of compiling (and loading) one config.

    On Triton 3.2 the compiled kernel has ``n_regs`` and ``n_spills`` (set once the binary is
    loaded), ``metadata.shared`` and ``asm`` with every IR stage plus the cubin. Fields are read
    with ``getattr``, so a renamed attribute in another Triton shows up as ``None``.
    """

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
    """Point ``TRITON_CACHE_DIR`` at a new empty directory, so the next compiles are cold."""
    path = tempfile.mkdtemp(prefix="archtune_triton_cache_", dir=base)
    os.environ["TRITON_CACHE_DIR"] = path
    return path


def _count_files(path: str | None) -> int:
    if not path or not os.path.isdir(path):
        return 0
    return sum(len(files) for _, _, files in os.walk(path))


def classify_error(exc: BaseException) -> str:
    """Map an exception from compiling/loading to a sweep ``status``."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if (name == "OutOfResources" or "out of resource" in msg
            or "too many resources requested for launch" in msg
            or ("shared memory" in msg and "exceed" in msg)):
        return "out_of_resources"
    return "compile_error"


def truncate(msg: str, limit: int = 500) -> str:
    """Shorten an error message for storage."""
    msg = " ".join(str(msg).split())
    return msg if len(msg) <= limit else msg[: limit - 3] + "..."


def _metadata_get(meta: Any, key: str) -> Any:
    if meta is None:
        return None
    if isinstance(meta, dict):
        return meta.get(key)
    return getattr(meta, key, None)


def inspect_compiled(ck: Any) -> dict[str, Any]:
    """n_regs / n_spills / shared_bytes / asm of a compiled kernel.

    ``_init_handles()`` loads the cubin first; that fills in ``n_regs`` and ``n_spills`` and raises
    ``OutOfResources`` when the kernel needs more shared memory than the device has.
    """
    init = getattr(ck, "_init_handles", None)
    if callable(init):
        init()
    shared = getattr(ck, "shared", None)
    if shared is None:
        shared = _metadata_get(getattr(ck, "metadata", None), "shared")
    return {"n_regs": getattr(ck, "n_regs", None), "n_spills": getattr(ck, "n_spills", None),
            "shared_bytes": int(shared) if shared is not None else None,
            "asm": dict(getattr(ck, "asm", None) or {})}


def compile_config(jit_fn: Any, args: tuple, grid: tuple, cfg_meta: dict[str, Any],
                   cfg_launch: dict[str, Any]) -> CompileResult:
    """Compile one config without launching it and inspect the result. Never raises.

    Compile time is the wall time of ``JITFunction.warmup``. Without ``warmup``, the first launch
    is timed and a second, cached launch subtracted (``method="first_launch_minus_cached"``).
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
        meta_shared = _metadata_get(getattr(ck, "metadata", None), "shared")  # known even if loading failed
        res.shared_bytes = int(meta_shared) if meta_shared is not None else None
        res.asm = dict(getattr(ck, "asm", None) or {})
        return res
    res.n_regs, res.n_spills = info["n_regs"], info["n_spills"]
    res.shared_bytes, res.asm = info["shared_bytes"], info["asm"]
    return res


def compile_kernel(k: Kernel, inputs: tuple, cfg: dict[str, Any], jit_fn: Any = None) -> CompileResult:
    """:func:`compile_config` for a registered kernel and its input tensors."""
    meta, launch = split_launch(cfg)
    g = k.grid(inputs[0].shape[0], k._grid_dim1(inputs), cfg)
    return compile_config(jit_fn or k.jit_fn, k.kernel_args(*inputs), g, meta, launch)


def describe_compiled(ck: Any) -> dict[str, str]:
    """Every public attribute of a compiled kernel (useful when Triton renames something)."""
    out = {}
    for name in sorted(dir(ck)):
        if name.startswith("__"):
            continue
        try:
            val = getattr(ck, name)
        except Exception as exc:  # noqa: BLE001
            out[name] = f"<error {exc}>"
            continue
        out[name] = f"{type(val).__name__}: {repr(val)[:160]}"
    return out


def find_cuda_tool(name: str) -> str | None:
    """``cuobjdump`` / ``nvdisasm`` / ``ptxas``: Triton's bundled copy, then CUDA, then ``PATH``."""
    candidates = []
    if triton is not None:
        candidates.append(Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / name)
    candidates.append(Path("/usr/local/cuda/bin") / name)
    for c in candidates:
        if c.exists():
            return str(c)
    return shutil.which(name)


def disassemble(cubin: bytes | None, tool: str | None = None) -> str | None:
    """SASS of a cubin via ``cuobjdump -sass`` (or ``nvdisasm``); ``None`` if unavailable."""
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
    """Write gzipped PTX / cubin (and SASS) of one config, once per config hash; return the folder."""
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
    """Read back an artifact from :func:`save_artifacts` (text for IR, bytes for the cubin)."""
    path = Path(artifact_dir) / f"{key}.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    return data if key == "cubin" else data.decode()


# =========================================================================== timing
#
# Kernels are timed with triton.testing.do_bench, which clears L2 between runs, keeping the median.
# A laptop GPU drops its clocks while the CPU compiles, so a kernel timed straight after its
# compile reads slow unless it runs for BENCH.post_idle_warmup_ms first.


def coefficient_of_variation(samples: Sequence[float]) -> float:
    """Sample standard deviation / mean."""
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2 or arr.mean() == 0:
        return float("nan")
    return float(arr.std(ddof=1) / arr.mean())


def is_tie(t_a: float, t_b: float, cv: float, factor: float | None = None) -> bool:
    """True when two times differ by less than ``factor x cv`` of the faster one."""
    factor = config.BENCH.tie_factor if factor is None else factor
    lo, hi = sorted((t_a, t_b))
    return (hi - lo) / lo < factor * cv


def summarize_times(samples: Sequence[float]) -> dict[str, float]:
    """Median and p10/p90 of per-repetition times (ms)."""
    arr = np.asarray(samples, dtype=float)
    return {"time_ms_median": float(np.median(arr)), "time_ms_p10": float(np.quantile(arr, 0.1)),
            "time_ms_p90": float(np.quantile(arr, 0.9)), "n_reps": int(arr.size)}


def check_correct(out: Any, ref: Any, rtol: float | None = None, atol: float | None = None
                  ) -> tuple[bool, float]:
    """Compare a kernel output with the reference; returns (correct, max_abs_err)."""
    import torch

    rtol = config.BENCH.rtol if rtol is None else rtol
    atol = config.BENCH.atol if atol is None else atol
    diff = (out.float() - ref.float()).abs()
    max_err = float(diff.max().item()) if diff.numel() else 0.0
    ok = bool(torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)) and bool(
        torch.isfinite(out.float()).all())
    return ok, max_err


def _median_ms(fn: Callable[[], Any], warmup: int, rep: int) -> float:
    from triton.testing import do_bench

    r = do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5])
    return float(r[0] if isinstance(r, (list, tuple)) else r)


def autotune_do_bench(kernel_call: Callable[[], Any], quantiles: Sequence[float] | None = None) -> Any:
    """``do_bench`` with the post-idle warm-up, for ``triton.autotune(do_bench=...)``.

    Triton 3.2's ``autotune`` decorator accepts ``do_bench`` but does not pass it on to the
    ``Autotuner``, so on that version the autotuner keeps its default benchmark.
    """
    from triton.testing import do_bench

    return do_bench(kernel_call, warmup=config.BENCH.post_idle_warmup_ms, rep=config.BENCH.rep_ms,
                    quantiles=quantiles)


def time_config(fn: Callable[[], Any], warmup: int | None = None, rep: int | None = None,
                n_repetitions: int = 1) -> dict[str, float]:
    """Time *fn* with ``do_bench``; with ``n_repetitions > 1``, summarize several medians."""
    import torch
    from triton.testing import do_bench

    warmup = config.BENCH.warmup_ms if warmup is None else warmup
    rep = config.BENCH.rep_ms if rep is None else rep
    t0 = time.perf_counter()
    torch.cuda.synchronize()
    if n_repetitions == 1:
        med, p10, p90 = do_bench(fn, warmup=warmup, rep=rep, quantiles=[0.5, 0.1, 0.9])
        out = {"time_ms_median": float(med), "time_ms_p10": float(p10),
               "time_ms_p90": float(p90), "n_reps": 1}
    else:
        out = summarize_times([_median_ms(fn, warmup, rep) for _ in range(n_repetitions)])
    torch.cuda.synchronize()
    out["bench_time_s"] = time.perf_counter() - t0
    return out


def noise_check(fn: Callable[[], Any], n: int | None = None, warmup: int | None = None,
                rep: int | None = None) -> dict[str, Any]:
    """Time one config *n* times and report the coefficient of variation.

    Differences under ``tie_factor x cv`` count as ties. One extra measurement is taken first and
    thrown away: right after idle the GPU is still raising its clocks.
    """
    b = config.BENCH
    n = b.noise_runs if n is None else n
    warmup = b.warmup_ms if warmup is None else warmup
    rep = b.rep_ms if rep is None else rep
    discarded = _median_ms(fn, max(warmup, b.post_idle_warmup_ms), rep)
    meds = [_median_ms(fn, warmup, rep) for _ in range(n)]
    cv = coefficient_of_variation(meds)
    return {"samples_ms": meds, "discarded_first_ms": discarded, "mean_ms": statistics.fmean(meds),
            "cv": cv, "tie_threshold": b.tie_factor * cv}


# =========================================================================== API check
#
# Triton's internals move between releases, so the attributes and hooks this project relies on are
# checked on the installed stack and saved to results/<gpu_tag>/api_check.json.


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


def api_report() -> dict[str, Any]:
    """Signatures and feature probes for the installed Triton / PyTorch."""
    import torch
    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction
    from triton.testing import do_bench

    out: dict[str, Any] = {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda}
    out["JITFunction.warmup"] = _sig(JITFunction.warmup) if hasattr(JITFunction, "warmup") else None
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
    """Where registers / spills / shared memory / asm live on a compiled kernel object."""
    meta = getattr(ck, "metadata", None)
    return {"type": type(ck).__name__,
            "has_n_regs": hasattr(ck, "n_regs"), "has_n_spills": hasattr(ck, "n_spills"),
            "metadata_type": type(meta).__name__,
            "metadata_fields": list(getattr(meta, "_fields", [])) or (sorted(meta) if isinstance(meta, dict) else None),
            "asm_keys": sorted((getattr(ck, "asm", None) or {}).keys())}


def maxnreg_supported() -> tuple[bool, str]:
    """Whether ``triton.Config`` accepts ``maxnreg`` on the installed Triton."""
    try:
        if "maxnreg" in inspect.signature(triton.Config.__init__).parameters:
            return True, f"triton {triton.__version__}: triton.Config(..., maxnreg=...) present"
        return False, f"triton {triton.__version__}: triton.Config has no maxnreg parameter"
    except Exception as exc:  # noqa: BLE001
        return False, f"cannot inspect triton.Config: {exc}"
