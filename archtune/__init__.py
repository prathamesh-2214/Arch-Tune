"""ArchTune: architecture-aware pruning for Triton / torch.compile autotuning.

Modules that need a GPU (``kernels``, ``compile_info``, ``bench``, ``sweep`` execution,
``triton_hook``, ``inductor_patch``) import torch/triton lazily, so the pure modules
(``space``, ``workloads``, ``arch``, ``predict``, ``search``, ``ptx_features``, ``ptx``)
are usable and testable on a CPU-only machine.
"""

__version__ = "0.1.0"
