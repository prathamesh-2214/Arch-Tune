"""ArchTune: predict how Triton matmul configs will compile, and skip the bad ones before autotuning.

The package is five modules:

* ``config``   every tunable parameter and hardware table;
* ``gpu``      environment, architecture, the Triton kernels, compiling, timing, API checks;
* ``model``    config space, workloads, the predictor, search replay, autotuner hooks;
* ``compiler`` PTX parser, CFG, liveness, PTX/SASS instruction counts, IR comparison;
* ``run``      sweep, analysis, experiments, pipeline steps, preflight, figures, reports, CLI.

Only ``gpu`` and the GPU steps in ``run`` need a GPU; Triton is imported guarded, so the predictor,
the search replay and the PTX analysis run on any machine.
"""

__version__ = "0.2.0"
