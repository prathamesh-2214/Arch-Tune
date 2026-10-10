"""Run every project phase on a local GPU (same functions as ``ArchTune_Colab.ipynb``).

    python scripts/run_pipeline_local.py --mini          # quick end-to-end check -> results_mini/
    python scripts/run_pipeline_local.py                 # full run -> results/
    python scripts/run_pipeline_local.py --only phase2_sweep phase4_5_analysis

Every phase is resumable; re-running skips work already saved. Each phase is wrapped so a
failure is printed with its traceback and the script stops (later phases need earlier data).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archtune import pipeline as P  # noqa: E402

PHASES = ["phase0_smoke", "phase2_sweep", "phase3_ir", "phase4_5_analysis", "phase5_spotcheck",
          "phase6_experiments", "phase8_integration", "phase9_liveness", "make_report"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {PHASES}")
    ap.add_argument("--no-inductor", action="store_true")
    ap.add_argument("--no-unpruned", action="store_true",
                    help="skip the slow unpruned triton.autotune baseline in phase 8")
    args = ap.parse_args()
    run = P.setup(ROOT, args.mini)
    t0 = time.perf_counter()
    for name in args.only or PHASES:
        fn = getattr(P, name)
        kwargs = ({"inductor": not args.no_inductor, "unpruned": not args.no_unpruned}
                  if name == "phase8_integration" else {})
        try:
            out = P.timed(name, fn, run, **kwargs)
            if name == "make_report":
                print(out[:3000])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            print(f"!!!!! {name} FAILED after {(time.perf_counter() - t0) / 60:.1f} min total")
            return 1
        sys.stdout.flush()
    print(f"ALL PHASES DONE in {(time.perf_counter() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
