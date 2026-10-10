"""Run the notebooks headlessly on a local GPU (e.g. WSL2), replacing the Colab bootstrap.

    python scripts/run_notebooks_local.py --mini          # all notebooks, tiny workload
    python scripts/run_notebooks_local.py --mini 02_sweep 03_predictor

Executed copies go to ``notebooks/executed_local[_mini]/`` with ``run_summary.md``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archtune import runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=list(runner.ORDER))
    ap.add_argument("--mini", action="store_true", help="ARCHTUNE_MINI=1 (results_mini/, wiped first)")
    ap.add_argument("--keep-going", action="store_true", help="do not stop at the first failing notebook")
    args = ap.parse_args()
    if args.mini and args.names == list(runner.ORDER):
        shutil.rmtree(ROOT / "results_mini", ignore_errors=True)
    out = ROOT / "notebooks" / ("executed_local_mini" if args.mini else "executed_local")
    res = runner.run_notebooks(ROOT, args.names, mini=args.mini, out_dir=out, local_bootstrap=True,
                               stop_on_error=not args.keep_going)
    print((out / "run_summary.md").read_text())
    return 0 if runner.all_ok(res, args.names) else 1


if __name__ == "__main__":
    sys.exit(main())
