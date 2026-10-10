# Setup and reproduction

Three ways to run ArchTune. **Route A** is how the reported results were produced.

| Route | Needs | Use it to |
|---|---|---|
| **A. Local GPU (Linux or WSL2)** | NVIDIA GPU (Ampere or newer for the full story), recent driver | reproduce every result and figure |
| **B. Google Colab** | a Google account (free T4, or paid L4/A100) | run without a local GPU |
| **C. CPU only** | Python 3.10+ | run the unit tests and regenerate the report from saved `results/` |

All routes use the **same pinned stack**: `torch 2.6.0` (CUDA 12.4 build) + `triton 3.2.0`
([docs/DECISIONS.md](docs/DECISIONS.md) D-001). Results from different stacks are never mixed.

---

## Route A: local GPU (Linux or Windows + WSL2)

### A1. Check the GPU is visible
On Windows, install WSL2 with Ubuntu and an up-to-date NVIDIA Windows driver; WSL2 then sees the GPU.
Inside Linux / WSL:
```bash
nvidia-smi
```
It should list your GPU. The results in this repo come from an RTX 3050 6 GB Laptop GPU (sm_86).

### A2. Python environment

Triton compiles a small CUDA driver helper at runtime, so it needs **a C compiler and Python headers**.

**Option 1: you have root (simplest)**
```bash
sudo apt update && sudo apt install -y build-essential python3-dev python3-venv
python3 -m venv ~/atenv
~/atenv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 triton==3.2.0
```

**Option 2: no root (what produced the reported results)**: a portable CPython that ships its own
headers, plus zig's bundled C compiler:
```bash
mkdir -p ~/pbs ~/zig ~/bin
curl -sSL https://github.com/astral-sh/python-build-standalone/releases/download/20241219/cpython-3.12.8+20241219-x86_64-unknown-linux-gnu-install_only.tar.gz | tar -xz -C ~/pbs
curl -sSL https://ziglang.org/download/0.13.0/zig-linux-x86_64-0.13.0.tar.xz | tar -xJ -C ~/zig --strip-components=1
printf '#!/bin/sh\nexec "$HOME/zig/zig" cc "$@"\n' > ~/bin/zcc && chmod +x ~/bin/zcc
~/pbs/python/bin/python3 -m venv ~/atenv
~/atenv/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 triton==3.2.0
export CC=$HOME/bin/zcc        # add to ~/.bashrc so every run uses it
```

> Tip: if `pip` downloads from PyPI are very slow on your network, `download.pytorch.org` (used above for
> torch and triton) is often much faster; the remaining packages are small.

### A3. Install the project
```bash
git clone <this-repo-url> archtune && cd archtune     # or cd into your copy of the folder
~/atenv/bin/pip install -e ".[dev]"
~/atenv/bin/python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
```
Expected: `2.6.0+cu124 3.2.0 <your GPU name>`.

### A4. Run, small first
Activate the environment (`source ~/atenv/bin/activate`) and keep the laptop **plugged in with the lid open**.

| Step | Command | What it does | Time (16-core laptop + RTX 3050) |
|---|---|---|---|
| 1 | `python -m archtune.preflight --out results_preflight` | 16 independent checks of every GPU code path; prints PASS / WARN / FAIL with tracebacks | a few minutes |
| 2 | `python scripts/run_pipeline_local.py --mini` | every phase end to end on 2 shapes × 12 configs, into `results_mini/` | a few minutes |
| 3 | `python scripts/run_pipeline_local.py` | the full run into `results/<gpu_tag>/` (resumable) | about an hour, mostly compilation |
| 4 | `python -m archtune.report` | regenerates `docs/RESULTS.md`, the README results block and all figures | seconds |
| 5 | `python scripts/make_project_doc.py` | regenerates the PDF write-up (needs Edge or Chrome) | seconds |

Only start step 3 when steps 1 and 2 are clean. Step 3 prints a measured time estimate after its first
10 configs, and writes live progress to `results/<gpu_tag>/progress.log`.

**Interrupted?** Re-run the same command; finished rows are skipped. To run selected phases:
```bash
python scripts/run_pipeline_local.py --only phase4_5_analysis make_report
python scripts/run_pipeline_local.py --no-unpruned     # skip the slowest step (unpruned autotune baseline)
```

### A5. What you get
```text
results/<gpu_tag>/
  sweep.parquet / sweep.csv     every (shape, config): status, compile + bench time, registers, spills,
                                shared memory, PTX/SASS instruction counts, full environment
  analysis/                     predictor accuracy, replay curves, table 6 (cost to reach 95%)
  experiments_*.parquet         maxnreg and num_stages studies
  integration.parquet           live triton.autotune: ArchTune top-k vs unpruned
  ptx_liveness.parquet          PTX liveness vs ptxas registers
  ir_exploration/               TTIR / TTGIR / LLVM IR / PTX / SASS of the fastest and slowest kernels
  artifacts/                    compressed PTX / cubin / SASS for every config
  noise.json, arch.json, api_check.json, pip_freeze.txt, progress.log, sessions.jsonl
results/figures/                every chart in the README
docs/RESULTS.md                 every table, generated
```

---

## Route B: Google Colab

Open **`notebooks/ArchTune_Colab.ipynb`** in Colab with a GPU runtime. It runs every phase in one notebook:

1. Copy the project folder to Google Drive as `My Drive/archtune`.
2. Open the notebook, choose **Runtime → Change runtime type → GPU**, run the setup cell (it installs the
   pinned torch/triton if needed and prints four numbered steps).
3. With `MINI = True`, **Run all** (preflight plus a quick end-to-end pass), then set `MINI = False` and **Run all**
   for the real run. Everything is resumable after a disconnect.

The full walkthrough, including what each check catches and troubleshooting, is in
[docs/COLAB_GUIDE.md](docs/COLAB_GUIDE.md). On a T4 (Turing), expect `num_stages` to have little effect: Triton 3.2
single-buffers shared memory there (DECISIONS D-022). Colab's 2 CPU cores make compilation, and therefore the
sweep, noticeably slower than on a multi-core laptop.

---

## Route C: CPU only

```bash
pip install -e ".[dev]"
pytest                      # unit tests for every pure function (no GPU needed)
python -m archtune.report   # regenerate docs/RESULTS.md, README numbers and figures from results/
```

---

## Checking your reproduction

- **Compare `docs/RESULTS.md`** with the committed version. Results on a different GPU will differ: that is
  expected, since the best configs, limits and compile times are architecture-specific.
- **Timing noise:** the noise check (`results/<gpu>/noise.json`) sets the tie threshold; differences smaller
  than twice its coefficient of variation are ties.
- **Same GPU, same stack:** expect the same statuses and register/shared-memory counts (compiler output is
  deterministic); timings will vary within the noise; the headline conclusions should hold.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Failed to find C compiler. Please specify via CC environment variable.` | Triton needs a C compiler for its driver helper | Install `build-essential`, or `export CC=$HOME/bin/zcc` (A2, Option 2) |
| `Python.h: No such file or directory` | missing Python headers | `python3-dev`, or use the portable CPython (A2, Option 2) |
| Preflight stage 9 reports `incorrect` rows | check `max_abs_err` in the notes; the reference is fp32 | send the report with an issue |
| Many `out_of_resources` rows | configs that need more shared memory than the GPU has | expected; that is data |
| Timings much slower right after compiles | GPU clocks drop while idle | already handled by 200 ms post-idle warm-ups (DECISIONS D-025/D-026) |
| Model config download fails | no access to huggingface.co | allow network access; it is cached in `results/` after the first run |
| Run stops when the laptop sleeps | sleep suspends WSL and the GPU job | keep it plugged in with the lid open; re-run to resume |

## Regenerating the documentation
```bash
python scripts/make_diagrams.py       # concept diagrams in docs/img/ (no measured numbers)
python -m archtune.report             # result figures + README block + RESULTS.md
python scripts/make_project_doc.py    # docs/PROJECT_DOCUMENTATION.{md,html,pdf}
```
