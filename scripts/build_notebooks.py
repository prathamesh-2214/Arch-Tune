"""Generate notebooks/*.ipynb (thin: bootstrap, call package functions, plot, explain).

Run: ``python scripts/build_notebooks.py``. Edit this file, not the .ipynb files, so every
notebook keeps the identical bootstrap cell required by CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

BOOTSTRAP = r'''# --- ArchTune setup (same in every notebook) ---
import importlib.metadata as md, os, subprocess, sys
PROJECT = "/content/drive/MyDrive/archtune"
NEED_GPU_STACK = {need_gpu}

print("1/4 Google Drive")
if not os.path.ismount("/content/drive"):
    from google.colab import drive
    drive.mount("/content/drive")
assert os.path.isdir(PROJECT), f"project folder not found at {{PROJECT}} (see docs/COLAB_GUIDE.md step 1)"
os.chdir(PROJECT)
sys.path.insert(0, PROJECT)   # import archtune straight from Drive, no pip install needed

def _ver(pkg):
    try:
        return md.version(pkg).split("+")[0]
    except md.PackageNotFoundError:
        return None

print("2/4 torch / triton:", _ver("torch"), _ver("triton"))
if NEED_GPU_STACK and (_ver("torch"), _ver("triton")) != ("2.6.0", "3.2.0"):
    print("    installing torch 2.6.0 + triton 3.2.0 (about 1 GB, a few minutes)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch==2.6.0", "triton==3.2.0"], check=True)
    if "torch" in sys.modules:
        raise SystemExit("Installed. Now: Runtime > Restart session, then run this cell again.")

print("3/4 importing archtune")
from archtune import env
ENV = env.capture()
MINI = env.mini_mode()   # ARCHTUNE_MINI=1 -> tiny debug workload (set by run_all)
RESULTS = os.environ.get("ARCHTUNE_RESULTS") or os.path.join(PROJECT, "results_mini" if MINI else "results")

print("4/4 ready:", "MINI" if MINI else "FULL", "mode, results ->", RESULTS)
print(ENV)
if NEED_GPU_STACK and not env.pinned_ok(ENV):
    raise SystemExit("Wrong torch/triton loaded. Runtime > Restart session, then run this cell again.")'''


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook(name: str, title: str, need_gpu: bool, cells: list) -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU" if need_gpu else "None",
                   "colab": {"provenance": [], "gpuType": "T4"} if need_gpu else {"provenance": []},
                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}}
    nb.cells = [md(title), code(BOOTSTRAP.format(need_gpu=need_gpu))] + cells
    path = ROOT / "notebooks" / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    nbf.write(nb, path)
    print("wrote", path.relative_to(ROOT))


SHAPES = '''
from archtune import workloads, space
MODEL_CFG = workloads.load_model_config(cache_dir=RESULTS)
DIMS = workloads.derive_dims(MODEL_CFG)
MM_SHAPES = workloads.matmul_shapes(DIMS)
RMS_SHAPES = workloads.rmsnorm_shapes(DIMS)
SPLIT = {**workloads.train_test_split([s.shape_id for s in MM_SHAPES]),
         **workloads.train_test_split([s.shape_id for s in RMS_SHAPES])}
if MINI:
    MINI_SHAPES = workloads.mini_shapes(MM_SHAPES + RMS_SHAPES)
    MM_SHAPES = [s for s in MINI_SHAPES if s.kernel == "matmul"]
    RMS_SHAPES = [s for s in MINI_SHAPES if s.kernel == "rmsnorm"]
print(DIMS)
for s in MM_SHAPES + RMS_SHAPES:
    print(f"{s.shape_id:24s} {SPLIT[s.shape_id]:5s} {','.join(s.roles)}")
'''


def nb00() -> None:
    notebook("00_setup_smoke", """
# 00 — Setup and smoke test (Phase 0)

Goals (spec §11 Phase 0): verify the pinned Triton supports this GPU, compile and run one matmul
config, locate registers / spills / shared memory / PTX / cubin on the compiled kernel, find
`cuobjdump`/`nvdisasm`, and measure timing noise.

**Accept when:** one config runs correctly, its regs/spills/smem/PTX are captured, versions are pinned.
Copy the printed API facts into `docs/DECISIONS.md` (they are also saved to `results/<gpu>/api_check.json`).
""", True, [
        md("## 1. Device and architecture parameters (live query vs reference table)"),
        code('''
import json, os
from archtune import arch
GPU_DIR = os.path.join(RESULTS, ENV["gpu_tag"]); os.makedirs(GPU_DIR, exist_ok=True)
live = arch.live()
print(live)
ref_key = arch.reference_for_name(live.name)
diffs = arch.verify(arch.reference(ref_key), live) if ref_key else "no reference entry for this GPU"
print("differences vs reference table:", diffs)
print("provenance:", live.provenance)
env.pip_freeze(os.path.join(GPU_DIR, "pip_freeze.txt"))
json.dump(env.device_properties(), open(os.path.join(GPU_DIR, "device_properties.json"), "w"), indent=2, default=str)
'''),
        md("## 2. Verify every API the project uses on the installed versions"),
        code('''
from archtune import apicheck
API = apicheck.report()
json.dump(API, open(os.path.join(GPU_DIR, "api_check.json"), "w"), indent=2, default=str)
for k, v in API.items():
    print(f"{k:34s} {v}")
'''),
        md("## 3. Compile one config, inspect the compiled kernel"),
        code('''
from archtune import compile_info, bench, space
from archtune.kernels import matmul as mm
compile_info.fresh_cache_dir()
cfg = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
M = N = K = 1024
a, b, c = mm.make_inputs(M, N, K)
meta, launch = space.split_launch(cfg)
res = compile_info.compile_config(mm.matmul_kernel, mm.kernel_args(a, b, c), mm.grid(M, N, cfg), meta, launch)
print(res.status, res.error_msg, f"compile {res.compile_time_s:.2f}s via {res.method}, cache_miss={res.cache_miss}")
print("n_regs", res.n_regs, "n_spills", res.n_spills, "shared_bytes", res.shared_bytes)
print(apicheck.compiled_kernel_attrs(res.compiled))
for name, desc in compile_info.describe_compiled(res.compiled).items():
    print(f"  {name:22s} {desc}")
'''),
        code('''
mm.launcher(a, b, c, cfg)(); import torch; torch.cuda.synchronize()
ok, err = bench.check_correct(c, mm.reference(a, b))
print("correct:", ok, "max_abs_err:", err)
print(res.asm["ptx"][:1500])
'''),
        md("## 4. SASS tools"),
        code('''
for tool in ("cuobjdump", "nvdisasm", "ptxas"):
    print(tool, "->", compile_info.find_cuda_tool(tool))
sass = compile_info.disassemble(res.asm.get("cubin"))
print(sass[:2000] if sass else "SASS unavailable: PTX-only analysis (degrade gracefully)")
'''),
        md("## 5. Noise check (spec §8): one config timed 20× → coefficient of variation"),
        code('''
M = N = K = 4096
a, b, c = mm.make_inputs(M, N, K)
fn = mm.launcher(a, b, c, {**cfg, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128})
noise = bench.noise_check(fn, n=5 if MINI else 20)
noise.update(shape="4096x4096x4096", cfg={**cfg, "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128}, **{k: ENV[k] for k in ("gpu_tag", "stack_id")})
json.dump(noise, open(os.path.join(GPU_DIR, "noise.json"), "w"), indent=2)
print(f"CV = {noise['cv']:.4f}; differences < {noise['tie_threshold']:.2%} are ties")
'''),
        md("""
## What to record in `docs/DECISIONS.md`
- Exact attribute names for registers / spills / shared memory / asm (cell 3 output).
- Whether `JITFunction.warmup`, `maxnreg`, `prune_configs_by` and `do_bench` L2 flushing match the spec (cell 2).
- Any live-vs-reference architecture differences (cell 1).
"""),
    ])


def nb01() -> None:
    notebook("01_ir_exploration", """
# 01 — IR and PTX exploration (Phase 3)

Take two contrasting configs on one shape (fastest vs slowest valid, or spilling vs not), dump
TTIR → TTGIR → LLVM IR → PTX (→ SASS), and explain *with quoted snippets* why one is faster:
tensor-core `mma`, `ldmatrix`, `cp.async` (sm_80+ only), vector loads, unrolling, spills (`ld.local`/`st.local`).

Run after notebook 02 (uses the sweep to pick configs); falls back to a fixed pair otherwise.
""", True, [
        code(SHAPES),
        code('''
import os, pandas as pd
from archtune import compile_info, explore, space, search
from archtune.kernels import matmul as mm
GPU_DIR = os.path.join(RESULTS, ENV["gpu_tag"])
SHAPE = next(s for s in MM_SHAPES if s.shape_id == "mm_M4096_N4096_K4096")
try:
    sw = search.prepare(pd.read_parquet(os.path.join(GPU_DIR, "sweep.parquet")))
    t = sw[(sw.shape_id == SHAPE.shape_id) & sw.valid].sort_values("time_ms_median")
    knob = [c for c in t.columns if c.startswith("cfg_") and c != "cfg_hash"]
    pick = lambda r: {c[4:]: int(r[c]) for c in knob if pd.notna(r[c])}
    FAST, SLOW = pick(t.iloc[0]), pick(t.iloc[-1])
    print("fast", t.iloc[0].time_ms_median, "ms", FAST); print("slow", t.iloc[-1].time_ms_median, "ms", SLOW)
except FileNotFoundError:
    FAST = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    SLOW = {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 2, "num_stages": 2}
'''),
        code('''
compile_info.fresh_cache_dir()
a, b, c = mm.make_inputs(SHAPE.M, SHAPE.N, SHAPE.K)
LISTINGS, RES = {}, {}
for name, cfg in (("fast", FAST), ("slow", SLOW)):
    meta, launch = space.split_launch(cfg)
    r = compile_info.compile_config(mm.matmul_kernel, mm.kernel_args(a, b, c), mm.grid(SHAPE.M, SHAPE.N, cfg), meta, launch)
    sass = compile_info.disassemble(r.asm.get("cubin")) if r.asm.get("cubin") else None
    explore.dump_ir(r.asm, os.path.join(GPU_DIR, "ir_exploration"), name, sass)
    LISTINGS[name] = {"ptx": r.asm.get("ptx"), "sass": sass}; RES[name] = r
    print(name, r.status, "regs", r.n_regs, "spills", r.n_spills, "smem", r.shared_bytes)
explore.compare_features(LISTINGS)
'''),
        md("## TTGIR layouts (how Triton maps the tile onto threads and tensor cores)"),
        code('''
for name in ("fast", "slow"):
    print(f"--- {name}"); print("\\n".join(explore.ttgir_layouts(RES[name].asm.get("ttgir", ""))))
'''),
        md("## PTX snippets: tensor cores, ldmatrix, async copies, vector loads, spills"),
        code('''
for name in ("fast", "slow"):
    ptx = LISTINGS[name]["ptx"]
    for pat in (r"mma\\.sync", r"ldmatrix", r"cp\\.async\\.c[ag]", r"ld\\.global.*\\.v[248]", r"(ld|st)\\.local"):
        print(f"=== {name}: {pat}\\n{explore.snippets(ptx, pat, context=0, max_hits=2)}\\n")
'''),
        md("""
## Explanation (write after running — quote the snippets above)

*Template — replace with observations from your run; do not keep claims the data does not show.*

1. **Tensor-core instruction mix:** compare `ptx_mma` counts and the `mma.sync` shape (e.g. `m16n8k8` on sm_75 vs `m16n8k16` on sm_80+).
2. **Operand staging:** `ldmatrix` present? On sm_80+ `cp.async` copies global→shared without registers; on sm_75 there is no `cp.async`, so pipelining goes through registers.
3. **Spills:** any `ld.local`/`st.local` in the slow config? Tie to `n_spills` and the register count.
4. **Occupancy / waves:** use `predict.occupancy(n_regs, shared_bytes, num_warps, arch)` and the tile count to explain the remaining gap.
"""),
    ])


def nb02() -> None:
    notebook("02_sweep", """
# 02 — Resumable exhaustive sweep (Phase 2)

Every (shape, config) gets one row with a status (spec §7). Small-first: 10 configs × 1 shape, then
an extrapolated estimate, then the full run. Re-running any cell resumes; rows already on Drive are skipped.

**Accept when:** the parquet contains every (shape, config) row with a status, interrupted runs resume
without duplicates, and the estimate vs actual time is recorded (`sessions.jsonl`).
""", True, [
        code(SHAPES),
        code('''
import os
from archtune import sweep
RUN_ID = sweep.DEFAULT_RUN_ID
MM_CFGS = space.mini_matmul_space() if MINI else space.matmul_space()
RMS_CFGS = lambda s: space.rmsnorm_space(s.N)[:6] if MINI else space.rmsnorm_space(s.N)
TASKS = sweep.make_tasks(MM_SHAPES, lambda s: MM_CFGS) + sweep.make_tasks(RMS_SHAPES, RMS_CFGS)
print(len(TASKS), "tasks:", len(MM_SHAPES), "matmul shapes x", len(MM_CFGS), "configs +", len(RMS_SHAPES), "rmsnorm shapes")
'''),
        md("## Small first: 10 configs × 1 shape"),
        code('''
small = [t for t in TASKS if t.shape.shape_id == MM_SHAPES[-1].shape_id][:10]
s1 = sweep.run_sweep(small, RESULTS, ENV, run_id=RUN_ID)
done = sweep.load_parts(os.path.join(RESULTS, ENV["gpu_tag"], "sweep"))
sample = done[done.cfg_hash.isin([t.cfg_hash for t in small]) & (done.shape_id == small[0].shape.shape_id)]
remaining = sweep.pending(TASKS, sweep.keys_of(done), ENV["gpu_tag"], RUN_ID)
n_cold = len({(t.shape.kernel, t.cfg_hash) for t in remaining})
EST = sweep.estimate_seconds(sample, n_tasks=len(remaining), n_cold_compiles=n_cold)
print(sample[["cfg_hash", "status", "compile_time_s", "n_regs", "n_spills", "shared_bytes", "time_ms_median", "task_wall_s"]])
print(f"Estimated full sweep: {EST['estimate_s']/3600:.2f} h for {len(remaining)} tasks ({n_cold} cold compiles)")
# HEALTH CHECK: these 10 configs all fit on every GPU, so any failure here is a bug, not data.
bad = sample[sample.status.isin(["compile_error", "launch_error", "incorrect"])]
assert len(sample) == len(small) and bad.empty, (
    f"small-first: {len(bad)} failing configs -> fix before the full sweep: {bad.error_msg.head(3).tolist()}")
assert sample.n_regs.notna().all() and sample.time_ms_median.notna().all(), "small-first: registers/timings not captured"
'''),
        md("## Full sweep (resumable — just re-run this cell after a disconnect)"),
        code('''
summary = sweep.run_sweep(TASKS, RESULTS, ENV, run_id=RUN_ID, estimate_s=EST["estimate_s"])
print(summary)
df = sweep.consolidate(os.path.join(RESULTS, ENV["gpu_tag"]))
print(df.groupby(["kernel", "status"]).size())
assert not df.duplicated(list(sweep.KEY_COLUMNS)).any()
missing = sweep.pending(TASKS, sweep.keys_of(df), ENV["gpu_tag"], RUN_ID)
print("missing rows:", len(missing))
# HEALTH CHECKS: fail the notebook (and stop run_all) if the data is not usable.
assert not missing, f"{len(missing)} tasks have no row"
broken = df.status.isin(["compile_error", "launch_error"]).mean()
assert broken < 0.02, f"{broken:.1%} of rows are compile/launch errors: {df.loc[df.status.isin(['compile_error','launch_error']), 'error_msg'].head(3).tolist()}"
'''),
    ])


def nb03() -> None:
    notebook("03_predictor", """
# 03 — Predictor (Phase 4) — CPU runtime OK

Fits the register and shared-memory models on **training-split shapes only**, then reports
median absolute register error, shared-memory exact-match rate and spill precision/recall for both the
analytic and the fitted variant (spec §9), and lists the top-10 register mispredictions.

Note: for one Triton specialization, registers/spills/shared memory depend on the config, not on the
shape, so the register / shared-memory metrics are in-sample over configs (DECISIONS D-007). The
held-out shapes matter for the search evaluation (notebook 04).

**Accept when:** figure 2 exists, the metrics are reported, and the top-10 mispredictions are explained or marked unexplained.
""", False, [
        code('''
import json, os, pandas as pd
from archtune import analysis
GPU = "t4_sm75"   # the GPU whose sweep to analyze
summary = analysis.run(RESULTS, GPU, refit=True)
print(json.dumps(summary["models"], indent=1))
print(json.dumps(summary["predictor"], indent=1))
'''),
        code('''
from archtune import plots
pm = pd.read_parquet(os.path.join(RESULTS, GPU, "analysis", "predictor.parquet"))
plots.fig2_predictor(pm, f"Predicted vs measured — {GPU} (fitted on train split)",
                     os.path.join(RESULTS, "figures", f"fig2_predictor_{GPU}.png"))
'''),
        md("## Top-10 register mispredictions + PTX evidence"),
        code('''
from archtune import compile_info, ptx_features
top = pd.read_csv(os.path.join(RESULTS, GPU, "analysis", "top_mispredictions.csv"))
sweep_df = analysis.load_sweep(RESULTS, GPU)
art = sweep_df.drop_duplicates("cfg_hash").set_index("cfg_hash")["artifact_path"]
feat_cols = [c for c in sweep_df.columns if c.startswith("ptx_")]
top = top.join(sweep_df.drop_duplicates("cfg_hash").set_index("cfg_hash")[feat_cols], on="cfg_hash")
top[[c for c in top.columns if c.startswith("cfg_")] + ["n_regs", "regs_pred", "n_spills", "regs_abs_err"] + feat_cols]
'''),
        md("""
### Explanations (fill in from the PTX diffs; leave "unexplained" where the evidence is not conclusive)

Edit `results/<gpu>/analysis/top_mispredictions.csv` column `explanation`. Typical causes to check:
operand-fragment registers that grow with `BLOCK_SIZE_K` (more `ldmatrix` destinations), extra
address registers for `num_stages` buffers, ptxas reusing accumulator registers, 2-warp configs
hitting the 255 cap.
"""),
    ])


def nb04() -> None:
    notebook("04_search_eval", """
# 04 — Search evaluation (Phase 5) — CPU runtime OK (live spot-check needs a GPU)

Replay exhaustive / random-k (20 seeds) / smart-k over the sweep data. Cost = cold compile +
benchmark seconds of every tried config (+ predictor time for smart-k). Held-out shapes are reported separately.

**Accept when:** figure 1 and table 6 exist for T4 with held-out results separated.
""", False, [
        code('''
import os, pandas as pd
from archtune import analysis, plots, search
GPU = "t4_sm75"
if not os.path.exists(os.path.join(RESULTS, GPU, "analysis", "curves.parquet")):
    analysis.run(RESULTS, GPU)
curves = pd.read_parquet(os.path.join(RESULTS, GPU, "analysis", "curves.parquet"))
plots.fig1_quality_vs_cost(curves, GPU, os.path.join(RESULTS, "figures", f"fig1_quality_vs_cost_{GPU}.png"))
'''),
        code('''
table6 = pd.read_csv(os.path.join(RESULTS, GPU, "analysis", "table6.csv"))
display(table6)
display(search.cost_ratio(table6))
'''),
        md("""
## Interpreting (spec §1.4) — write from the table, never overstate
- Smart-k reaches 95% at clearly lower cost than random-k on **test** shapes → main claim holds.
- Smart ≈ random (ratio within noise) → valid result; explain with the performance landscape (how many configs are within 5%?).
- Smart misses the target → error analysis of predictor vs ptxas (notebook 03) is the deliverable.
"""),
        md("## Live spot-check (GPU runtime): smart-k with `triton.autotune` on 3 held-out shapes"),
        code('''
import torch
PINNED = torch.__version__.split("+")[0] == "2.6.0"
if not (torch.cuda.is_available() and PINNED):
    print("Skipping live spot-check: needs a GPU runtime with the pinned stack "
          "(run notebook 00 or 02 first on this runtime so torch 2.6.0 is installed).")
else:
    from archtune import arch, triton_hook, space
    from archtune.analysis import load_models
    reg, smem = load_models(RESULTS, ENV["gpu_tag"])
    A = arch.live()
    sw = search.prepare(analysis.load_sweep(RESULTS, ENV["gpu_tag"]))
    summ = __import__("json").load(open(os.path.join(RESULTS, ENV["gpu_tag"], "analysis", "summary.json")))
    test_ids = [s for s, v in summ["split"].items() if v == "test" and s.startswith("mm_")][:3]
    rows = []
    for sid in test_ids:
        r0 = sw[sw.shape_id == sid].iloc[0]
        cfgs = space.mini_matmul_space() if MINI else space.matmul_space()
        live = triton_hook.tune_live(cfgs, int(r0.M), int(r0.N), int(r0.K),
                                     prune_by=triton_hook.archtune_prune_configs_by(A, 8, reg, smem))
        best = search.best_time(sw[sw.shape_id == sid])
        rows.append({**live, "shape_id": sid, "replay_best_ms": best, "quality_live": best / live["time_ms_median"]})
    spot = pd.DataFrame(rows); spot.to_parquet(os.path.join(RESULTS, ENV["gpu_tag"], "spotcheck.parquet")); display(spot)
'''),
    ])


def nb05() -> None:
    notebook("05_compiler_experiments", """
# 05 — Compiler experiments (Phase 6)

1. **`maxnreg` sweep** on 3 representative configs (fastest; most registers without spilling; fastest spilling).
2. **`num_stages` sweep** (1–6) on the 5 fastest configs of two shapes, with PTX evidence (`cp.async` count, shared memory).

**Accept when:** figures 3–4 exist with a written explanation tying spills / occupancy / pipelining to runtime.
""", True, [
        code(SHAPES),
        code('''
import os, pandas as pd
from archtune import experiments, sweep, search, analysis, plots, arch
GPU_DIR = os.path.join(RESULTS, ENV["gpu_tag"])
SW = analysis.load_sweep(RESULTS, ENV["gpu_tag"])
supported, why = experiments.maxnreg_supported(); print(why)
SHAPE = next(s for s in MM_SHAPES if s.shape_id == "mm_M4096_N4096_K4096")
'''),
        code('''
if supported:
    reps = experiments.representative_configs(SW, SHAPE.shape_id)
    tasks = sweep.make_tasks([SHAPE], lambda s: experiments.with_maxnreg(reps))
    print(len(tasks), "maxnreg tasks")
    sweep.run_sweep(tasks, RESULTS, ENV, run_id="maxnreg_v1", table="experiments_maxnreg")
    mx = sweep.consolidate(GPU_DIR, "experiments_maxnreg")
    frame = experiments.maxnreg_frame(mx, arch.live()); display(frame)
    plots.fig3_maxnreg(frame, f"Register cap sweep — {ENV['gpu_tag']}, {SHAPE.shape_id}",
                       os.path.join(RESULTS, "figures", f"fig3_maxnreg_{ENV['gpu_tag']}.png"))
else:
    print("maxnreg not supported on this Triton: documented in DECISIONS.md, skipped")
'''),
        code('''
v = search.prepare(SW); v = v[v.valid & (v.kernel == "matmul")]
base = []
for sid in ("mm_M4096_N4096_K4096", sorted(v.shape_id.unique())[0]):
    top = v[v.shape_id == sid].nsmallest(5, "time_ms_median")
    base += [(sid, experiments.cfg_of(r)) for _, r in top.iterrows()]
shapes = {s.shape_id: s for s in MM_SHAPES}
tasks = []
for sid, cfg in base:
    tasks += sweep.make_tasks([shapes[sid]], lambda s, c=cfg: experiments.with_num_stages([c]))
print(len(tasks), "num_stages tasks")
sweep.run_sweep(tasks, RESULTS, ENV, run_id="stages_v1", table="experiments_stages")
st = sweep.consolidate(GPU_DIR, "experiments_stages")
frame4 = experiments.num_stages_frame(st)
display(frame4.groupby("num_stages")[["shared_bytes", "ptx_cp_async", "n_regs", "time_rel"]].median())
plots.fig4_num_stages(frame4, f"Pipelining depth — {ENV['gpu_tag']}", os.path.join(RESULTS, "figures", f"fig4_num_stages_{ENV['gpu_tag']}.png"))
'''),
        md("""
## Explanation (write from the figures; quote PTX where possible)
- **Register cap:** as `maxnreg` drops, where do spills start (`n_spills`, `ld.local` count), how much does occupancy (blocks/SM) rise, and which wins at runtime?
- **Pipelining:** shared memory should grow ~linearly with `num_stages`; on sm_75 check whether `cp.async` is absent and whether extra stages still help; on sm_80+ relate the `cp.async` count to the speedup and find where shared memory starts limiting blocks/SM.
"""),
    ])


def nb07() -> None:
    notebook("07_integration", """
# 07 — Integration (Phase 8): ArchTune inside `triton.autotune`

Live tuning time and chosen-kernel performance, unpruned (all 384 configs) vs ArchTune pruning
(`prune_configs_by` with `early_config_prune` + `perf_model` + `top_k`). Optional, feature-flagged
Inductor `max-autotune` experiment at the end.

**Accept when:** live tuning time and chosen-kernel performance are reported for the Triton hook.
""", True, [
        code(SHAPES),
        code('''
import os, pandas as pd
from archtune import arch, triton_hook, space
from archtune.analysis import load_models
A = arch.live()
models = load_models(RESULTS, ENV["gpu_tag"])
reg, smem = models if models else (None, None)
kw = {"reg_model": reg, "smem_model": smem} if models else {}
SHAPES = [s for s in MM_SHAPES if s.shape_id in ("mm_M4096_N4096_K4096", "mm_M2048_N2560_K960", "mm_M128_N1600_K960")]
CFGS = space.mini_matmul_space() if MINI else space.matmul_space()
rows = []
for s in SHAPES:
    for k in (8, 16):
        r = triton_hook.tune_live(CFGS, s.M, s.N, s.K, prune_by=triton_hook.archtune_prune_configs_by(A, k, **kw))
        rows.append({"mode": f"archtune_top{k}", "shape_id": s.shape_id, **r})
        print(rows[-1]["mode"], s.shape_id, f"{r['tuning_s']:.1f}s", f"{r['time_ms_median']:.4f} ms")
'''),
        md("Unpruned baseline: compiles and benchmarks all 384 configs per shape (slow on T4 — tens of minutes)."),
        code('''
for s in SHAPES:
    r = triton_hook.tune_live(CFGS, s.M, s.N, s.K, prune_by=None)
    rows.append({"mode": "unpruned", "shape_id": s.shape_id, **r})
    print("unpruned", s.shape_id, f"{r['tuning_s']:.1f}s", f"{r['time_ms_median']:.4f} ms")
out = pd.DataFrame(rows); out["best_cfg"] = out["best_cfg"].astype(str)
out.to_parquet(os.path.join(RESULTS, ENV["gpu_tag"], "integration.parquet")); display(out)
'''),
        md("## Optional: Inductor `max-autotune` baseline (+ feature-flagged pruning patch)"),
        code('''
from archtune import inductor_patch, predict
base = [inductor_patch.compile_and_time(s.M, s.N, s.K) for s in SHAPES]
os.environ[inductor_patch.FLAG] = "1"
keep = lambda cfg, m, n, k: not predict.predict_matmul(cfg, m, n, k, A)["spill_pred"] and predict.predict_matmul(cfg, m, n, k, A)["feasible"]
patched, msg = inductor_patch.patch_mm_configs(keep); print(patched, msg)
pat = [inductor_patch.compile_and_time(s.M, s.N, s.K) for s in SHAPES] if patched else []
inductor_patch.unpatch()
ind = pd.DataFrame([{**r, "mode": "inductor"} for r in base] + [{**r, "mode": "inductor_patched"} for r in pat])
ind.to_parquet(os.path.join(RESULTS, ENV["gpu_tag"], "inductor.parquet")); display(ind)
'''),
    ])


def nb08() -> None:
    notebook("08_ptx_liveness", """
# 08 — STRETCH: PTX liveness analyzer vs ptxas (Phase 9) — CPU runtime OK

For every swept config: parse its saved PTX, build the CFG, run backward liveness
(IN = USE ∪ (OUT − DEF)), and compare the peak simultaneously-live 32-bit registers with ptxas `n_regs`.
Expect gaps: ptxas rematerializes, reschedules, reuses registers and packs 16-bit values.

**Accept when:** unit tests on hand-written PTX pass (`pytest tests/test_ptx.py`) and the comparison plot exists.
""", False, [
        code('''
import os, pandas as pd
from archtune import analysis, compile_info, plots
from archtune.ptx.liveness import peak_pressure
GPU = "t4_sm75"
SW = analysis.load_sweep(RESULTS, GPU)
per_cfg = SW[(SW.kernel == "matmul") & SW.artifact_path.notna()].drop_duplicates("cfg_hash")
rows = []
for r in per_cfg.itertuples():
    ptx = compile_info.load_artifact(r.artifact_path, "ptx")
    if ptx:
        rows.append({"cfg_hash": r.cfg_hash, "n_regs": r.n_regs, "n_spills": r.n_spills, **peak_pressure(ptx)})
lv = pd.DataFrame(rows); lv.to_parquet(os.path.join(RESULTS, GPU, "ptx_liveness.parquet"))
plots.fig_liveness(lv, f"PTX liveness vs ptxas — {GPU}", os.path.join(RESULTS, "figures", f"fig_liveness_{GPU}.png"))
lv.describe()
'''),
    ])


def run_all() -> None:
    notebook("run_all", """
# run_all — every phase in ONE Colab runtime, debug-first

Open this notebook on a **T4 GPU** runtime and use **Runtime → Run all**. It stops at the first
problem, so a bug shows up within minutes instead of hours into the sweep.

| Stage | What | Time | Writes to |
|---|---|---|---|
| A. Preflight | 16 targeted checks of every GPU code path (compile, inspect, out-of-resources, spills, resume, hook, Inductor, liveness, report) | ~5–10 min | `results_preflight/` |
| B. Mini notebooks | every notebook end to end on 2 shapes × 12 configs (`ARCHTUNE_MINI=1`) | ~10–20 min | `results_mini/`, `notebooks/executed_mini/` |
| C. Full notebooks | the real runs | hours (resumable) | `results/`, `notebooks/executed/` |
| D. Report | README / RESULTS numbers and figures | ~1 min | `docs/`, `results/figures/` |

Stages A and B never touch `results/`. See `docs/COLAB_GUIDE.md` for how to read failures.
""", True, [
        md("## Settings"),
        code('''
RUN_STAGE_A = True     # preflight checks
RUN_STAGE_B = True     # all notebooks in mini mode
RUN_STAGE_C = True     # full runs (set False to stop after debugging)
FULL_NOTEBOOKS = ["00_setup_smoke", "02_sweep", "01_ir_exploration", "03_predictor",
                  "04_search_eval", "05_compiler_experiments", "07_integration", "08_ptx_liveness"]
SKIP_INDUCTOR_PREFLIGHT = False
'''),
        md("## Stage A — preflight (fails fast; read the table it prints)"),
        code('''
import subprocess, sys
if RUN_STAGE_A:
    cmd = [sys.executable, "-m", "archtune.preflight", "--out", f"{PROJECT}/results_preflight"]
    if SKIP_INDUCTOR_PREFLIGHT:
        cmd.append("--skip-inductor")
    proc = subprocess.Popen(cmd, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:          # stream live
        print(line, end="")
    rc = proc.wait()
    assert rc == 0, (f"PREFLIGHT FAILED - fix before the long runs. Report: "
                     f"{PROJECT}/results_preflight/{ENV['gpu_tag']}/preflight.md")
'''),
        md("## Stage B — every notebook in mini mode (notebook glue, plots, file paths)"),
        code('''
import shutil
from archtune import runner
if RUN_STAGE_B:
    shutil.rmtree(f"{PROJECT}/results_mini", ignore_errors=True)
    mini = runner.run_notebooks(PROJECT, mini=True, out_dir=f"{PROJECT}/notebooks/executed_mini")
    assert runner.all_ok(mini), ("MINI RUN FAILED - open notebooks/executed_mini/run_summary.md "
                                 "and the failing notebook in notebooks/executed_mini/")
    print("Stage B OK")
'''),
        md("""
## Stage C — full runs

Long. If Colab disconnects: reconnect, run the first cell, set `RUN_STAGE_A = RUN_STAGE_B = False`,
and run all again: the sweep resumes from Drive. While a cell runs here, follow progress in
`results/<gpu>/progress.log` (open it in Drive).
"""),
        code('''
if RUN_STAGE_C:
    full = runner.run_notebooks(PROJECT, FULL_NOTEBOOKS, mini=False, out_dir=f"{PROJECT}/notebooks/executed")
    assert runner.all_ok(full, FULL_NOTEBOOKS), "FULL RUN FAILED - see notebooks/executed/run_summary.md"
'''),
        md("## Stage D — generate README / RESULTS numbers and figures from results/"),
        code('''
if RUN_STAGE_C:
    subprocess.run([sys.executable, "-m", "archtune.report"], cwd=PROJECT, check=True)
    print(open(f"{PROJECT}/docs/RESULTS.md").read()[:3000])
'''),
    ])


def all_in_one() -> None:
    notebook("ArchTune_Colab", """
# ArchTune — the whole project in one Colab notebook

**How to use (T4 GPU runtime):**
1. Run the setup cell below (it installs torch 2.6.0 / Triton 3.2.0 if needed).
2. Leave `MINI = True` and **Runtime → Run all**: every phase runs on a tiny workload in ~10–15 min
   and writes to `results_mini/`. If anything is broken, it fails here, not hours into the real run.
3. Set `MINI = False` and **Run all** again for the real results (`results/`). Expected: ~1–2 h on a T4
   (the sweep prints a measured estimate after its first 10 configs).

Everything is **resumable**: after a disconnect, run the setup cell, then Run all — finished work is
read back from Drive and the sweep continues where it stopped. Turn phases off in `DO` to skip them.
Live sweep progress is also in `results/<gpu>/progress.log`.
""", True, [
        md("## Settings"),
        code('''
MINI = True            # True: quick end-to-end check (results_mini/). False: real run (results/).
RUN_PREFLIGHT = MINI   # 16 targeted checks of the GPU code paths (~5-10 min), useful on the first run
DO = {
    "phase0_smoke": True,        # API check, smoke compile, noise          (~2 min)
    "phase2_sweep": True,        # exhaustive sweep, resumable              (longest)
    "phase3_ir": True,           # IR / PTX of fastest vs slowest config    (~1 min)
    "phase4_5_analysis": True,   # predictor fit, replay, figures 1-2, table 6 (CPU, ~2 min)
    "phase5_spotcheck": True,    # live smart-k on 3 held-out shapes        (~5 min)
    "phase6_experiments": True,  # maxnreg + num_stages, figures 3-4        (~10 min)
    "phase8_integration": True,  # triton.autotune hook vs unpruned         (~15-30 min)
    "phase9_liveness": True,     # PTX liveness (stretch)                   (~2 min)
    "report": True,              # README / RESULTS numbers + figures
}
INDUCTOR_BASELINE = True       # optional torch.compile max-autotune baseline in phase 8
from archtune import pipeline as P
run = P.setup(PROJECT, MINI)
'''),
        md("## Preflight (first run only)"),
        code('''
if RUN_PREFLIGHT:
    import subprocess, sys
    proc = subprocess.Popen([sys.executable, "-m", "archtune.preflight", "--out", f"{PROJECT}/results_preflight"],
                            cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
    assert proc.wait() == 0, f"PREFLIGHT FAILED: see {PROJECT}/results_preflight/{ENV['gpu_tag']}/preflight.md"
'''),
        md("## Phase 0 — setup checks and noise"),
        code('if DO["phase0_smoke"]:\n    P.timed("Phase 0", P.phase0_smoke, run)'),
        md("## Phase 2 — exhaustive sweep (resumable)"),
        code('if DO["phase2_sweep"]:\n    P.timed("Phase 2 sweep", P.phase2_sweep, run)'),
        md("## Phase 3 — IR / PTX exploration\nWrite the explanation in `notebooks/01_ir_exploration.ipynb` using `results/<gpu>/ir_exploration/`."),
        code('if DO["phase3_ir"]:\n    display(P.timed("Phase 3", P.phase3_ir, run))'),
        md("## Phases 4–5 — predictor and search evaluation"),
        code('''
if DO["phase4_5_analysis"]:
    P.timed("Phases 4-5", P.phase4_5_analysis, run)
    from IPython.display import Image
    tag = run.env["gpu_tag"]
    display(Image(str(run.figures / f"fig1_quality_vs_cost_{tag}.png")))
    display(Image(str(run.figures / f"fig2_predictor_{tag}.png")))
'''),
        code('if DO["phase5_spotcheck"]:\n    display(P.timed("Phase 5 live spot-check", P.phase5_spotcheck, run))'),
        md("## Phase 6 — compiler experiments"),
        code('if DO["phase6_experiments"]:\n    P.timed("Phase 6", P.phase6_experiments, run)'),
        md("## Phase 8 — integration"),
        code('if DO["phase8_integration"]:\n    display(P.timed("Phase 8", P.phase8_integration, run, inductor=INDUCTOR_BASELINE))'),
        md("## Phase 9 — PTX liveness (stretch)"),
        code('if DO["phase9_liveness"]:\n    P.timed("Phase 9", P.phase9_liveness, run)'),
        md("## Report"),
        code('if DO["report"]:\n    print(P.timed("Report", P.make_report, run)[:4000])'),
        md("""
**Done.** If this was the `MINI = True` pass and everything succeeded, set `MINI = False` in Settings
and Run all again. Full results: `results/`, `docs/RESULTS.md`, `README.md`.
"""),
    ])


if __name__ == "__main__":
    for f in (nb00, nb01, nb02, nb03, nb04, nb05, nb07, nb08, run_all, all_in_one):
        f()
