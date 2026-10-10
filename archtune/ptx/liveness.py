"""Backward liveness dataflow over a PTX CFG and peak register-pressure estimate.

``IN[b] = USE[b] ∪ (OUT[b] − DEF[b])``, ``OUT[b] = ∪ IN[s]`` over successors, iterated to a
fixed point. The peak number of simultaneously live virtual registers (weighted by width in
32-bit units) is a lower-level analogue of ptxas' register count. It is *not* expected to
match ptxas exactly: ptxas rematerializes values, reorders instructions, reuses registers and
packs 16-bit values, and Triton's PTX is in SSA-like form with many short-lived temporaries.

A predicated (guarded) definition does not kill the old value: on the not-taken path the
previous value survives.
"""

from __future__ import annotations

from dataclasses import dataclass

from archtune.ptx.cfg import Block, build_cfg
from archtune.ptx.parser import Function, parse_function


@dataclass
class LivenessResult:
    """Per-block IN/OUT sets and register-pressure summary."""

    live_in: list[set[str]]
    live_out: list[set[str]]
    peak_units: int  # 32-bit register units (64-bit = 2, predicates excluded)
    peak_count: int  # number of live non-predicate virtual registers at the peak
    peak_index: int  # instruction index where the peak occurs
    peak_pred: int  # peak number of live predicate registers
    n_blocks: int


def _width_units(fn: Function, reg: str) -> int:
    bits = fn.reg_bits(reg)
    if bits <= 1:
        return 0
    return max(1, (bits + 31) // 32)


def _use_def(fn: Function, b: Block) -> tuple[set[str], set[str]]:
    use: set[str] = set()
    kill: set[str] = set()
    for ins in fn.instructions[b.start:b.end]:
        srcs = list(ins.srcs) + ([ins.guard] if ins.guard else [])
        use.update(r for r in srcs if r not in kill)
        if ins.guard is None:
            kill.update(ins.dsts)
    return use, kill


def analyze(fn: Function, blocks: list[Block] | None = None) -> LivenessResult:
    """Run liveness on *fn* and compute the peak weighted live set."""
    blocks = blocks if blocks is not None else build_cfg(fn)
    nb = len(blocks)
    ud = [_use_def(fn, b) for b in blocks]
    live_in: list[set[str]] = [set() for _ in range(nb)]
    live_out: list[set[str]] = [set() for _ in range(nb)]
    changed = True
    while changed:
        changed = False
        for b in reversed(blocks):
            out = set().union(*(live_in[s] for s in b.succs)) if b.succs else set()
            use, kill = ud[b.id]
            new_in = use | (out - kill)
            if out != live_out[b.id] or new_in != live_in[b.id]:
                live_out[b.id], live_in[b.id] = out, new_in
                changed = True

    peak_units = peak_count = peak_pred = 0
    peak_index = -1
    for b in blocks:
        live = set(live_out[b.id])
        for ins in reversed(fn.instructions[b.start:b.end]):
            # Live set *after* ins is `live`; registers defined here are live at this point too.
            point = live | set(ins.dsts)
            units = sum(_width_units(fn, r) for r in point)
            preds = sum(1 for r in point if fn.reg_bits(r) <= 1)
            if units > peak_units:
                peak_units, peak_index = units, ins.index
                peak_count = sum(1 for r in point if fn.reg_bits(r) > 1)
            peak_pred = max(peak_pred, preds)
            if ins.guard is None:
                live -= set(ins.dsts)
            live |= set(ins.srcs)
            if ins.guard:
                live.add(ins.guard)
    return LivenessResult(live_in, live_out, peak_units, peak_count, peak_index, peak_pred, nb)


def peak_pressure(ptx: str, name: str | None = None) -> dict[str, int]:
    """Convenience: parse, build the CFG and return the peak-pressure summary."""
    fn = parse_function(ptx, name)
    res = analyze(fn)
    return {"peak_live_units": res.peak_units, "peak_live_count": res.peak_count,
            "peak_live_pred": res.peak_pred, "n_blocks": res.n_blocks,
            "n_instructions": len(fn.instructions)}
