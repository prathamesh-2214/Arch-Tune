"""Looking inside what Triton and ptxas produced.

* a small PTX front end: parser, basic blocks and control-flow graph, backward liveness and the
  peak register pressure it implies;
* static instruction counts from PTX and SASS listings (tensor cores, ``ldmatrix``, ``cp.async``,
  spills, barriers), stored in every sweep row;
* helpers for dumping and comparing the IR stages of individual configs.

All of it is text processing, so it runs without a GPU.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

# =========================================================================== PTX parser
#
# Covers what Triton emits: .reg declarations, labels, predicated instructions, vector operands
# {%r1, %r2}, address operands [%rd1+16] and branches.

#: Opcodes (first dotted component) whose operands are all sources.
NO_DST_OPCODES = {"st", "bar", "barrier", "bra", "ret", "exit", "membar", "fence", "red",
                  "prefetch", "trap", "cp", "stmatrix", "wgmma", "griddepcontrol", "mbarrier"}

#: Register width in bits by the type of its ``.reg`` declaration.
TYPE_BITS = {"pred": 1, "b8": 16, "u8": 16, "s8": 16, "b16": 16, "u16": 16, "s16": 16, "f16": 16,
             "bf16": 16, "f16x2": 32, "bf16x2": 32, "b32": 32, "u32": 32, "s32": 32, "f32": 32,
             "b64": 64, "u64": 64, "s64": 64, "f64": 64, "b128": 128}

_REG_DECL = re.compile(r"\.reg\s+\.(\w+)\s+(%[A-Za-z_][\w$]*?)(?:<(\d+)>)?\s*;")
_LABEL = re.compile(r"^\s*([$\w][\w$]*)\s*:\s*$")
_REG_TOKEN = re.compile(r"%[A-Za-z_][\w$]*")


@dataclass
class Instruction:
    """One PTX instruction."""

    index: int
    opcode: str
    guard: str | None = None
    guard_negated: bool = False
    dsts: list[str] = field(default_factory=list)
    srcs: list[str] = field(default_factory=list)
    target: str | None = None  # branch target label
    text: str = ""

    @property
    def base(self) -> str:
        """First component of the opcode, e.g. ``ld`` for ``ld.global.v4.b32``."""
        return self.opcode.split(".")[0]

    @property
    def is_branch(self) -> bool:
        return self.base == "bra"

    @property
    def is_return(self) -> bool:
        return self.base in ("ret", "exit")


@dataclass
class Function:
    """A parsed PTX function body."""

    name: str
    reg_decls: dict[str, tuple[str, int]]  # prefix -> (type, count)
    instructions: list[Instruction]
    labels: dict[str, int]  # label -> index of the next instruction

    def reg_bits(self, reg: str) -> int:
        """Width in bits of a virtual register (32 if undeclared)."""
        prefix = re.match(r"%[A-Za-z_$]+", reg)
        decl = self.reg_decls.get(prefix.group(0) if prefix else reg) or self.reg_decls.get(reg)
        return TYPE_BITS.get(decl[0], 32) if decl else 32


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def split_operands(s: str) -> list[str]:
    """Split an operand list on commas that are not inside ``{}`` or ``[]``."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


def _regs_in(operand: str, declared: set[str]) -> list[str]:
    """Virtual registers in an operand (special registers such as %tid.x left out)."""
    regs = []
    for tok in _REG_TOKEN.findall(operand):
        prefix = re.match(r"%[A-Za-z_$]+", tok).group(0)
        if tok in declared or prefix in declared:
            regs.append(tok)
    return regs


def parse_instruction(stmt: str, index: int, declared: set[str]) -> Instruction:
    """Parse one statement (without its trailing ``;``)."""
    stmt = stmt.strip()
    guard, neg = None, False
    m = re.match(r"@(!?)(%\w+)\s+", stmt)
    if m:
        neg, guard = m.group(1) == "!", m.group(2)
        stmt = stmt[m.end():]
    parts = stmt.split(None, 1)
    opcode = parts[0]
    operands = split_operands(parts[1]) if len(parts) > 1 else []
    ins = Instruction(index=index, opcode=opcode, guard=guard, guard_negated=neg, text=stmt)
    base = opcode.split(".")[0]
    if base == "bra":
        ins.target = operands[0] if operands else None
        return ins
    if not operands:
        return ins
    if base in NO_DST_OPCODES or (base == "atom" and len(operands) < 3):
        for op in operands:
            ins.srcs += _regs_in(op, declared)
        return ins
    first, rest = operands[0], operands[1:]
    if first.startswith("["):  # an address as the first operand is read, not written
        ins.srcs += _regs_in(first, declared)
    else:
        ins.dsts += _regs_in(first, declared)  # "setp ... %p1|%p2" writes two predicates
    for op in rest:
        ins.srcs += _regs_in(op, declared)
    return ins


def parse_function(ptx: str, name: str | None = None) -> Function:
    """Parse the first (or the named) ``.entry`` / ``.func`` of a PTX module."""
    text = _strip_comments(ptx)
    pat = re.compile(r"\.(?:visible\s+)?(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)")
    matches = list(pat.finditer(text))
    if not matches:
        raise ValueError("no .entry/.func found")
    m = next((mm for mm in matches if name is None or mm.group(1) == name), None)
    if m is None:
        raise ValueError(f"function {name!r} not found")
    start = text.index("{", m.end())
    depth, i = 0, start
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    body = text[start + 1:i]

    decls: dict[str, tuple[str, int]] = {}
    for d in _REG_DECL.finditer(body):
        typ, reg, count = d.group(1), d.group(2), d.group(3)
        decls[reg] = (typ, int(count) if count else 1)
    body = _REG_DECL.sub("", body)
    declared = set(decls)

    instrs: list[Instruction] = []
    labels: dict[str, int] = {}
    # Labels end with ':', statements with ';', directives start with '.'.
    for chunk in re.split(r";", body):
        stmt_lines = []
        for line in chunk.strip().splitlines():
            if line.strip() in ("{", "}"):  # scope braces around inline asm blocks
                continue
            lm = _LABEL.match(line)
            if lm:
                labels[lm.group(1)] = len(instrs)
            elif line.strip():
                stmt_lines.append(line.strip())
        stmt = " ".join(stmt_lines).strip()
        while True:  # a label can share its line with a statement: "$L__BB0_2: add.s32 ..."
            lm2 = re.match(r"^([$\w][\w$]*)\s*:\s*(.*)$", stmt)
            if lm2 and not stmt.startswith("@"):
                labels[lm2.group(1)] = len(instrs)
                stmt = lm2.group(2).strip()
            else:
                break
        if not stmt or stmt.startswith("."):
            continue
        instrs.append(parse_instruction(stmt, len(instrs), declared))
    return Function(name=m.group(1), reg_decls=decls, instructions=instrs, labels=labels)


# =========================================================================== control-flow graph


@dataclass
class Block:
    """A basic block: instructions ``[start, end)`` and its neighbours."""

    id: int
    start: int
    end: int
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)


def build_cfg(fn: Function) -> list[Block]:
    """Split *fn* into basic blocks and connect them.

    Leaders are the first instruction, every label target and every instruction after a branch or
    return. A guarded branch (``@%p bra``) falls through as well as jumping.
    """
    n = len(fn.instructions)
    if n == 0:
        return []
    leaders = {0}
    leaders.update(i for i in fn.labels.values() if i < n)
    for ins in fn.instructions:
        if (ins.is_branch or ins.is_return) and ins.index + 1 < n:
            leaders.add(ins.index + 1)
    starts = sorted(leaders)
    blocks = [Block(i, s, (starts[i + 1] if i + 1 < len(starts) else n)) for i, s in enumerate(starts)]
    block_at = {b.start: b.id for b in blocks}
    for b in blocks:
        last = fn.instructions[b.end - 1]
        succs: list[int] = []
        if last.is_branch and last.target in fn.labels and fn.labels[last.target] in block_at:
            succs.append(block_at[fn.labels[last.target]])
        falls_through = not last.is_return and (not last.is_branch or last.guard is not None)
        if falls_through and b.id + 1 < len(blocks):
            succs.append(b.id + 1)
        b.succs = list(dict.fromkeys(succs))
    for b in blocks:
        for s in b.succs:
            blocks[s].preds.append(b.id)
    return blocks


# =========================================================================== liveness
#
# IN[b] = USE[b] | (OUT[b] - DEF[b]) and OUT[b] = union of IN[s] over successors, iterated until
# nothing changes. The largest number of values alive at once (64-bit values count twice,
# predicates separately) estimates how many registers the kernel needs before ptxas runs.
#
# It won't match ptxas exactly: ptxas reorders, recomputes cheap values and reuses registers,
# while this takes the PTX order as given. A predicated definition does not end the old value's
# life, because on the not-taken path the old value is still the live one.


@dataclass
class LivenessResult:
    """Per-block IN/OUT sets and the register-pressure summary."""

    live_in: list[set[str]]
    live_out: list[set[str]]
    peak_units: int  # 32-bit register units (64-bit = 2, predicates excluded)
    peak_count: int  # live non-predicate virtual registers at the peak
    peak_index: int  # instruction index of the peak
    peak_pred: int  # peak number of live predicates
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


def analyze_liveness(fn: Function, blocks: list[Block] | None = None) -> LivenessResult:
    """Run liveness on *fn* and find the peak weighted live set."""
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
            point = live | set(ins.dsts)  # values defined here are live at this point too
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
    """Parse, build the CFG and return the peak-pressure summary."""
    fn = parse_function(ptx, name)
    res = analyze_liveness(fn)
    return {"peak_live_units": res.peak_units, "peak_live_count": res.peak_count,
            "peak_live_pred": res.peak_pred, "n_blocks": res.n_blocks,
            "n_instructions": len(fn.instructions)}


# =========================================================================== instruction counts
#
# Counts of instructions in the listing, not executed ones, so a loop body counts once. They show
# which hardware paths a config uses; they are not a performance model.

#: PTX feature -> regex matched at the start of an instruction (after an optional predicate).
PTX_PATTERNS: dict[str, str] = {
    "mma": r"(?:mma\.sync|wgmma\.mma_async)\b",
    "ldmatrix": r"ldmatrix\.sync\b",
    "cp_async": r"cp\.async\.(?:ca|cg)\b",
    "cp_async_commit": r"cp\.async\.commit_group\b",
    "cp_async_wait": r"cp\.async\.wait_group\b",
    "ld_global_vec": r"ld\.global(?:\.\w+)*\.v[248]\b",
    "ld_global": r"ld\.global\b",
    "st_global": r"st\.global\b",
    "ld_shared": r"ld\.shared\b",
    "st_shared": r"st\.shared\b",
    "ld_local": r"ld\.local\b",
    "st_local": r"st\.local\b",
    "bar_sync": r"(?:bar|barrier)(?:\.cta)?\.sync\b",
}

#: SASS feature -> regex on the opcode.
SASS_PATTERNS: dict[str, str] = {
    "hmma": r"HMMA\b", "ldsm": r"LDSM\b", "ldgsts": r"LDGSTS\b", "ldg": r"LDG\b", "stg": r"STG\b",
    "lds": r"LDS\b", "sts": r"STS\b", "ldl": r"LDL\b", "stl": r"STL\b", "bar": r"BAR\b",
}

#: Counts stored in every sweep row.
PTX_ROW_FEATURES = ("mma", "ldmatrix", "cp_async", "ld_global_vec", "ld_local", "st_local", "bar_sync")
SASS_ROW_FEATURES = ("hmma", "ldsm", "ldgsts", "ldl", "stl", "bar")

_PTX_INSTR = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)", re.MULTILINE)
# A SASS line looks like "        /*0080*/                   HMMA.16816.F32 R4, R8, R12, R4 ;"
_SASS_INSTR = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def ptx_instructions(ptx: str) -> list[str]:
    """Instruction mnemonics in PTX text, without directives and comments."""
    text = re.sub(r"//.*", "", ptx)
    return [m.group(1) for m in _PTX_INSTR.finditer(text)]


def count_ptx(ptx: str) -> dict[str, int]:
    """Static count of each :data:`PTX_PATTERNS` feature."""
    instrs = ptx_instructions(ptx)
    return {name: sum(1 for ins in instrs if re.match(pat, ins)) for name, pat in PTX_PATTERNS.items()}


def sass_opcodes(sass: str) -> list[str]:
    """Opcodes in a ``cuobjdump -sass`` / ``nvdisasm`` listing."""
    return [m.group(1) for m in _SASS_INSTR.finditer(sass)]


def count_sass(sass: str) -> dict[str, int]:
    """Static count of each :data:`SASS_PATTERNS` opcode."""
    ops = sass_opcodes(sass)
    return {name: sum(1 for op in ops if re.match(pat, op)) for name, pat in SASS_PATTERNS.items()}


def row_features(ptx: str | None, sass: str | None) -> dict[str, int | None]:
    """The ``ptx_*`` / ``sass_*`` columns of a sweep row (``None`` without a listing)."""
    out: dict[str, int | None] = {}
    p = count_ptx(ptx) if ptx else None
    s = count_sass(sass) if sass else None
    for f in PTX_ROW_FEATURES:
        out[f"ptx_{f}"] = p[f] if p else None
    for f in SASS_ROW_FEATURES:
        out[f"sass_{f}"] = s[f] if s else None
    return out


def ptx_target(ptx: str) -> str | None:
    """The ``.target`` of a PTX module, e.g. ``sm_86``."""
    m = re.search(r"^\s*\.target\s+(\w+)", ptx, re.MULTILINE)
    return m.group(1) if m else None


# =========================================================================== comparing configs

IR_KEYS = ("ttir", "ttgir", "llir", "ptx")


def dump_ir(asm: dict[str, Any], out_dir: str | Path, name: str, sass: str | None = None) -> dict[str, str]:
    """Write every IR stage (and the SASS) of one compiled config to text files."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in IR_KEYS:
        if asm.get(k):
            p = d / f"{name}.{k}"
            p.write_text(str(asm[k]), encoding="utf-8")
            paths[k] = str(p)
    if sass:
        p = d / f"{name}.sass"
        p.write_text(sass, encoding="utf-8")
        paths["sass"] = str(p)
    return paths


def snippets(text: str, pattern: str, context: int = 1, max_hits: int = 4) -> str:
    """The first *max_hits* lines matching *pattern*, with *context* lines around each."""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if re.search(pattern, line)][:max_hits]
    out = []
    for i in hits:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        out.append("\n".join(f"{j + 1:6d}  {lines[j]}" for j in range(lo, hi)))
    return "\n   ...\n".join(out) if out else f"(no line matches {pattern!r})"


def compare_features(listings: dict[str, dict[str, str | None]]) -> pd.DataFrame:
    """Static counts side by side. *listings* maps a name to ``{"ptx": ..., "sass": ...}``."""
    rows = {}
    for name, listing in listings.items():
        row: dict[str, Any] = {}
        if listing.get("ptx"):
            row.update({f"ptx_{k}": v for k, v in count_ptx(listing["ptx"]).items()})
            row["ptx_target"] = ptx_target(listing["ptx"])
            row["ptx_lines"] = len(listing["ptx"].splitlines())
        if listing.get("sass"):
            row.update({f"sass_{k}": v for k, v in count_sass(listing["sass"]).items()})
        rows[name] = row
    return pd.DataFrame(rows)


def ttgir_layouts(ttgir: str) -> list[str]:
    """Distinct layout attributes (#blocked / #mma / #shared / #dot_op) declared in TTGIR."""
    return sorted(set(re.findall(r"^#\w+\s*=\s*#triton_gpu\.[^\n]+|^#\w+\s*=\s*#ttg\.[^\n]+", ttgir, re.MULTILINE)))
