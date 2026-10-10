"""Parse the subset of PTX that Triton emits into register declarations and instructions."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Opcodes (first dotted component) whose operands are all sources (no register destination).
NO_DST_OPCODES = {"st", "bar", "barrier", "bra", "ret", "exit", "membar", "fence", "red",
                  "prefetch", "trap", "cp", "stmatrix", "wgmma", "griddepcontrol", "mbarrier"}

#: Register width in bits by PTX type suffix of a ``.reg`` declaration.
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
        """``bra`` instruction."""
        return self.base == "bra"

    @property
    def is_return(self) -> bool:
        """``ret`` / ``exit``."""
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
    """Split an operand list on top-level commas (not inside ``{}`` or ``[]``)."""
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
    """Virtual registers referenced in an operand (special registers like %tid.x excluded)."""
    regs = []
    for tok in _REG_TOKEN.findall(operand):
        prefix = re.match(r"%[A-Za-z_$]+", tok).group(0)
        if tok in declared or prefix in declared:
            regs.append(tok)
    return regs


def parse_instruction(stmt: str, index: int, declared: set[str]) -> Instruction:
    """Parse one statement (without the trailing ``;``)."""
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
    if first.startswith("["):  # address destination, e.g. st-like forms
        ins.srcs += _regs_in(first, declared)
    else:
        # "setp ... %p1|%p2" writes two predicates.
        ins.dsts += _regs_in(first, declared)
    for op in rest:
        ins.srcs += _regs_in(op, declared)
    return ins


def parse_function(ptx: str, name: str | None = None) -> Function:
    """Parse the first (or named) ``.entry`` / ``.func`` of a PTX module."""
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
    # Labels end with ':'; statements end with ';'. Directive lines start with '.'.
    for chunk in re.split(r";", body):
        lines = chunk.strip().splitlines()
        stmt_lines = []
        for line in lines:
            if line.strip() in ("{", "}"):  # scope braces around inline asm blocks
                continue
            lm = _LABEL.match(line)
            if lm:
                labels[lm.group(1)] = len(instrs)
            elif line.strip():
                stmt_lines.append(line.strip())
        stmt = " ".join(stmt_lines).strip()
        # A label may share a line with the following statement: "$L__BB0_2: add.s32 ..."
        while True:
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
