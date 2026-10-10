"""Regex-level static instruction counts from PTX and SASS text.

Counts are *static* (instructions in the listing, not dynamically executed); loop bodies
count once. They are evidence for "which hardware paths does this config use" (tensor cores,
``ldmatrix``, ``cp.async``, vector loads, spills, barriers), not a performance model.
"""

from __future__ import annotations

import re

#: PTX feature -> regex matched at the start of an instruction (after optional predicate).
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

#: SASS opcode -> regex on the opcode token.
SASS_PATTERNS: dict[str, str] = {
    "hmma": r"HMMA\b",
    "ldsm": r"LDSM\b",
    "ldgsts": r"LDGSTS\b",
    "ldg": r"LDG\b",
    "stg": r"STG\b",
    "lds": r"LDS\b",
    "sts": r"STS\b",
    "ldl": r"LDL\b",
    "stl": r"STL\b",
    "bar": r"BAR\b",
}

#: PTX columns that go into every sweep row (spec §7).
PTX_ROW_FEATURES = ("mma", "ldmatrix", "cp_async", "ld_global_vec", "ld_local", "st_local", "bar_sync")
SASS_ROW_FEATURES = ("hmma", "ldsm", "ldgsts", "ldl", "stl", "bar")

_PTX_INSTR = re.compile(r"^\s*(?:@!?%\w+\s+)?([a-z][\w.:]*)", re.MULTILINE)
# SASS lines look like: "        /*0080*/                   HMMA.16816.F32 R4, R8, R12, R4 ;"
_SASS_INSTR = re.compile(r"/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)")


def ptx_instructions(ptx: str) -> list[str]:
    """Instruction mnemonics (with modifiers) in PTX text, skipping directives and comments."""
    text = re.sub(r"//.*", "", ptx)
    return [m.group(1) for m in _PTX_INSTR.finditer(text)]


def count_ptx(ptx: str) -> dict[str, int]:
    """Static counts of each :data:`PTX_PATTERNS` feature in a PTX listing."""
    instrs = ptx_instructions(ptx)
    return {name: sum(1 for ins in instrs if re.match(pat, ins)) for name, pat in PTX_PATTERNS.items()}


def sass_opcodes(sass: str) -> list[str]:
    """Opcodes (with modifiers) in a ``cuobjdump -sass`` / ``nvdisasm`` listing."""
    return [m.group(1) for m in _SASS_INSTR.finditer(sass)]


def count_sass(sass: str) -> dict[str, int]:
    """Static counts of each :data:`SASS_PATTERNS` opcode in a SASS listing."""
    ops = sass_opcodes(sass)
    return {name: sum(1 for op in ops if re.match(pat, op)) for name, pat in SASS_PATTERNS.items()}


def row_features(ptx: str | None, sass: str | None) -> dict[str, int | None]:
    """``ptx_*`` / ``sass_*`` columns for a sweep row (``None`` when the listing is missing)."""
    out: dict[str, int | None] = {}
    p = count_ptx(ptx) if ptx else None
    s = count_sass(sass) if sass else None
    for f in PTX_ROW_FEATURES:
        out[f"ptx_{f}"] = p[f] if p else None
    for f in SASS_ROW_FEATURES:
        out[f"sass_{f}"] = s[f] if s else None
    return out


def ptx_target(ptx: str) -> str | None:
    """The ``.target`` directive (e.g. ``sm_75``) of a PTX module."""
    m = re.search(r"^\s*\.target\s+(\w+)", ptx, re.MULTILINE)
    return m.group(1) if m else None
