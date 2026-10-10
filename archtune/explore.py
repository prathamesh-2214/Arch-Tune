"""Helpers for IR / PTX exploration (Phase 3, notebook 01)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from archtune import ptx_features

IR_KEYS = ("ttir", "ttgir", "llir", "ptx")


def dump_ir(asm: dict[str, Any], out_dir: str | Path, name: str, sass: str | None = None) -> dict[str, str]:
    """Write every IR stage (and SASS) of one compiled config to text files."""
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
    """First *max_hits* lines matching *pattern* with *context* lines around each."""
    lines = text.splitlines()
    hits = [i for i, line in enumerate(lines) if re.search(pattern, line)][:max_hits]
    out = []
    for i in hits:
        lo, hi = max(0, i - context), min(len(lines), i + context + 1)
        out.append("\n".join(f"{j + 1:6d}  {lines[j]}" for j in range(lo, hi)))
    return "\n   ...\n".join(out) if out else f"(no line matches {pattern!r})"


def compare_features(listings: dict[str, dict[str, str | None]]) -> pd.DataFrame:
    """Side-by-side static instruction counts. *listings*: name -> {"ptx": ..., "sass": ...}."""
    rows = {}
    for name, l in listings.items():
        row: dict[str, Any] = {}
        if l.get("ptx"):
            row.update({f"ptx_{k}": v for k, v in ptx_features.count_ptx(l["ptx"]).items()})
            row["ptx_target"] = ptx_features.ptx_target(l["ptx"])
            row["ptx_lines"] = len(l["ptx"].splitlines())
        if l.get("sass"):
            row.update({f"sass_{k}": v for k, v in ptx_features.count_sass(l["sass"]).items()})
        rows[name] = row
    return pd.DataFrame(rows)


def ttgir_layouts(ttgir: str) -> list[str]:
    """Distinct layout attributes (#blocked / #mma / #shared / #dot_op) declared in TTGIR."""
    return sorted(set(re.findall(r"^#\w+\s*=\s*#triton_gpu\.[^\n]+|^#\w+\s*=\s*#ttg\.[^\n]+", ttgir, re.MULTILINE)))
