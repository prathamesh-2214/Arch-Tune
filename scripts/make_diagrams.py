"""Draw the concept diagrams used by README.md as SVG files in docs/img/.

    python scripts/make_diagrams.py

The diagrams contain no measured numbers (those live in generated charts), so they only need
re-drawing when the explanation changes. Each SVG has its own white card background so it reads
well on GitHub's light and dark themes.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"
FONT = "Segoe UI, Inter, Helvetica Neue, Arial, sans-serif"
MONO = "Cascadia Mono, Consolas, DejaVu Sans Mono, monospace"
INK, INK2, LINE = "#16181d", "#4a4f59", "#8a909a"

#: (fill, stroke) per role
C = {
    "gray": ("#f4f5f7", "#b9bfc8"),
    "blue": ("#e8f1fc", "#2a78d6"),
    "violet": ("#eeecf9", "#4a3aa7"),
    "orange": ("#fdeee7", "#eb6834"),
    "red": ("#fdecec", "#e34948"),
    "aqua": ("#e6f6f0", "#1baf7a"),
    "green": ("#eef7dd", "#76b900"),
    "yellow": ("#fdf4dc", "#eda100"),
}


class Svg:
    """Tiny SVG builder."""

    def __init__(self, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.parts: list[str] = [
            f'<rect x="1" y="1" width="{w - 2}" height="{h - 2}" rx="16" fill="#ffffff" stroke="#e3e5e8"/>',
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker>'
            '<marker id="ahg" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#76b900"/></marker></defs>',
        ]

    def text(self, x: float, y: float, s: str, size: float = 14, color: str = INK, weight: str = "400",
             anchor: str = "start", mono: bool = False, italic: bool = False) -> None:
        style = ' font-style="italic"' if italic else ""
        self.parts.append(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                          f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}"{style}>{escape(s)}</text>')

    def box(self, x: float, y: float, w: float, h: float, role: str, title: str, sub: str | list[str] = "",
            title_size: float = 15, sub_size: float = 12.5, align: str = "center", dashed: bool = False,
            stroke_w: float = 2) -> None:
        fill, stroke = C[role]
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
                          f'stroke="{stroke}" stroke-width="{stroke_w}"{dash}/>')
        subs = [sub] if isinstance(sub, str) else sub
        subs = [s for s in subs if s]
        total = title_size + len(subs) * (sub_size + 5)
        ty = y + (h - total) / 2 + title_size - 2
        ax = x + w / 2 if align == "center" else x + 14
        anchor = "middle" if align == "center" else "start"
        self.text(ax, ty, title, title_size, INK, "700", anchor)
        for i, s in enumerate(subs):
            self.text(ax, ty + (i + 1) * (sub_size + 5), s, sub_size, INK2, "400", anchor)

    def arrow(self, x1: float, y1: float, x2: float, y2: float, green: bool = False, dashed: bool = False,
              width: float = 2) -> None:
        color, m = ("#76b900", "ahg") if green else (LINE, "ah")
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
                          f'stroke-width="{width}" marker-end="url(#{m})"{dash}/>')

    def path(self, d: str, color: str = LINE, green: bool = False, dashed: bool = False) -> None:
        m = "ahg" if green else "ah"
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2" marker-end="url(#{m})"{dash}/>')

    def pill(self, x: float, y: float, w: float, h: float, role: str, label: str, size: float = 12) -> None:
        fill, stroke = C[role]
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{h / 2}" fill="{stroke}"/>')
        self.text(x + w / 2, y + h / 2 + size / 3, label, size, "#ffffff", "700", "middle")

    def save(self, name: str) -> Path:
        OUT.mkdir(parents=True, exist_ok=True)
        p = OUT / name
        p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                     f'viewBox="0 0 {self.w} {self.h}">' + "".join(self.parts) + "</svg>", encoding="utf-8")
        return p


def stack() -> Path:
    """torch.compile → Triton → PTX → ptxas → GPU, and where ArchTune acts."""
    s = Svg(960, 676)
    s.text(40, 46, "Where ArchTune sits in the torch.compile stack", 22, INK, "700")
    s.text(40, 72, "Every candidate kernel config goes down this whole stack before autotuning can time it.", 14, INK2)
    rows = [
        ("gray", "PyTorch model", "your nn.Module / functions"),
        ("blue", "TorchDynamo", "captures the Python program as an FX graph"),
        ("blue", "TorchInductor", "fuses ops and generates Triton kernels (GEMM templates)"),
        ("orange", "Autotuning  (max-autotune / triton.autotune)", "compile + benchmark every config, keep the fastest"),
        ("violet", "Triton compiler", "TTIR  >  TTGIR (layouts, pipelining)  >  LLVM IR  >  PTX"),
        ("red", "ptxas  (NVIDIA PTX compiler)", "register allocation, scheduling, spills  >  SASS"),
        ("aqua", "GPU", "driver loads the cubin, checks resources, runs on the SMs"),
    ]
    x, w, h, y0, step = 40, 560, 60, 100, 76
    for i, (role, title, sub) in enumerate(rows):
        y = y0 + i * step
        s.box(x, y, w, h, role, title, sub, align="left")
        if i:
            s.arrow(x + w / 2, y - step + h + 2, x + w / 2, y - 3)
    # ArchTune callout next to the autotuning box
    ay = y0 + 3 * step
    s.box(660, ay - 34, 270, 128, "green", "ArchTune", ["predicts registers, spills,", "shared memory, occupancy", "BEFORE compiling; prunes", "doomed configs"], stroke_w=3)
    s.arrow(660, ay + h / 2, x + w + 6, ay + h / 2, green=True, width=3)
    # dashed "predicts" link to ptxas / GPU
    py = y0 + 5 * step + h / 2
    s.path(f"M 795 {ay + 94} L 795 {py} L {x + w + 6} {py}", color="#76b900", green=True, dashed=True)
    s.text(805, py - 40, "predicts what", 13, "#4f7f00", "600")
    s.text(805, py - 22, "ptxas will do", 13, "#4f7f00", "600")
    # brackets
    s.text(40, 676 - 20, "Expensive per config: Triton lowering + ptxas run once for every candidate.", 13, INK2, italic=True)
    return s.save("stack.svg")


def flow() -> Path:
    """Default autotuning vs ArchTune-pruned autotuning."""
    s = Svg(960, 456)
    s.text(40, 46, "Default autotuning vs ArchTune", 22, INK, "700")
    # lane A
    s.pill(40, 74, 220, 28, "gray", "DEFAULT  triton.autotune")
    ya, h = 116, 84
    xs = [40, 270, 500, 730]
    a = [("gray", "All configs", ["every tile / warps /", "stages combination"]),
         ("red", "Compile ALL", ["Triton + ptxas for each;", "doomed ones fail late"]),
         ("orange", "Benchmark ALL", ["time every kernel", "that compiled"]),
         ("aqua", "Fastest kernel", ["after paying for", "every compile"])]
    for i, (role, t, sub) in enumerate(a):
        s.box(xs[i], ya, 190, h, role, t, sub)
        if i:
            s.arrow(xs[i - 1] + 192, ya + h / 2, xs[i] - 4, ya + h / 2)
    # lane B
    s.pill(40, 238, 220, 28, "green", "ARCHTUNE  pruned autotune")
    yb = 280
    xb = [40 + i * 150 for i in range(6)]
    b = [("gray", "All configs", ["same", "candidates"]),
         ("green", "Predict", ["no compiling:", "pure arithmetic"]),
         ("yellow", "Filter", ["drop: won't fit", "drop: will spill"]),
         ("blue", "Rank", ["roofline +", "wave score"]),
         ("orange", "Compile top-k", ["only k configs", "reach ptxas"]),
         ("aqua", "Fastest", ["same quality,", "fraction of time"])]
    for i, (role, t, sub) in enumerate(b):
        s.box(xb[i], yb, 132, h, role, t, sub, title_size=14, sub_size=11.5)
        if i:
            s.arrow(xb[i - 1] + 134, yb + h / 2, xb[i] - 4, yb + h / 2)
    s.text(40, 410, "The filter removes configs that would overflow shared memory or spill registers, which are", 13, INK2)
    s.text(40, 430, "also the slowest ones to compile. Ranking then picks a short list; only that list is compiled and timed.", 13, INK2)
    return s.save("archtune_flow.svg")


def predictor() -> Path:
    """Inputs → cost model → predictions → decision."""
    s = Svg(960, 522)
    s.text(40, 46, "What the predictor computes (before any compilation)", 22, INK, "700")
    ins = [("blue", "Config knobs", ["BLOCK_M, BLOCK_N, BLOCK_K", "num_warps, num_stages"]),
           ("violet", "Problem shape", ["M x N x K of the matmul"]),
           ("aqua", "GPU limits", ["registers, shared memory,", "SMs, peak FLOP/s, bandwidth"])]
    for i, (role, t, sub) in enumerate(ins):
        y = 92 + i * 112
        s.box(40, y, 250, 88, role, t, sub, align="left")
        s.arrow(292, y + 44, 366, 255)
    s.box(370, 190, 220, 130, "green", "Cost model", ["formulas from hardware", "rules + constants fitted", "on training shapes"], stroke_w=3)
    outs = [("yellow", "Shared memory / block", "stages x tile buffers"),
            ("red", "Registers / thread", "accumulators + operands"),
            ("red", "Will it spill?", "estimate vs learned threshold"),
            ("blue", "Occupancy", "blocks per SM (CUDA rules)"),
            ("blue", "Wave efficiency", "how evenly tiles fill SMs"),
            ("orange", "Time score", "roofline / wave efficiency")]
    for i, (role, t, sub) in enumerate(outs):
        y = 84 + i * 62
        s.box(670, y, 250, 52, role, t, sub, title_size=13.5, sub_size=11.5, align="left")
        s.arrow(592, 255, 666, y + 26)
    s.box(40, 450, 880, 50, "gray", "Decision:  drop configs that won't fit or will spill  ·  rank the rest by time score  ·  compile only the top k",
          title_size=14)
    return s.save("predictor.svg")


def knobs() -> Path:
    """The tunable knobs and the resources they consume."""
    s = Svg(960, 392)
    s.text(40, 46, "What autotuning is choosing, and what each choice costs", 22, INK, "700")
    cards = [
        ("blue", "Tile size", "BLOCK_M x BLOCK_N", ["bigger tile = more data", "reuse, fewer DRAM reads"], ["more registers and", "shared memory; idle", "SMs on small shapes"]),
        ("violet", "K step", "BLOCK_K", ["fewer loop", "iterations"], ["more operand", "registers and shared", "memory per stage"]),
        ("orange", "Warps", "num_warps", ["more threads share", "the tile, fewer", "registers each"], ["more synchronization;", "too few warps cannot", "hide latency"]),
        ("aqua", "Pipelining", "num_stages", ["hides memory latency", "(async copies on", "Ampere and newer)"], ["shared memory per", "stage, which can", "lower occupancy"]),
        ("yellow", "Tile order", "GROUP_SIZE_M", ["better L2 cache", "reuse between", "neighbouring tiles"], ["mostly free;", "fixed in this", "study"]),
    ]
    w, gap, x0, y0 = 168, 15, 40, 76
    for i, (role, title, knob, gain, cost) in enumerate(cards):
        x = x0 + i * (w + gap)
        fill, stroke = C[role]
        s.parts.append(f'<rect x="{x}" y="{y0}" width="{w}" height="296" rx="14" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        s.parts.append(f'<rect x="{x}" y="{y0}" width="{w}" height="56" rx="14" fill="{stroke}"/>')
        s.parts.append(f'<rect x="{x}" y="{y0 + 40}" width="{w}" height="16" fill="{stroke}"/>')
        s.text(x + w / 2, y0 + 26, title, 16, "#ffffff", "700", "middle")
        s.text(x + w / 2, y0 + 46, knob, 11.5, "#ffffff", "400", "middle", mono=True)
        s.text(x + 12, y0 + 84, "GAIN", 11, stroke, "700")
        for j, line in enumerate(gain):
            s.text(x + 12, y0 + 104 + j * 18, line, 12.5, INK)
        s.text(x + 12, y0 + 176, "COST", 11, stroke, "700")
        for j, line in enumerate(cost):
            s.text(x + 12, y0 + 196 + j * 18, line, 12.5, INK)
    return s.save("knobs.svg")


def compile_pipeline() -> Path:
    """Inside one compile: which stage turns each knob into resource usage."""
    s = Svg(960, 318)
    s.text(40, 46, "Inside one compile: where registers, spills and shared memory are decided", 22, INK, "700")
    stages = [("violet", "TTIR", ["tile ops:", "load, dot, store"]),
              ("violet", "TTGIR", ["layouts, tensor cores,", "pipelining buffers"]),
              ("violet", "LLVM IR", ["per-thread", "code"]),
              ("blue", "PTX", ["virtual ISA,", "unlimited registers"]),
              ("red", "ptxas", ["allocates real", "registers (max 255)"]),
              ("aqua", "SASS", ["machine code;", "spills = LDL/STL"])]
    w, gap, x0, y = 132, 18, 40, 92
    for i, (role, t, sub) in enumerate(stages):
        x = x0 + i * (w + gap)
        s.box(x, y, w, 92, role, t, sub, title_size=15, sub_size=11.5)
        if i:
            s.arrow(x - gap + 1, y + 46, x - 3, y + 46)
    notes = [(x0 + 1 * (w + gap), "tile sizes + warps set how many", "accumulators each thread holds"),
             (x0 + 1 * (w + gap), "num_stages sets shared-memory", "buffers + async copies"),
             (x0 + 4 * (w + gap), "spills only appear here,", "after register allocation")]
    for k, (x, a, b) in enumerate(notes):
        yy = 222 + (k % 2) * 46 if k < 2 else 222
        role = "yellow" if k < 2 else "red"
        fill, stroke = C[role]
        s.parts.append(f'<rect x="{x - 8}" y="{yy - 18}" width="{w + 150}" height="40" rx="8" fill="{fill}" stroke="{stroke}"/>')
        s.text(x, yy, a, 12, INK)
        s.text(x, yy + 15, b, 12, INK)
    return s.save("compile_pipeline.svg")


if __name__ == "__main__":
    for f in (stack, flow, predictor, knobs, compile_pipeline):
        print("wrote", f().relative_to(OUT.parents[1]))
