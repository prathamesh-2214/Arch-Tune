"""Basic blocks and control-flow graph of a parsed PTX function."""

from __future__ import annotations

from dataclasses import dataclass, field

from archtune.ptx.parser import Function


@dataclass
class Block:
    """A basic block: instruction index range ``[start, end)`` and successor block ids."""

    id: int
    start: int
    end: int
    succs: list[int] = field(default_factory=list)
    preds: list[int] = field(default_factory=list)


def build_cfg(fn: Function) -> list[Block]:
    """Split *fn* into basic blocks and connect them.

    Leaders: the first instruction, every label target, and every instruction following a
    branch or return. A guarded branch (``@%p bra``) falls through as well as jumping.
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
