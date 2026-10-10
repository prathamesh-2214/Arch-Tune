"""STRETCH (Phase 9): a small PTX parser, CFG builder and liveness analysis.

Handles the subset of PTX that Triton emits: ``.reg`` declarations, labels, predicated
instructions, vector operands ``{%r1, %r2}``, address operands ``[%rd1+16]`` and branches.
"""
