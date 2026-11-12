"""Candidate-order modes for the order-sensitivity test (paper Table VI)."""
from __future__ import annotations

import random
from typing import Sequence, TypeVar

T = TypeVar("T")

ORDER_MODES = ("original", "reversed", "random")


def apply_order(items: Sequence[T], mode: str, qid: str = "", seed: int = 0) -> list[T]:
    """Reorder a candidate list.

    "original": keep the first-stage (e.g. BM25) order
    "reversed": worst candidate first
    "random":   a shuffle that is deterministic given (seed, qid), so runs are reproducible
    """
    out = list(items)
    if mode == "original":
        return out
    if mode == "reversed":
        return out[::-1]
    if mode == "random":
        random.Random(f"{seed}-{qid}").shuffle(out)
        return out
    raise ValueError(f"unknown order mode '{mode}', expected one of {ORDER_MODES}")
