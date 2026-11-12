"""Candidate-list manipulation: shrink a list (memory) and shuffle it (avoid position shortcuts)."""
from __future__ import annotations

import random

from resrank.schemas import RankingSample


def _remap(sample: RankingSample, keep_order: list[int]) -> RankingSample:
    """Build a sample whose candidates are `sample.passages[i] for i in keep_order`, in that order."""
    new_index = {old: new for new, old in enumerate(keep_order)}
    return RankingSample(
        query=sample.query,
        passages=[sample.passages[i] for i in keep_order],
        ranking=[new_index[i] for i in sample.ranking if i in new_index],
        pos_index=new_index[sample.pos_index],
        source=sample.source,
        qid=sample.qid,
    )


def subsample_candidates(sample: RankingSample, n: int, rng: random.Random) -> RankingSample:
    """Keep the positive plus a random (n-1) of the others, preserving the original candidate order
    and the relative ranking. Returns the sample unchanged if it already has <= n passages."""
    if n < 2:
        raise ValueError("need at least 2 candidates (positive + one negative)")
    if sample.n <= n:
        return sample
    chosen = rng.sample(sample.negative_indices, n - 1)
    return _remap(sample, sorted(chosen + [sample.pos_index]))


def shuffle_candidates(sample: RankingSample, rng: random.Random) -> RankingSample:
    """Randomly permute the candidate order (ranking and positive follow their passages)."""
    order = list(range(sample.n))
    rng.shuffle(order)
    return _remap(sample, order)
