"""Ranking metrics: nDCG@k, Recall@k, and a paired bootstrap for comparing two runs.

nDCG uses linear gain (gain = relevance grade), the same convention as trec_eval's ndcg_cut.
Queries are taken from the qrels; a qrels query with no run entry scores 0.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from resrank.io_utils import Qrels, Run


def ndcg_at_k(ranked_docids: Sequence[str], qrel: dict[str, int], k: int = 10) -> float:
    """nDCG@k for one query. `qrel` maps docid -> graded relevance (0 = not relevant)."""
    dcg = sum(
        qrel.get(doc, 0) / math.log2(i + 2)
        for i, doc in enumerate(ranked_docids[:k])
        if qrel.get(doc, 0) > 0
    )
    ideal = sorted((r for r in qrel.values() if r > 0), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_docids: Sequence[str], qrel: dict[str, int], k: int = 100) -> float:
    """Fraction of relevant documents (grade > 0) found in the top k."""
    relevant = {d for d, r in qrel.items() if r > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_docids[:k])) / len(relevant)


def evaluate_run(run: Run, qrels: Qrels, metric: str = "ndcg", k: int = 10) -> dict:
    """Evaluate a run against qrels.

    Returns:
        {"mean": float, "per_query": {qid: value}, "n_queries": int}
    """
    fn = {"ndcg": ndcg_at_k, "recall": recall_at_k}.get(metric)
    if fn is None:
        raise ValueError("metric must be 'ndcg' or 'recall'")
    per_query: dict[str, float] = {}
    for qid, qrel in qrels.items():
        ranked = [doc for doc, _ in run.get(qid, [])]
        per_query[qid] = fn(ranked, qrel, k)
    mean = float(np.mean(list(per_query.values()))) if per_query else 0.0
    return {"mean": mean, "per_query": per_query, "n_queries": len(per_query)}


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0
) -> dict:
    """Paired bootstrap over queries for the difference mean(a) - mean(b).

    `a` and `b` are per-query scores aligned by query. Returns the observed mean difference,
    a 95% percentile interval, and the fraction of resamples where a > b.
    """
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a_arr.shape != b_arr.shape or a_arr.ndim != 1 or a_arr.size == 0:
        raise ValueError("a and b must be non-empty 1-D sequences of equal length")
    diffs = a_arr - b_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot = diffs[idx].mean(axis=1)
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "prob_a_better": float((boot > 0).mean()),
    }
