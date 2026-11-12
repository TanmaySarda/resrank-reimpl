import math

import pytest

from resrank.eval.metrics import evaluate_run, ndcg_at_k, paired_bootstrap, recall_at_k
from resrank.eval.ordering import apply_order


def test_ndcg_perfect_and_zero():
    qrel = {"a": 1, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], qrel) == pytest.approx(1.0)
    assert ndcg_at_k(["c", "d"], qrel) == 0.0
    assert ndcg_at_k(["a"], {}) == 0.0


def test_ndcg_known_value():
    # One relevant doc (grade 1) at rank 3: DCG = 1/log2(4) = 0.5, IDCG = 1.
    assert ndcg_at_k(["x", "y", "a"], {"a": 1}) == pytest.approx(0.5)


def test_ndcg_graded_gain_is_linear():
    qrel = {"a": 2, "b": 1}
    # Best order a, b: 1.0. Swapped: (1/1 + 2/log2(3)) / (2/1 + 1/log2(3)).
    expected = (1 + 2 / math.log2(3)) / (2 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "b"], qrel) == pytest.approx(1.0)
    assert ndcg_at_k(["b", "a"], qrel) == pytest.approx(expected)


def test_ndcg_cutoff():
    assert ndcg_at_k(["x"] * 10 + ["a"], {"a": 1}, k=10) == 0.0


def test_recall():
    assert recall_at_k(["a", "x", "b"], {"a": 1, "b": 1, "c": 1}, k=3) == pytest.approx(2 / 3)
    assert recall_at_k(["a"], {"a": 0}, k=3) == 0.0


def test_evaluate_run_counts_missing_queries_as_zero():
    qrels = {"q1": {"a": 1}, "q2": {"b": 1}}
    run = {"q1": [("a", 1.0)]}
    out = evaluate_run(run, qrels)
    assert out["per_query"] == {"q1": 1.0, "q2": 0.0}
    assert out["mean"] == 0.5 and out["n_queries"] == 2


def test_paired_bootstrap_detects_clear_difference():
    a = [0.6] * 50 + [0.7] * 50
    b = [0.4] * 100
    r = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert r["mean_diff"] == pytest.approx(0.25)
    assert r["ci_low"] > 0 and r["prob_a_better"] == 1.0


def test_paired_bootstrap_no_difference_interval_spans_zero():
    import random
    rng = random.Random(0)
    a = [rng.random() for _ in range(100)]
    b = [x + rng.uniform(-0.1, 0.1) for x in a]
    r = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert r["ci_low"] < 0 < r["ci_high"]


def test_apply_order_modes():
    items = list(range(10))
    assert apply_order(items, "original") == items
    assert apply_order(items, "reversed") == items[::-1]
    r1 = apply_order(items, "random", qid="q1", seed=0)
    assert r1 == apply_order(items, "random", qid="q1", seed=0)
    assert r1 != apply_order(items, "random", qid="q2", seed=0)
    assert sorted(r1) == items
    with pytest.raises(ValueError):
        apply_order(items, "bogus")
