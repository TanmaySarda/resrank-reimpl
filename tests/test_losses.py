import math

import pytest
import torch

from resrank.model.losses import info_nce_loss, ranknet_loss, total_loss


# ---- InfoNCE ---------------------------------------------------------------
def test_infonce_matches_manual_softmax():
    # cos values 0.80 / 0.60 / 0.75 at tau 0.05 (the example in the conversation)
    def vec(c):  # unit vector with cosine c to the query [1, 0]
        return [c, math.sqrt(1 - c * c)]

    q = torch.tensor([[1.0, 0.0]])
    pos = torch.tensor([vec(0.80)])
    neg = torch.tensor([[vec(0.60), vec(0.75)]])
    tau = 0.05
    exps = [math.exp(c / tau) for c in (0.80, 0.60, 0.75)]
    expected = -math.log(exps[0] / sum(exps))
    assert info_nce_loss(q, pos, neg, tau).item() == pytest.approx(expected, rel=1e-4)


def test_infonce_prefers_positive_closer():
    q = torch.tensor([[1.0, 0.0]])
    neg = torch.tensor([[[0.0, 1.0], [-1.0, 0.0]]])
    good = info_nce_loss(q, torch.tensor([[1.0, 0.1]]), neg)
    bad = info_nce_loss(q, torch.tensor([[0.0, 1.0]]), neg)
    assert good < bad


def test_infonce_neg_mask_ignores_padding():
    q = torch.randn(2, 8)
    pos = torch.randn(2, 8)
    neg = torch.randn(2, 5, 8)
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    masked = info_nce_loss(q, pos, neg, neg_mask=mask)
    # Dropping the padded negatives by hand must give the same loss for sample 0.
    manual0 = info_nce_loss(q[:1], pos[:1], neg[:1, :3])
    manual1 = info_nce_loss(q[1:], pos[1:], neg[1:])
    assert masked.item() == pytest.approx((manual0.item() + manual1.item()) / 2, rel=1e-5)


# ---- RankNet ---------------------------------------------------------------
def test_ranknet_zero_when_ordering_correct_and_far_apart():
    scores = torch.tensor([[0.9, 0.0, -0.9]])
    ranks = torch.tensor([[0, 1, 2]])
    assert ranknet_loss(scores, ranks).item() < 1e-6


def test_ranknet_large_when_ordering_reversed():
    scores = torch.tensor([[-0.9, 0.0, 0.9]])
    ranks = torch.tensor([[0, 1, 2]])
    assert ranknet_loss(scores, ranks).item() > 50


def test_ranknet_matches_manual_pair_sum():
    scores = torch.tensor([[0.2, 0.5]])
    ranks = torch.tensor([[0, 1]])            # passage 0 should beat passage 1
    tau = 0.05
    expected = math.log(1 + math.exp((0.5 - 0.2) / tau))
    assert ranknet_loss(scores, ranks, tau).item() == pytest.approx(expected, rel=1e-5)


def test_ranknet_ties_make_no_pairs():
    scores = torch.tensor([[0.1, 0.9, 0.5]])
    ranks = torch.tensor([[1, 1, 1]])
    assert ranknet_loss(scores, ranks).item() == 0.0


def test_ranknet_mask_and_mean_reduction():
    scores = torch.tensor([[0.9, 0.0, 123.0]])
    ranks = torch.tensor([[0, 1, 2]])
    mask = torch.tensor([[True, True, False]])  # third passage is padding
    masked = ranknet_loss(scores, ranks, mask=mask)
    two_only = ranknet_loss(scores[:, :2], ranks[:, :2])
    assert masked.item() == pytest.approx(two_only.item(), rel=1e-6)
    s = torch.tensor([[0.0, 0.1, 0.2]])
    r = torch.tensor([[0, 1, 2]])
    assert ranknet_loss(s, r, pair_reduction="mean").item() == pytest.approx(
        ranknet_loss(s, r, pair_reduction="sum").item() / 3, rel=1e-5
    )


def test_ranknet_gradient_pushes_scores_toward_correct_order():
    scores = torch.tensor([[0.0, 0.0]], requires_grad=True)
    ranks = torch.tensor([[0, 1]])
    ranknet_loss(scores, ranks).backward()
    # Better passage (index 0) should be pushed up (negative grad), worse one down.
    assert scores.grad[0, 0] < 0 < scores.grad[0, 1]


def test_losses_reject_bad_args():
    with pytest.raises(ValueError):
        ranknet_loss(torch.zeros(1, 3), torch.zeros(1, 2))
    with pytest.raises(ValueError):
        ranknet_loss(torch.zeros(1, 3), torch.zeros(1, 3), tau=0)
    with pytest.raises(ValueError):
        info_nce_loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 1, 2), tau=-1)


def test_total_loss_weighting():
    a, b = torch.tensor(2.0), torch.tensor(3.0)
    assert total_loss(a, b, lam=0.1).item() == pytest.approx(0.1 * 2.0 + 3.0)
