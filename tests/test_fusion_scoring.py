import pytest
import torch

from resrank.model.fusion import fuse
from resrank.model.scoring import cosine_scores, rank_order


def test_residual_is_sum():
    h, e = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    assert torch.allclose(fuse(h, e, "residual"), h + e)


def test_ablation_modes():
    h, e = torch.randn(5, 8), torch.randn(5, 8)
    assert torch.equal(fuse(h, e, "no_residual"), h)
    assert torch.equal(fuse(h, e, "no_hidden"), e)


def test_fuse_rejects_bad_shapes_and_modes():
    with pytest.raises(ValueError):
        fuse(torch.randn(3, 4), torch.randn(3, 5))
    with pytest.raises(ValueError):
        fuse(torch.randn(3, 4), torch.randn(3, 4), "bogus")


def test_cosine_matches_manual_example_from_plan():
    # The worked example in the plan: passage 1 should score ~0.997, passage 2 ~0.23.
    h_eos = torch.tensor([1.0, 0.5, 0.0, 0.1])
    r = torch.tensor([[1.4, 0.7, 0.1, 0.2], [-0.1, 0.9, 0.9, -0.2]])
    s = cosine_scores(h_eos, r)
    assert s.shape == (2,)
    assert s[0] == pytest.approx(0.997, abs=1e-3)
    assert s[1] == pytest.approx(0.23, abs=1e-2)
    assert rank_order(s).tolist() == [0, 1]


def test_cosine_batched_and_bounded():
    s = cosine_scores(torch.randn(3, 16), torch.randn(3, 7, 16))
    assert s.shape == (3, 7)
    assert (s <= 1 + 1e-6).all() and (s >= -1 - 1e-6).all()


def test_cosine_is_scale_invariant():
    h, r = torch.randn(16), torch.randn(4, 16)
    assert torch.allclose(cosine_scores(h, r), cosine_scores(3.0 * h, 0.1 * r), atol=1e-6)


def test_cosine_shape_errors():
    with pytest.raises(ValueError):
        cosine_scores(torch.randn(16), torch.randn(16))
    with pytest.raises(ValueError):
        cosine_scores(torch.randn(16), torch.randn(4, 8))


def test_rank_order_descending_and_stable():
    scores = torch.tensor([0.1, 0.9, 0.9, -0.5])
    assert rank_order(scores).tolist() == [1, 2, 0, 3]  # tie keeps input order
