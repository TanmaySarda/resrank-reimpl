"""Cosine-similarity scoring (paper Eq. 5): s_i = cos(h_eos, r_i)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def cosine_scores(h_eos: torch.Tensor, r: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Score every passage against the global aggregation embedding.

    Args:
        h_eos: hidden state at the EOS position,       shape (..., d)
        r:     fused passage representations,          shape (..., n, d)

    Returns:
        scores of shape (..., n), each in [-1, 1].
    """
    if r.dim() != h_eos.dim() + 1:
        raise ValueError(
            f"r must have one more dim than h_eos, got h_eos {tuple(h_eos.shape)}, r {tuple(r.shape)}"
        )
    if h_eos.shape[-1] != r.shape[-1]:
        raise ValueError("h_eos and r must have the same hidden size")
    # Compute in float32 for stability when the model runs in fp16.
    h = h_eos.float().unsqueeze(-2)
    return F.cosine_similarity(h, r.float(), dim=-1, eps=eps)


def rank_order(scores: torch.Tensor) -> torch.Tensor:
    """Indices that sort passages by descending score (ties keep input order)."""
    return torch.argsort(scores, dim=-1, descending=True, stable=True)
