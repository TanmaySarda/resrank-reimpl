"""Training losses (paper Eq. 6, 7, 8).

    L = lambda * InfoNCE + RankNet        (lambda = 0.1, tau1 = tau2 = 0.05)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def info_nce_loss(
    q: torch.Tensor,
    pos: torch.Tensor,
    neg: torch.Tensor,
    tau: float = 0.05,
    neg_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """InfoNCE over (1 positive + K negatives) per query, with cosine similarity (Eq. 6).

    Args:
        q:   query embeddings,                    (B, d)
        pos: positive document embeddings,        (B, d)
        neg: negative document embeddings,        (B, K, d)
        tau: temperature (0.05 in the paper)
        neg_mask: optional bool (B, K); False marks padded negatives to ignore.

    Returns:
        scalar loss, mean over the batch.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    q = F.normalize(q.float(), dim=-1)
    pos = F.normalize(pos.float(), dim=-1)
    neg = F.normalize(neg.float(), dim=-1)

    pos_logit = (q * pos).sum(-1, keepdim=True)            # (B, 1)
    neg_logit = torch.einsum("bd,bkd->bk", q, neg)         # (B, K)
    logits = torch.cat([pos_logit, neg_logit], dim=1) / tau

    if neg_mask is not None:
        pad = torch.zeros(q.shape[0], 1, dtype=torch.bool, device=logits.device)
        invalid = torch.cat([pad, ~neg_mask.to(logits.device)], dim=1)
        logits = logits.masked_fill(invalid, float("-inf"))

    target = torch.zeros(q.shape[0], dtype=torch.long, device=logits.device)  # positive is column 0
    return F.cross_entropy(logits, target)


def ranknet_loss(
    scores: torch.Tensor,
    ranks: torch.Tensor,
    tau: float = 0.05,
    mask: torch.Tensor | None = None,
    pair_reduction: str = "sum",
) -> torch.Tensor:
    """RankNet pairwise loss over a ranked list (Eq. 7).

    For every ordered pair (j, k) where passage j should rank above k (ranks[j] < ranks[k]):
        loss += log(1 + exp((S_k - S_j) / tau))
    Equal ranks create no pair, so graded labels with ties work.

    Args:
        scores: S(q, d_j) = cos(h_eos, r_j),                  (B, n)
        ranks:  rank of each passage, lower = more relevant,  (B, n)
        tau:    temperature (0.05 in the paper)
        mask:   optional bool (B, n); False marks padded passages to ignore.
        pair_reduction: "sum" (paper: sum over pairs) or "mean" (divide by number of pairs).

    Returns:
        scalar loss, mean over the batch.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    if pair_reduction not in ("sum", "mean"):
        raise ValueError("pair_reduction must be 'sum' or 'mean'")
    if scores.shape != ranks.shape:
        raise ValueError(f"scores {tuple(scores.shape)} and ranks {tuple(ranks.shape)} must match")

    scores = scores.float()
    # diff[b, j, k] = S_k - S_j
    diff = scores.unsqueeze(1) - scores.unsqueeze(2)
    # pair_mask[b, j, k] = True when j should be ranked above k
    pair_mask = ranks.unsqueeze(2) < ranks.unsqueeze(1)
    if mask is not None:
        mask = mask.to(pair_mask.device)
        pair_mask = pair_mask & mask.unsqueeze(2) & mask.unsqueeze(1)

    # softplus(x) = log(1 + exp(x)), computed stably.
    per_pair = F.softplus(diff / tau) * pair_mask
    per_sample = per_pair.sum(dim=(1, 2))
    if pair_reduction == "mean":
        per_sample = per_sample / pair_mask.sum(dim=(1, 2)).clamp(min=1)
    return per_sample.mean()


def total_loss(
    infonce: torch.Tensor, ranknet: torch.Tensor, lam: float = 0.1
) -> torch.Tensor:
    """Combined objective (Eq. 8): lambda * InfoNCE + RankNet."""
    return lam * infonce + ranknet
