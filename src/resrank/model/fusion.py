"""Residual fusion of encoder embeddings and reranker hidden states (paper Eq. 4)."""
from __future__ import annotations

import torch

FUSION_MODES = ("residual", "no_residual", "no_hidden")


def fuse(h_p: torch.Tensor, e: torch.Tensor, mode: str = "residual") -> torch.Tensor:
    """Combine reranker hidden states `h_p` with encoder embeddings `e`.

    Args:
        h_p: reranker hidden states at the passage positions, shape (..., n, d)
        e:   encoder embeddings of the same passages,           shape (..., n, d)
        mode:
            "residual"    r = h_p + e      (the paper's method, Eq. 4)
            "no_residual" r = h_p          (ablation "W/O Residual Conn.")
            "no_hidden"   r = e            (ablation "W/O Hidden State")

    Returns:
        fused passage representations r, shape (..., n, d)
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"unknown fusion mode '{mode}', expected one of {FUSION_MODES}")
    if h_p.shape != e.shape:
        raise ValueError(f"shape mismatch: h_p {tuple(h_p.shape)} vs e {tuple(e.shape)}")
    if mode == "no_residual":
        return h_p
    e = e.to(h_p.dtype)
    if mode == "no_hidden":
        return e
    return h_p + e
