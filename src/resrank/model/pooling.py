"""Last-token pooling, as used by Qwen3-Embedding (from its model card)."""
from __future__ import annotations

import torch


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Take each sequence's last real token's hidden state.

    Works for both left padding (last position is always real) and right padding.

    Args:
        last_hidden_states: (B, L, d)
        attention_mask:     (B, L), 1 for real tokens
    Returns:
        (B, d)
    """
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]
