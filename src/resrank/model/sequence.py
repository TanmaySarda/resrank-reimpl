"""Builds the reranker input sequence and reads hidden states back out (paper Eq. 2-3).

    X = [ prefix ; e_1 ... e_n ; query_anchor ; EOS ]
    prefix = instruction tokens + query tokens (already embedded by the reranker's token embeddings)

This module works purely on tensors, so it does not depend on any tokenizer or model.
Layouts record where the passage block and the EOS position sit, which is what the
reranker's hidden states are read from.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SequenceLayout:
    prefix_len: int      # instruction + query tokens
    n_passages: int      # one position per passage embedding
    anchor_len: int      # query tokens repeated after the passages

    @property
    def passage_start(self) -> int:
        return self.prefix_len

    @property
    def passage_end(self) -> int:
        return self.prefix_len + self.n_passages

    @property
    def anchor_start(self) -> int:
        return self.passage_end

    @property
    def eos_index(self) -> int:
        return self.passage_end + self.anchor_len

    @property
    def seq_len(self) -> int:
        return self.eos_index + 1

    @property
    def passage_slice(self) -> slice:
        return slice(self.passage_start, self.passage_end)

    def to_dict(self) -> dict:
        return {
            "prefix_len": self.prefix_len,
            "n_passages": self.n_passages,
            "anchor_len": self.anchor_len,
            "passage_start": self.passage_start,
            "passage_end": self.passage_end,
            "eos_index": self.eos_index,
            "seq_len": self.seq_len,
        }


def build_inputs_embeds(
    prefix_emb: torch.Tensor,
    passage_emb: torch.Tensor,
    anchor_emb: torch.Tensor,
    eos_emb: torch.Tensor,
) -> tuple[torch.Tensor, SequenceLayout]:
    """Concatenate one sample's pieces into a single (seq_len, d) embedding sequence.

    Args:
        prefix_emb:  (P, d) token embeddings of instruction + query
        passage_emb: (n, d) encoder embeddings e_1..e_n, in candidate order
        anchor_emb:  (A, d) token embeddings of the query repeated after the passages
        eos_emb:     (d,) or (1, d) token embedding of the end-of-sequence token
    """
    if eos_emb.dim() == 1:
        eos_emb = eos_emb.unsqueeze(0)
    for name, t in (("prefix_emb", prefix_emb), ("passage_emb", passage_emb),
                    ("anchor_emb", anchor_emb), ("eos_emb", eos_emb)):
        if t.dim() != 2:
            raise ValueError(f"{name} must be 2-D (length, d), got shape {tuple(t.shape)}")
    d = prefix_emb.shape[1]
    for name, t in (("passage_emb", passage_emb), ("anchor_emb", anchor_emb), ("eos_emb", eos_emb)):
        if t.shape[1] != d:
            raise ValueError(f"{name} hidden size {t.shape[1]} does not match prefix hidden size {d}")
    if eos_emb.shape[0] != 1:
        raise ValueError("eos_emb must be a single position")

    dtype = prefix_emb.dtype
    x = torch.cat(
        [prefix_emb, passage_emb.to(dtype), anchor_emb.to(dtype), eos_emb.to(dtype)], dim=0
    )
    layout = SequenceLayout(
        prefix_len=prefix_emb.shape[0],
        n_passages=passage_emb.shape[0],
        anchor_len=anchor_emb.shape[0],
    )
    assert x.shape[0] == layout.seq_len
    return x, layout


def collate_sequences(
    items: list[tuple[torch.Tensor, SequenceLayout]],
) -> tuple[torch.Tensor, torch.Tensor, list[SequenceLayout]]:
    """Right-pad a list of sequences into a batch.

    Returns:
        inputs_embeds:  (B, L_max, d), zeros in padded positions
        attention_mask: (B, L_max), 1 for real positions and 0 for padding
        layouts:        one SequenceLayout per sample
    """
    if not items:
        raise ValueError("no sequences to collate")
    batch = pad_stack([x for x, _ in items])
    mask = length_mask([x.shape[0] for x, _ in items], batch.shape[1], batch.device).long()
    return batch, mask, [layout for _, layout in items]


def gather_hidden(
    hidden: torch.Tensor, layouts: list[SequenceLayout]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the hidden states we need out of the reranker output.

    Args:
        hidden:  (B, L_max, d) last hidden states
        layouts: one layout per sample (same order as the batch)

    Returns:
        h_p:           (B, n_max, d) hidden states at passage positions, zeros where padded
        passage_mask:  (B, n_max) bool, True for real passages
        h_eos:         (B, d) hidden state at the EOS position
    """
    if hidden.shape[0] != len(layouts):
        raise ValueError("batch size and number of layouts differ")
    h_p = pad_stack([hidden[i, layout.passage_slice] for i, layout in enumerate(layouts)])
    passage_mask = length_mask([layout.n_passages for layout in layouts], h_p.shape[1], hidden.device)
    h_eos = torch.stack([hidden[i, layout.eos_index] for i, layout in enumerate(layouts)])
    return h_p, passage_mask, h_eos


def length_mask(lengths: list[int], n_max: int, device=None) -> torch.Tensor:
    """(B, n_max) bool mask, True for the first lengths[i] positions of row i (the real, unpadded ones)."""
    return torch.arange(n_max, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)


def pad_stack(tensors: list[torch.Tensor], n_max: int | None = None) -> torch.Tensor:
    """Stack (n_i, d) tensors into (B, n_max, d), padding with zeros."""
    if not tensors:
        raise ValueError("nothing to stack")
    n_max = n_max if n_max is not None else max(t.shape[0] for t in tensors)
    d = tensors[0].shape[1]
    out = torch.zeros(len(tensors), n_max, d, dtype=tensors[0].dtype, device=tensors[0].device)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0]] = t
    return out
