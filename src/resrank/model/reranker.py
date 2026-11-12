"""Reranker-LLM: a causal LM backbone (no LM head) that reads token embeddings and injected passage
embeddings, and returns hidden states (paper Eq. 3)."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from resrank.model.encoder import _load_with_dtype
from resrank.model.prompts import DEFAULT_EOS_TOKEN


class RerankerLM(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, eos_token: str = DEFAULT_EOS_TOKEN) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.eos_token = eos_token
        self._eos_id: Optional[int] = None

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        dtype=None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        eos_token: str = DEFAULT_EOS_TOKEN,
    ) -> "RerankerLM":
        # AutoModel returns the backbone without the LM head, which is what we want.
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        model = _load_with_dtype(AutoModel, name_or_path, dtype, attn_implementation)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return cls(model, tokenizer, eos_token=eos_token)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    @property
    def eos_id(self) -> int:
        if self._eos_id is None:
            tid = self.tokenizer.convert_tokens_to_ids(self.eos_token)
            if tid is None or tid == getattr(self.tokenizer, "unk_token_id", None):
                raise ValueError(f"EOS token '{self.eos_token}' is not in the tokenizer vocabulary")
            self._eos_id = int(tid)
        return self._eos_id

    def tokenize(self, text: str) -> list[int]:
        """Token ids for a text piece, without special tokens added."""
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def token_embeddings(self, ids: list[int]) -> torch.Tensor:
        """(len(ids), d) embeddings from the backbone's input embedding table."""
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        return self.model.get_input_embeddings()(t)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run the backbone on an embedding sequence and return last hidden states (B, L, d)."""
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.last_hidden_state
