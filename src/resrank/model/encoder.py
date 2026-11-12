"""Encoder-LLM: compresses a passage (or query) into one vector.

Matches the Qwen3-Embedding recipe: left-padded batch, last-token pooling. Queries get the
'Instruct: ...\\nQuery:...' format; passages are encoded as plain text.

`encode` returns the *raw* pooled vector (not normalized). Callers normalize when they need cosine
similarity (InfoNCE does this itself); whether the vector injected into the reranker is raw or
normalized is decided in ResRankModel (config: inject_mode).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from resrank.model.pooling import last_token_pool
from resrank.model.prompts import DEFAULT_QUERY_TASK, format_query_for_encoder


def _load_with_dtype(auto_cls, name: str, dtype, attn_implementation: Optional[str]):
    """Load a HF model, passing the dtype under whichever argument name this transformers uses."""
    kwargs = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    if dtype is None:
        return auto_cls.from_pretrained(name, **kwargs)
    try:
        return auto_cls.from_pretrained(name, dtype=dtype, **kwargs)
    except TypeError:
        return auto_cls.from_pretrained(name, torch_dtype=dtype, **kwargs)


class PassageEncoder(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        max_length: int = 512,
        query_max_length: int = 128,
        query_task: str = DEFAULT_QUERY_TASK,
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.query_task = query_task

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        dtype=None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        **kwargs,
    ) -> "PassageEncoder":
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name_or_path, padding_side="left")
        model = _load_with_dtype(AutoModel, name_or_path, dtype, attn_implementation)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return cls(model, tokenizer, **kwargs)

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _tokenize(self, texts: list[str], max_length: int) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in dict(batch).items()}

    def encode(
        self,
        texts: list[str],
        *,
        is_query: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Encode texts into raw pooled vectors of shape (len(texts), d), keeping gradients."""
        if not texts:
            raise ValueError("no texts to encode")
        if is_query:
            texts = [format_query_for_encoder(self.query_task, t) for t in texts]
            max_length = self.query_max_length
        else:
            max_length = self.max_length
        chunk = chunk_size or len(texts)
        outputs = []
        for start in range(0, len(texts), chunk):
            batch = self._tokenize(texts[start:start + chunk], max_length)
            hidden = self.model(**batch).last_hidden_state
            outputs.append(last_token_pool(hidden, batch["attention_mask"]))
        return torch.cat(outputs, dim=0)
