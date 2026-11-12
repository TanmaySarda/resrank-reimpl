"""ResRank: Encoder-LLM + Reranker-LLM wired together (paper Sec. III).

    e_i      = encoder(d_i)                                     (passage compression, Eq. 1)
    X        = [instruction ; query ; e_1..e_n ; query ; EOS]   (Eq. 2)
    H        = reranker(X)                                      (Eq. 3)
    r_i      = h_i + e_i                                        (Eq. 4, fusion)
    s_i      = cos(h_eos, r_i)                                  (Eq. 5)

Both models are trained jointly; gradients reach the encoder through the injected embeddings and
through the residual shortcut.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from resrank.io_utils import read_json, write_json
from resrank.model.encoder import PassageEncoder
from resrank.model.fusion import FUSION_MODES, fuse
from resrank.model.losses import info_nce_loss, ranknet_loss, total_loss
from resrank.model.prompts import DEFAULT_INSTRUCTION_TEMPLATE, format_instruction
from resrank.model.reranker import RerankerLM
from resrank.model.scoring import cosine_scores
from resrank.model.sequence import (build_inputs_embeds, collate_sequences, gather_hidden,
                                    length_mask, pad_stack)
from resrank.paths import checkpoint_parts
from resrank.schemas import RankingSample

INJECT_MODES = ("raw", "unit")


@dataclass
class ResRankConfig:
    fusion_mode: str = "residual"          # residual | no_residual | no_hidden
    inject_mode: str = "raw"               # raw: use pooled vector as is; unit: L2-normalize first
    inject_scale: float = 1.0              # multiplier on the injected vector (decide from norm check)
    instruction_template: str = DEFAULT_INSTRUCTION_TEMPLATE
    tau_infonce: float = 0.05
    tau_ranknet: float = 0.05
    lam: float = 0.1
    pair_reduction: str = "sum"
    passage_chunk_size: Optional[int] = None   # encode passages in chunks of this many (memory)

    def __post_init__(self) -> None:
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        if self.inject_mode not in INJECT_MODES:
            raise ValueError(f"inject_mode must be one of {INJECT_MODES}")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResRankConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


class ResRankModel(nn.Module):
    def __init__(self, encoder: PassageEncoder, reranker: RerankerLM,
                 cfg: Optional[ResRankConfig] = None) -> None:
        super().__init__()
        if encoder.hidden_size != reranker.hidden_size:
            raise ValueError(
                f"encoder hidden size {encoder.hidden_size} != reranker hidden size "
                f"{reranker.hidden_size}; the no-projector design needs them equal"
            )
        self.encoder = encoder
        self.reranker = reranker
        self.cfg = cfg or ResRankConfig()

    # ---- encoding ----------------------------------------------------------
    def encode_passages(self, passages: list[list[str]]) -> list[torch.Tensor]:
        """Encode each sample's passages. Returns raw pooled vectors, one (n_i, d) tensor per sample."""
        sizes = [len(p) for p in passages]
        if min(sizes, default=0) == 0:
            raise ValueError("every sample needs at least one passage")
        flat = [text for ps in passages for text in ps]
        e = self.encoder.encode(flat, chunk_size=self.cfg.passage_chunk_size)
        return list(torch.split(e, sizes, dim=0))

    def inject(self, e: torch.Tensor) -> torch.Tensor:
        """Turn a raw pooled passage vector into the vector placed in the reranker input."""
        if self.cfg.inject_mode == "unit":
            e = F.normalize(e.float(), dim=-1).to(e.dtype)
        e = e * self.cfg.inject_scale
        return e.to(device=self.reranker.device, dtype=self.reranker.dtype)

    # ---- reranking ---------------------------------------------------------
    def _sequence_for(self, query: str, e_inj: torch.Tensor):
        rr = self.reranker
        instruction = format_instruction(self.cfg.instruction_template, e_inj.shape[0])
        query_ids = rr.tokenize(query)
        if not query_ids:
            raise ValueError("query tokenizes to nothing")
        prefix = rr.token_embeddings(rr.tokenize(instruction) + query_ids)
        anchor = rr.token_embeddings(query_ids)
        eos = rr.token_embeddings([rr.eos_id])
        return build_inputs_embeds(prefix, e_inj, anchor, eos)

    def rerank_from_embeddings(
        self,
        queries: list[str],
        e_list: list[torch.Tensor],
        *,
        return_intermediates: bool = False,
    ) -> dict[str, Any]:
        """Score candidates given their precomputed passage vectors.

        Args:
            queries: one query string per sample
            e_list:  one raw (n_i, d) tensor per sample, in candidate order
        Returns:
            {"scores": (B, n_max), "passage_mask": (B, n_max) bool, ...}
            Scores in padded positions are meaningless; use `passage_mask`.
            With return_intermediates=True also: e, inputs_embeds, attention_mask, H, h_p, h_eos,
            r, layouts.
        """
        if len(queries) != len(e_list):
            raise ValueError("queries and e_list must have the same length")
        e_inj = [self.inject(e) for e in e_list]
        items = [self._sequence_for(q, e) for q, e in zip(queries, e_inj)]
        inputs_embeds, attention_mask, layouts = collate_sequences(items)
        hidden = self.reranker(inputs_embeds, attention_mask)
        h_p, passage_mask, h_eos = gather_hidden(hidden, layouts)
        e_pad = pad_stack(e_inj, n_max=h_p.shape[1])
        r = fuse(h_p, e_pad, self.cfg.fusion_mode)
        scores = cosine_scores(h_eos, r)
        out: dict[str, Any] = {"scores": scores, "passage_mask": passage_mask}
        if return_intermediates:
            out.update(e=e_pad, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                       H=hidden, h_p=h_p, h_eos=h_eos, r=r, layouts=layouts)
        return out

    def rerank(self, queries: list[str], passages: list[list[str]], **kw) -> dict[str, Any]:
        """Encode passages then rerank. For inference with a precomputed index use
        `rerank_from_embeddings`."""
        return self.rerank_from_embeddings(queries, self.encode_passages(passages), **kw)

    # ---- training ----------------------------------------------------------
    def training_losses(self, samples: list[RankingSample]) -> dict[str, torch.Tensor]:
        """L = lambda * InfoNCE + RankNet (Eq. 8) on a batch of RankingSample."""
        queries = [s.query for s in samples]
        e_list = self.encode_passages([s.passages for s in samples])
        out = self.rerank_from_embeddings(queries, e_list)
        scores, pmask = out["scores"], out["passage_mask"]

        n_max = scores.shape[1]
        ranks = torch.full((len(samples), n_max), n_max + 1, dtype=torch.long, device=scores.device)
        for i, s in enumerate(samples):
            ranks[i, : s.n] = torch.tensor(s.ranks, device=scores.device)
        l_rank = ranknet_loss(scores, ranks, self.cfg.tau_ranknet, mask=pmask,
                              pair_reduction=self.cfg.pair_reduction)

        # InfoNCE on the encoder's own query/document embeddings (keeps it a usable retriever).
        q_emb = self.encoder.encode(queries, is_query=True)
        pos = torch.stack([e_list[i][s.pos_index] for i, s in enumerate(samples)])
        negs = [e_list[i][s.negative_indices] for i, s in enumerate(samples)]
        neg = pad_stack(negs)
        neg_mask = length_mask([n.shape[0] for n in negs], neg.shape[1], neg.device)
        l_nce = info_nce_loss(q_emb, pos, neg, self.cfg.tau_infonce, neg_mask)

        return {"total": total_loss(l_nce, l_rank, self.cfg.lam), "infonce": l_nce, "ranknet": l_rank}

    # ---- save / load -------------------------------------------------------
    def save_pretrained(self, out_dir: str | Path) -> None:
        out = Path(out_dir)
        for name, part in (("encoder", self.encoder), ("reranker", self.reranker)):
            part.model.save_pretrained(out / name)
            part.tokenizer.save_pretrained(out / name)
        write_json(out / "resrank_config.json", asdict(self.cfg))

    @classmethod
    def from_pretrained(
        cls,
        encoder_path: str,
        reranker_path: str,
        cfg: Optional[ResRankConfig] = None,
        *,
        dtype=None,
        attn_implementation: Optional[str] = None,
        gradient_checkpointing: bool = False,
        encoder_kwargs: Optional[dict] = None,
    ) -> "ResRankModel":
        common = dict(dtype=dtype, attn_implementation=attn_implementation,
                      gradient_checkpointing=gradient_checkpointing)
        encoder = PassageEncoder.from_pretrained(encoder_path, **common, **(encoder_kwargs or {}))
        reranker = RerankerLM.from_pretrained(reranker_path, **common)
        return cls(encoder, reranker, cfg)

    @classmethod
    def load_checkpoint(cls, ckpt_dir: str | Path, **kw) -> "ResRankModel":
        """Load a directory written by `save_pretrained`."""
        cfg_path = Path(ckpt_dir) / "resrank_config.json"
        cfg = ResRankConfig.from_dict(read_json(cfg_path)) if cfg_path.exists() else None
        enc, rr = checkpoint_parts(ckpt_dir)
        return cls.from_pretrained(str(enc), str(rr), cfg, **kw)
