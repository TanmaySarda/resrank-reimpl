"""Shared test fixtures: a fake tokenizer and tiny random Qwen3 models (no downloads needed)."""
import json
import random
from pathlib import Path

import pytest
import torch

from resrank.model.encoder import PassageEncoder
from resrank.model.reranker import RerankerLM
from resrank.model.resrank import ResRankConfig, ResRankModel
from resrank.schemas import RankingSample

PAD_ID, UNK_ID, EOS_ID = 0, 1, 2
VOCAB = 260


class FakeTokenizer:
    """Byte-level tokenizer with the small API surface the project uses."""

    unk_token_id = UNK_ID
    pad_token_id = PAD_ID

    def _ids(self, text):
        return [3 + (ord(c) % 250) for c in text]

    def __call__(self, text, add_special_tokens=True, padding=False, truncation=False,
                 max_length=None, return_tensors=None):
        if isinstance(text, str):
            return {"input_ids": self._ids(text)}
        seqs = [self._ids(t) for t in text]
        if truncation and max_length:
            seqs = [s[:max_length] for s in seqs]
        width = max(len(s) for s in seqs)
        ids = [[PAD_ID] * (width - len(s)) + s for s in seqs]        # left padding
        mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}

    def convert_tokens_to_ids(self, token):
        return EOS_ID if token == "<|endoftext|>" else UNK_ID

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "fake_tokenizer.json").write_text(json.dumps({"fake": True}))


def tiny_backbone(seed):
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(seed)
    cfg = Qwen3Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                      vocab_size=VOCAB, max_position_embeddings=1024)
    return Qwen3Model(cfg)


def make_model(**cfg_kwargs):
    tok = FakeTokenizer()
    enc = PassageEncoder(tiny_backbone(1), tok, max_length=64, query_max_length=32)
    rr = RerankerLM(tiny_backbone(2), tok)
    return ResRankModel(enc, rr, ResRankConfig(**cfg_kwargs))


def make_sample(n=5, seed=0, qid="q0"):
    rng = random.Random(seed)
    ranking = list(range(n))
    rng.shuffle(ranking)
    passages = [f"passage {i} about topic {rng.randint(0, 99)} and more words" for i in range(n)]
    return RankingSample(query=f"what is topic {seed}", passages=passages, ranking=ranking,
                         pos_index=ranking[0], source="test", qid=qid)


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def samples():
    return [make_sample(5, seed=i, qid=f"q{i}") for i in range(3)]
