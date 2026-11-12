"""A small pure-Python/numpy BM25, so candidate generation needs no Java or Pyserini.

Good for small BEIR corpora (SciFact, NFCorpus). It is slow on corpora with hundreds of thousands of
documents. Scores will not match Pyserini/Lucene exactly (different analyzer: no stemming here), so
the BM25 baseline will differ slightly from the paper's Table I.
Parameters default to Pyserini's (k1=0.9, b=0.4); idf follows Lucene's formula.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, k1: float = 0.9, b: float = 0.4) -> None:
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.doc_len = np.zeros(0)
        self.avg_len = 0.0
        self.n_docs = 0

    def fit(self, doc_ids: Sequence[str], texts: Sequence[str]) -> "BM25":
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.n_docs = len(doc_ids)
        lens = np.zeros(self.n_docs)
        index: dict[str, list[tuple[int, int]]] = {}
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            lens[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                index.setdefault(term, []).append((i, tf))
        self.doc_len = lens
        self.avg_len = float(lens.mean()) if self.n_docs else 0.0
        self.postings = {
            term: (np.array([d for d, _ in plist]), np.array([tf for _, tf in plist], dtype=float))
            for term, plist in index.items()
        }
        return self

    def idf(self, df: int) -> float:
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 100) -> list[tuple[str, float]]:
        scores = np.zeros(self.n_docs)
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tf = self.postings[term]
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[docs] / max(self.avg_len, 1e-9))
            scores[docs] += self.idf(len(docs)) * tf * (self.k1 + 1) / (tf + norm)
        k = min(k, self.n_docs)
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.doc_ids[i], float(scores[i])) for i in top if scores[i] > 0]
