"""Readers for BEIR-format files.

ASSUMPTION (verify on the real download): corpus.jsonl rows are {"_id", "title", "text"}, queries.jsonl
rows are {"_id", "text"}, and qrels are TSV `query-id  corpus-id  score` with a header line.
"""
from __future__ import annotations

from pathlib import Path

from resrank.io_utils import read_jsonl


def doc_text(rec: dict) -> str:
    """Passage text as fed to rankers: title and body joined by a space."""
    title = (rec.get("title") or "").strip()
    text = (rec.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def read_beir_corpus(path: str | Path) -> dict[str, str]:
    return {str(r["_id"]): doc_text(r) for r in read_jsonl(path)}


def read_beir_queries(path: str | Path) -> dict[str, str]:
    return {str(r["_id"]): r["text"] for r in read_jsonl(path)}
