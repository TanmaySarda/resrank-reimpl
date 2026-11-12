"""Read/write helpers for the file formats used between stages.

- jsonl / json
- numpy arrays (.npy)
- TREC run files:   qid Q0 docid rank score tag
- TREC qrels files: qid 0 docid relevance
All writes are atomic (temp file + rename) so a crashed run never leaves a half-written file.
Reading supports .gz files; writing is plain text only (writing to a path ending in .gz raises).
"""
from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

Run = dict[str, list[tuple[str, float]]]      # qid -> [(docid, score)], best first
Qrels = dict[str, dict[str, int]]             # qid -> {docid: relevance}


def _open(path: Path, mode: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _atomic_write(path: str | Path, write_fn, binary: bool = False) -> None:
    path = Path(path)
    if str(path).endswith(".gz"):
        raise ValueError("Writing .gz files is not supported; write plain text and compress separately")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        if binary:
            with open(tmp, "wb") as f:
                write_fn(f)
        else:
            with open(tmp, "w", encoding="utf-8") as f:
                write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---- jsonl / json ----------------------------------------------------------
def read_jsonl(path: str | Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with _open(path, "r") as f:
        count = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    """Write rows as JSON lines. Returns the number of rows written."""
    n = 0

    def _write(f):
        nonlocal n
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    _atomic_write(path, _write)
    return n


def init_counts(stats: dict | None, keys) -> dict:
    """The caller's stats dict (or a new one) with every key in `keys` set to 0 if missing."""
    counts = stats if stats is not None else {}
    for key in keys:
        counts.setdefault(key, 0)
    return counts


def read_json(path: str | Path) -> Any:
    with _open(Path(path), "r") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    _atomic_write(path, lambda f: json.dump(obj, f, ensure_ascii=False, indent=indent))


# ---- numpy -----------------------------------------------------------------
def save_npy(path: str | Path, array: np.ndarray) -> None:
    _atomic_write(path, lambda f: np.save(f, array), binary=True)


def load_npy(path: str | Path, mmap: bool = False) -> np.ndarray:
    return np.load(path, mmap_mode="r" if mmap else None)


# ---- TREC run / qrels ------------------------------------------------------
def write_trec_run(path: str | Path, run: Run, tag: str = "resrank") -> None:
    """Write a run. Ranks are assigned by descending score (stable), starting at 1."""

    def _write(f):
        for qid, items in run.items():
            ordered = sorted(items, key=lambda x: -x[1])
            for rank, (docid, score) in enumerate(ordered, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")

    _atomic_write(path, _write)


def read_trec_run(path: str | Path) -> Run:
    """Read a run. Each query's list is ordered by the rank column."""
    tmp: dict[str, list[tuple[int, str, float]]] = {}
    with _open(Path(path), "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            qid, _, docid, rank, score = parts[:5]
            tmp.setdefault(qid, []).append((int(rank), docid, float(score)))
    return {
        qid: [(d, s) for _, d, s in sorted(items, key=lambda x: x[0])]
        for qid, items in tmp.items()
    }


def read_qrels(path: str | Path) -> Qrels:
    """Read qrels in either format (a header line is skipped):

    - BEIR TSV, 3 columns:  query-id  corpus-id  score
    - TREC,     4 columns:  qid  0  docid  relevance
    """
    qrels: Qrels = {}
    with _open(Path(path), "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                qid, docid, rel = parts
            elif len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
            else:
                continue
            try:
                rel_i = int(rel)
            except ValueError:
                continue  # header line
            qrels.setdefault(qid, {})[docid] = rel_i
    return qrels
