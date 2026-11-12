"""Stage s02: build the training files.

IN : data/01_raw/pe_rank/train.jsonl, data/01_raw/e2rank/train.jsonl
OUT: data/02_train/<name>.jsonl   one RankingSample.to_dict() per line (0-indexed, passage text inline)
     data/02_train/<name>.stats.json   counts of what was read, filtered and kept

Stage 1 = PE-Rank (positive = top-ranked passage). Stage 2 = E2Rank filtered to English.

Decontamination (optional, off by default): --exclude-queries takes evaluation query files (BEIR queries.jsonl,
rows with "text") and drops every training sample whose query matches one of them after lowercasing and
keeping only letters/digits. Without it nothing is removed: the build does not know about the evaluation sets.
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Iterable, Optional

from resrank.data.e2rank import iter_e2rank_samples, reservoir_sample
from resrank.data.pe_rank import iter_pe_rank_samples
from resrank.data.sampling import shuffle_candidates, subsample_candidates
from resrank.io_utils import read_jsonl, write_json, write_jsonl
from resrank.paths import Paths
from resrank.schemas import RankingSample


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def load_eval_queries(paths: Iterable[str | Path]) -> set[str]:
    """Normalized query texts from evaluation query files (jsonl rows with a "text" field)."""
    return {normalize_query(str(r.get("text", ""))) for p in paths for r in read_jsonl(p)} - {""}


def _drop_eval_queries(samples: Iterable[RankingSample], exclude: Optional[set[str]],
                       stats: dict) -> Iterable[RankingSample]:
    if not exclude:
        yield from samples
        return
    stats.setdefault("dropped_eval_overlap", 0)
    for s in samples:
        if normalize_query(s.query) in exclude:
            stats["dropped_eval_overlap"] += 1
            continue
        yield s


def _finish(samples: Iterable[RankingSample], out_path: Path, n_samples: int, seed: int,
            n_candidates: Optional[int], shuffle: bool, stats: dict) -> dict:
    chosen = reservoir_sample(samples, n_samples, seed)
    rng = random.Random(seed)
    out = []
    for s in chosen:
        if n_candidates:
            s = subsample_candidates(s, n_candidates, rng)
        if shuffle:
            s = shuffle_candidates(s, rng)
        out.append(s)
    written = write_jsonl(out_path, (s.to_dict() for s in out))
    stats = dict(stats, written=written, seed=seed, n_candidates=n_candidates, shuffled=shuffle)
    write_json(Path(str(out_path).removesuffix(".jsonl") + ".stats.json"), stats)
    return stats


def build_stage1(pe_path, out_path, n_samples: int, seed: int = 42,
                 n_candidates: Optional[int] = None, shuffle: bool = False,
                 passage_source: str = "parse", exclude_queries: Optional[set[str]] = None) -> dict:
    stats: dict = {}
    gen = iter_pe_rank_samples(pe_path, passage_source=passage_source, stats=stats)
    gen = _drop_eval_queries(gen, exclude_queries, stats)
    return _finish(gen, Path(out_path), n_samples, seed, n_candidates, shuffle, stats)


def build_stage2(e2_path, out_path, n_samples: int, seed: int = 42,
                 n_candidates: Optional[int] = None, shuffle: bool = False,
                 allow_sources=None, deny_sources=None, max_cjk_ratio: float = 0.0,
                 exclude_queries: Optional[set[str]] = None) -> dict:
    stats: dict = {}
    gen = iter_e2rank_samples(e2_path, allow_sources=allow_sources, deny_sources=deny_sources,
                              english_only=True, max_cjk_ratio=max_cjk_ratio, stats=stats)
    gen = _drop_eval_queries(gen, exclude_queries, stats)
    return _finish(gen, Path(out_path), n_samples, seed, n_candidates, shuffle, stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["stage1", "stage2"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--name", default=None, help="output file name (default: the stage name)")
    ap.add_argument("--n", type=int, required=True, help="number of samples to keep")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-candidates", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--allow-sources", nargs="*")
    ap.add_argument("--deny-sources", nargs="*")
    ap.add_argument("--exclude-queries", nargs="*", default=None, metavar="QUERIES_JSONL",
                    help="evaluation query files; training samples with the same query are dropped (off by default)")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    paths = Paths.from_env(a.root).ensure()
    out = paths.train_file(a.name or a.stage)
    exclude = load_eval_queries(a.exclude_queries) if a.exclude_queries else None
    if a.stage == "stage1":
        stats = build_stage1(a.input, out, a.n, a.seed, a.n_candidates, a.shuffle, exclude_queries=exclude)
    else:
        stats = build_stage2(a.input, out, a.n, a.seed, a.n_candidates, a.shuffle,
                             a.allow_sources, a.deny_sources, exclude_queries=exclude)
    print(out, stats)


if __name__ == "__main__":
    main()
