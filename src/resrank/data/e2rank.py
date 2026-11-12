"""Loader for the E2Rank ranking dataset (Alibaba-NLP/E2Rank_ranking_datasets, train.jsonl).

Row schema (confirmed from a sample):
    query: str
    document: list[str]       candidate passages (16 in the sample)
    ranking: list[int]        1-indexed permutation, best first
    pos_index: int            1-indexed position of the positive in `document`
    model_output: str         the ranking as text, e.g. "[14] > [1] > ..."
    source: str               origin dataset, e.g. "cMedQAv2"

Samples are converted to 0-indexed RankingSample objects.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, Optional

from resrank.data.filters import cjk_ratio, english_word_rate, is_english_sample
from resrank.io_utils import init_counts, read_jsonl
from resrank.schemas import RankingSample, SchemaError


def iter_e2rank_samples(
    path: str | Path,
    *,
    allow_sources: Optional[Iterable[str]] = None,
    deny_sources: Optional[Iterable[str]] = None,
    english_only: bool = True,
    max_cjk_ratio: float = 0.0,
    limit: Optional[int] = None,
    stats: Optional[dict] = None,
) -> Iterator[RankingSample]:
    """Stream valid, filtered samples.

    Args:
        allow_sources: if given, keep only rows whose `source` is in this set
        deny_sources:  drop rows whose `source` is in this set
        english_only:  drop rows with CJK text in the query or any passage
        max_cjk_ratio: tolerated CJK ratio per text when english_only is on
        limit:         stop after yielding this many samples
        stats:         optional dict that is filled with counts
                       (read, invalid, dropped_source, dropped_language, kept)
    """
    allow = set(allow_sources) if allow_sources is not None else None
    deny = set(deny_sources) if deny_sources is not None else set()
    counts = init_counts(stats, ("read", "invalid", "dropped_source", "dropped_language", "kept"))

    for i, row in enumerate(read_jsonl(path)):
        counts["read"] += 1
        try:
            sample = RankingSample.from_e2rank_row(row, qid=f"e2r-{i}")
        except SchemaError:
            counts["invalid"] += 1
            continue
        if (allow is not None and sample.source not in allow) or sample.source in deny:
            counts["dropped_source"] += 1
            continue
        if english_only and not is_english_sample(sample, max_cjk_ratio):
            counts["dropped_language"] += 1
            continue
        counts["kept"] += 1
        yield sample
        if limit is not None and counts["kept"] >= limit:
            return


def source_report(path: str | Path, limit: Optional[int] = None, passages_per_row: int = 3,
                  text_rows_per_source: Optional[int] = 1000) -> dict[str, dict]:
    """Summarize each `source` value, to decide which sources are English.

    Returns {source: {"count", "mean_query_cjk_ratio", "frac_samples_with_cjk",
                      "frac_pass_script_filter", "english_word_rate"}}.
    frac_pass_script_filter: share of rows that pass is_english_sample (CJK and other non-Latin scripts).
    english_word_rate: mean share of common English words in the query and the first `passages_per_row`
    passages; English sources score roughly 0.25-0.4, Latin-script non-English sources close to 0.
    Every row is counted, but the text statistics use only the first `text_rows_per_source` rows of each
    source (None = all), which keeps a pass over the full 1.8 GB file to a few minutes.
    """
    count: Counter = Counter()
    cjk_sum: dict[str, float] = defaultdict(float)
    cjk_samples: Counter = Counter()
    passed: Counter = Counter()
    word_rate: dict[str, float] = defaultdict(float)
    for row in read_jsonl(path, limit=limit):
        source = row.get("source", "<missing>")
        count[source] += 1
        if text_rows_per_source is not None and count[source] > text_rows_per_source:
            continue
        query = str(row.get("query", ""))
        docs = [str(p) for p in row.get("document", [])]
        cjk_sum[source] += cjk_ratio(query)
        if any(cjk_ratio(t) > 0 for t in [query] + docs):
            cjk_samples[source] += 1
        texts = [query] + docs[:passages_per_row]
        word_rate[source] += sum(english_word_rate(t) for t in texts) / len(texts)
        try:
            passed[source] += int(is_english_sample(RankingSample.from_e2rank_row(row)))
        except SchemaError:
            pass
    def seen(s: str) -> int:        # rows that went into the text statistics
        return count[s] if text_rows_per_source is None else min(count[s], text_rows_per_source)

    return {
        s: {
            "count": n,
            "mean_query_cjk_ratio": cjk_sum[s] / seen(s),
            "frac_samples_with_cjk": cjk_samples[s] / seen(s),
            "frac_pass_script_filter": passed[s] / seen(s),
            "english_word_rate": word_rate[s] / seen(s),
        }
        for s, n in count.most_common()
    }


def suggest_english_sources(report: dict[str, dict], min_word_rate: float = 0.15,
                            min_pass: float = 0.9) -> list[str]:
    """Sources that look English: most rows pass the script filter and the text is full of English
    function words. A suggestion to check by eye, not a guarantee (ASSUMED thresholds)."""
    return sorted(s for s, r in report.items()
                  if r["frac_pass_script_filter"] >= min_pass and r["english_word_rate"] >= min_word_rate)


def reservoir_sample(items: Iterable, k: int, seed: int = 0) -> list:
    """Uniform random sample of k items from a stream, using O(k) memory."""
    rng = random.Random(seed)
    reservoir: list = []
    for i, item in enumerate(items):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir


def e2rank_stats(path: str | Path, limit: Optional[int] = None) -> dict:
    """Format checks on raw E2Rank rows: list lengths, validity, how often the labeler's top passage
    is `pos_index`, and passage-length percentiles (characters), which inform the encoder max length."""
    import numpy as np

    n_rows = n_valid = n_top_is_pos = 0
    list_lengths: Counter = Counter()
    lengths: list[int] = []
    for row in read_jsonl(path, limit=limit):
        n_rows += 1
        list_lengths[len(row.get("document", []))] += 1
        try:
            s = RankingSample.from_e2rank_row(row)
        except SchemaError:
            continue
        n_valid += 1
        n_top_is_pos += int(s.ranking[0] == s.pos_index)
        lengths.extend(len(p) for p in s.passages)
    out: dict = {"rows": n_rows, "valid": n_valid, "list_length_counts": dict(list_lengths)}
    if n_valid:
        out["frac_top_ranked_is_pos_index"] = n_top_is_pos / n_valid
        pct = np.percentile(lengths, [50, 90, 99, 100])
        out["passage_chars_p50_p90_p99_max"] = [float(x) for x in pct]
    return out
