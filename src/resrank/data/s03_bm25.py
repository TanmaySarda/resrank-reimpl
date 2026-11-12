"""Stage s03: BM25 candidates for an evaluation dataset.

IN : BEIR-format corpus.jsonl and queries.jsonl (data/01_raw/<dataset>/)
OUT: data/03_candidates/<dataset>.bm25.run   (TREC run, top-k per query)
"""
from __future__ import annotations

import argparse

from resrank.data.beir import read_beir_corpus, read_beir_queries
from resrank.data.bm25 import BM25
from resrank.io_utils import Run, write_trec_run
from resrank.paths import Paths


def run_bm25(corpus: dict[str, str], queries: dict[str, str], k: int = 100) -> Run:
    ids = list(corpus)
    bm25 = BM25().fit(ids, [corpus[i] for i in ids])
    return {qid: bm25.search(q, k) for qid, q in queries.items()}


def restrict_queries(queries: dict[str, str], qrels: dict) -> dict[str, str]:
    """Keep only the queries that appear in the qrels (e.g. a dataset's test split)."""
    return {q: t for q, t in queries.items() if q in qrels}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", default=None, help="only run the queries that appear in this qrels file (e.g. test.tsv)")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    paths = Paths.from_env(a.root).ensure()
    queries = read_beir_queries(a.queries)
    if a.qrels:
        from resrank.io_utils import read_qrels
        queries = restrict_queries(queries, read_qrels(a.qrels))
    run = run_bm25(read_beir_corpus(a.corpus), queries, a.k)
    out = paths.candidate_run(a.dataset, "bm25")
    write_trec_run(out, run, tag="bm25")
    print(f"wrote {out} ({len(run)} queries)")


if __name__ == "__main__":
    main()
