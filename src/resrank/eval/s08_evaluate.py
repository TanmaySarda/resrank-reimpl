"""Stage s08: score runs against qrels and write a results table.

IN : run files (data/03_candidates/*.run and data/05_runs/*.run), qrels
OUT: data/06_results/<name>.csv  with columns dataset, label, metric, k, value, n_queries
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from resrank.eval.metrics import evaluate_run
from resrank.io_utils import read_qrels, read_trec_run
from resrank.paths import Paths


def evaluate_files(dataset: str, qrels_path: str | Path, runs: dict[str, str | Path],
                   metrics=(("ndcg", 10),)) -> list[dict]:
    """Evaluate each labeled run file; returns table rows."""
    qrels = read_qrels(qrels_path)
    rows = []
    for label, path in runs.items():
        run = read_trec_run(path)
        for metric, k in metrics:
            res = evaluate_run(run, qrels, metric, k)
            rows.append({"dataset": dataset, "label": label, "metric": metric, "k": k,
                         "value": round(res["mean"], 4), "n_queries": res["n_queries"]})
    return rows


def write_results(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "label", "metric", "k", "value", "n_queries"])
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--name", default="ndcg")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    paths = Paths.from_env(a.root).ensure()
    runs = dict(r.split("=", 1) for r in a.run)
    rows = evaluate_files(a.dataset, a.qrels, runs)
    write_results(paths.results_file(a.name), rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
