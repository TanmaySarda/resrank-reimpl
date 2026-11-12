"""Stage s07: rerank first-stage candidates with ResRank.

IN : data/03_candidates/<dataset>.bm25.run, data/04_index/<dataset>/, queries, a checkpoint
OUT: data/05_runs/<dataset>.<tag>.<order>.run   (TREC run with cosine scores)
Orders: original (first-stage order), reversed, random (see eval/ordering.py).
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from resrank.eval.ordering import ORDER_MODES, apply_order
from resrank.io_utils import Run, load_npy, read_json, read_trec_run, write_trec_run
from resrank.model.resrank import ResRankModel
from resrank.paths import Paths


@torch.no_grad()
def rerank_run(model: ResRankModel, queries: dict[str, str], candidates: Run, doc_ids: list[str],
               embeddings: np.ndarray, order: str = "original", top_k: int = 100,
               batch_size: int = 1, seed: int = 0) -> Run:
    """Rerank each query's top_k candidates using precomputed passage embeddings."""
    model.eval()
    row_of = {d: i for i, d in enumerate(doc_ids)}
    emb = torch.from_numpy(np.asarray(embeddings))
    qids = [q for q in candidates if q in queries]
    out: Run = {}
    for start in range(0, len(qids), batch_size):
        batch_q = qids[start:start + batch_size]
        cand_lists, e_list = [], []
        for qid in batch_q:
            docs = apply_order([d for d, _ in candidates[qid][:top_k]], order, qid, seed)
            missing = [d for d in docs if d not in row_of]
            if missing:
                raise KeyError(f"{len(missing)} candidate docs of query {qid} are not in the index, e.g. {missing[0]}")
            cand_lists.append(docs)
            e_list.append(emb[[row_of[d] for d in docs]].to(model.reranker.device))
        res = model.rerank_from_embeddings([queries[q] for q in batch_q], e_list)
        for i, qid in enumerate(batch_q):
            scores = res["scores"][i, : len(cand_lists[i])].float().cpu().tolist()
            out[qid] = list(zip(cand_lists[i], scores))
    return out


def main() -> None:
    from resrank.config import load_config
    from resrank.data.beir import read_beir_queries
    from resrank.model.loading import load_eval_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint name, or 'pretrained' (must match what s06 encoded)")
    ap.add_argument("--tag", default=None, help="run label (default: the checkpoint name)")
    ap.add_argument("--orders", nargs="*", default=["original"], choices=ORDER_MODES)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths.from_env(a.root or cfg.get("root")).ensure()
    model = load_eval_model(cfg, paths, a.ckpt)
    queries = read_beir_queries(a.queries)
    cands = read_trec_run(paths.candidate_run(a.dataset, "bm25"))
    ids = read_json(paths.index_ids_file(a.dataset, a.ckpt))
    emb = load_npy(paths.embeddings_file(a.dataset, a.ckpt))
    for order in a.orders:
        run = rerank_run(model, queries, cands, ids, emb, order, a.top_k, a.batch_size, cfg.get("seed", 0))
        out = paths.run_file(a.dataset, a.tag or a.ckpt, order)
        write_trec_run(out, run, tag=a.tag or a.ckpt)
        print("wrote", out)


if __name__ == "__main__":
    main()
