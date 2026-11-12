"""Stage s06: encode a corpus with the trained encoder and cache the vectors.

IN : corpus (BEIR format), a checkpoint
OUT: data/04_index/<dataset>/embeddings.npy  (N, d) float32, rows aligned with ids.json
     data/04_index/<dataset>/ids.json        doc ids in row order
Only candidate documents are needed for reranking, so --only-ids can restrict this to the BM25 candidates.
"""
from __future__ import annotations

import argparse
from typing import Iterable, Optional

import numpy as np
import torch

from resrank.io_utils import read_trec_run, save_npy, write_json
from resrank.model.resrank import ResRankModel
from resrank.paths import Paths


@torch.no_grad()
def encode_texts(model: ResRankModel, texts: list[str], batch_size: int = 32) -> np.ndarray:
    model.eval()
    chunks = []
    for i in range(0, len(texts), batch_size):
        chunks.append(model.encoder.encode(texts[i:i + batch_size]).float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, model.encoder.hidden_size), "float32")


def build_index(model: ResRankModel, corpus: dict[str, str], doc_ids: Optional[Iterable[str]] = None,
                batch_size: int = 32) -> tuple[list[str], np.ndarray]:
    ids = [d for d in (doc_ids if doc_ids is not None else corpus) if d in corpus]
    # Sort by length so batches pad little.
    order = sorted(range(len(ids)), key=lambda i: len(corpus[ids[i]]))
    emb = encode_texts(model, [corpus[ids[i]] for i in order], batch_size)
    out = np.zeros_like(emb)
    out[order] = emb
    return ids, out


def main() -> None:
    from resrank.config import load_config
    from resrank.data.beir import read_beir_corpus
    from resrank.model.loading import load_eval_model

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint name under checkpoints/ (e.g. stage1), or 'pretrained' for the untrained "
                         "Qwen3 models (a baseline to see what training added)")
    ap.add_argument("--only-candidates", default=None, help="BM25 run file; encode only its documents")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    paths = Paths.from_env(a.root or cfg.get("root")).ensure()
    model = load_eval_model(cfg, paths, a.ckpt)
    only = None
    if a.only_candidates:
        only = sorted({d for items in read_trec_run(a.only_candidates).values() for d, _ in items})
    ids, emb = build_index(model, read_beir_corpus(a.corpus), only, a.batch_size)
    save_npy(paths.embeddings_file(a.dataset, a.ckpt), emb)
    write_json(paths.index_ids_file(a.dataset, a.ckpt), ids)
    print(f"encoded {len(ids)} documents -> {paths.index_dir(a.dataset, a.ckpt)}")


if __name__ == "__main__":
    main()
