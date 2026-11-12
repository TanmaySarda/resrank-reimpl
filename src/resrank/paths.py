"""Single source of truth for every directory and file location.

Runtime data lives under one root (default /kaggle/working, or $RESRANK_ROOT),
so moving to another machine only means changing the root.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ROOT = "/kaggle/working"


def checkpoint_parts(ckpt_dir: str | Path) -> tuple[Path, Path]:
    """(encoder dir, reranker dir) of a checkpoint folder written by ResRankModel.save_pretrained."""
    d = Path(ckpt_dir)
    return d / "encoder", d / "reranker"


@dataclass(frozen=True)
class Paths:
    root: Path

    @classmethod
    def from_env(cls, root: str | Path | None = None) -> "Paths":
        return cls(Path(root or os.environ.get("RESRANK_ROOT", DEFAULT_ROOT)))

    # ---- top-level folders -------------------------------------------------
    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "01_raw"

    @property
    def train(self) -> Path:
        return self.data / "02_train"

    @property
    def candidates(self) -> Path:
        return self.data / "03_candidates"

    @property
    def index(self) -> Path:
        return self.data / "04_index"

    @property
    def runs(self) -> Path:
        return self.data / "05_runs"

    @property
    def results(self) -> Path:
        return self.data / "06_results"

    @property
    def debug(self) -> Path:
        return self.data / "07_debug"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    # ---- specific files ----------------------------------------------------
    def train_file(self, name: str) -> Path:
        """e.g. train_file("stage2_e2rank_en") -> 02_train/stage2_e2rank_en.jsonl"""
        return self.train / f"{name}.jsonl"

    def candidate_run(self, dataset: str, source: str = "bm25") -> Path:
        """TREC run of first-stage candidates, e.g. 03_candidates/scifact.bm25.run"""
        return self.candidates / f"{dataset}.{source}.run"

    def index_dir(self, dataset: str, tag: str | None = None) -> Path:
        """04_index/<dataset>[/<tag>]. Use the checkpoint name as tag so two models never share an index."""
        base = self.index / dataset
        return base / tag if tag else base

    def embeddings_file(self, dataset: str, tag: str | None = None) -> Path:
        return self.index_dir(dataset, tag) / "embeddings.npy"

    def index_ids_file(self, dataset: str, tag: str | None = None) -> Path:
        return self.index_dir(dataset, tag) / "ids.json"

    def run_file(self, dataset: str, tag: str, order: str = "original") -> Path:
        """Reranked TREC run, e.g. 05_runs/scifact.run_a.original.run"""
        return self.runs / f"{dataset}.{tag}.{order}.run"

    def results_file(self, name: str = "ndcg") -> Path:
        return self.results / f"{name}.csv"

    def debug_dir(self, run_name: str) -> Path:
        return self.debug / run_name

    def stage_ckpt(self, name: str) -> Path:
        """e.g. stage_ckpt("stage1") -> checkpoints/stage1 (holds encoder/ and reranker/)."""
        return self.checkpoints / name

    def encoder_ckpt(self, name: str) -> Path:
        return checkpoint_parts(self.stage_ckpt(name))[0]

    def reranker_ckpt(self, name: str) -> Path:
        return checkpoint_parts(self.stage_ckpt(name))[1]

    # ---- helpers -----------------------------------------------------------
    def ensure(self) -> "Paths":
        """Create all top-level folders."""
        for p in (
            self.raw, self.train, self.candidates, self.index,
            self.runs, self.results, self.debug, self.checkpoints,
        ):
            p.mkdir(parents=True, exist_ok=True)
        return self
