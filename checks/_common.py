"""Shared helpers for the check scripts (run them from the repo root: python checks/<script>.py)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resrank.config import load_config  # noqa: E402
from resrank.model.loading import load_model  # noqa: E402
from resrank.paths import Paths  # noqa: E402
from resrank.schemas import RankingSample  # noqa: E402
from resrank.torch_utils import default_device  # noqa: E402
from resrank.train.trainer import load_samples  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--root", default=None)
    ap.add_argument("--train-file", default="stage1", help="name under data/02_train/ to take samples from")
    ap.add_argument("--n", type=int, default=2, help="number of samples")
    ap.add_argument("--longest-of", type=int, default=None, metavar="POOL",
                    help="instead of the first n samples, take the n samples with the most passage text out of the "
                         "first POOL samples (worst-case memory test)")
    ap.add_argument("--ckpt", default=None, help="checkpoint name under checkpoints/ (default: pretrained)")
    ap.add_argument("--weights-dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="memory fallback: do not train the token-embedding tables")
    ap.add_argument("--pair-reduction", choices=["sum", "mean"], default=None, help="override loss.ranknet_pair_reduction")
    ap.add_argument("--inject-mode", choices=["raw", "unit"], default=None, help="override model.inject_mode")
    ap.add_argument("--inject-scale", type=float, default=None, help="override model.inject_scale")
    return ap


def setup(args):
    cfg = load_config(args.config)
    paths = Paths.from_env(args.root or cfg.get("root")).ensure()
    if args.pair_reduction:
        cfg.setdefault("loss", {})["ranknet_pair_reduction"] = args.pair_reduction
    if args.inject_mode:
        cfg.setdefault("model", {})["inject_mode"] = args.inject_mode
    if args.inject_scale is not None:
        cfg.setdefault("model", {})["inject_scale"] = args.inject_scale
    if args.freeze_embeddings:
        cfg.setdefault("model", {})["freeze_embeddings"] = True
    model = load_model(cfg, paths.stage_ckpt(args.ckpt) if args.ckpt else None,
                       device=default_device(), weights_dtype=args.weights_dtype)
    if args.longest_of:
        pool = load_samples(paths.train_file(args.train_file), args.longest_of)
        pool.sort(key=lambda smp: sum(len(p) for p in smp.passages), reverse=True)
        samples = pool[:args.n]
        print(f"using the {len(samples)} longest samples of {len(pool)}; passage characters: "
              f"{[sum(len(p) for p in smp.passages) for smp in samples]}")
    else:
        samples = load_samples(paths.train_file(args.train_file), args.n)
    return cfg, paths, model, samples


def show(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))
