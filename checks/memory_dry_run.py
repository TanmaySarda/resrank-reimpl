"""Run a few real optimizer steps and report peak GPU memory, using the same optimizer and autocast
settings as training. Optimizer state is included, so this is the number that says whether a config fits.

    python checks/memory_dry_run.py --train-file stage1 --n 2 --longest-of 500 --optimizer adamw8bit --precision fp16
Use --longest-of: the first few samples are usually short, and a long-passage batch is what runs out of memory.
If it runs out of memory, retry with --freeze-embeddings, then fewer candidates (rebuild with --n-candidates).
Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.
"""
from _common import base_parser, setup, show

from resrank.model.diagnostics import train_steps

ap = base_parser(__doc__)
ap.add_argument("--steps", type=int, default=8,
                help="use >= 6 with fp16: the loss scaler skips the first few steps, and skipped steps allocate no optimizer state")
ap.add_argument("--optimizer", default="adamw8bit", choices=["adamw", "adamw8bit", "sgd"])
ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
ap.add_argument("--init-loss-scale", type=float, default=65536.0)
args = ap.parse_args()
cfg, paths, model, samples = setup(args)
out = train_steps(model, samples, args.steps, lr=1e-6, optimizer=args.optimizer, precision=args.precision,
                  init_loss_scale=args.init_loss_scale)
if out["skipped_steps"] >= len(out["history"]):
    print("WARNING: every step was skipped by the loss scaler; peak memory excludes optimizer state. Raise --steps.")
show({**out, "n_samples": len(samples), "candidates_per_sample": [s.n for s in samples],
      "optimizer": args.optimizer, "precision": args.precision, "freeze_embeddings": args.freeze_embeddings,
      "pair_reduction": cfg.get("loss", {}).get("ranknet_pair_reduction"), "inject_mode": cfg.get("model", {}).get("inject_mode"),
      "init_loss_scale": args.init_loss_scale})
