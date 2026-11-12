"""Train on a few samples and watch the loss fall. It should drop well below its starting value.

    python checks/check_overfit.py --train-file stage1 --n 8 --steps 30 --lr 1e-5 --optimizer adamw8bit --precision fp16
Plain fp32 AdamW does not fit on a 16 GB GPU for the 0.6B pair; use adamw8bit (and --freeze-embeddings if needed).
"""
from _common import base_parser, setup, show

from resrank.model.diagnostics import train_steps

ap = base_parser(__doc__)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--optimizer", default="adamw8bit", choices=["adamw", "adamw8bit", "sgd"])
ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16", "bf16"])
args = ap.parse_args()
cfg, paths, model, samples = setup(args)
out = train_steps(model, samples, args.steps, args.lr, args.optimizer, args.precision)
h = out["history"]
show({"first": h[0], "last": h[-1], "peak_gpu_gb": out["peak_gpu_gb"], "history": [round(x, 3) for x in h]})
