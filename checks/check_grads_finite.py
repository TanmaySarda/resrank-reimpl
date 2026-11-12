"""Why does fp16 training skip steps? Run one forward/backward under several settings and print one line
each: are the encoder's or reranker's gradients non-finite, and how large are the finite ones?

    python checks/check_grads_finite.py --train-file stage1 --n 1
Also prints the sizes (norms) of the injected passage vectors and the reranker hidden states.
"""
from _common import base_parser, setup

import torch

from resrank.model.diagnostics import forward_with_intermediates, grad_finite_report, norm_report

args = base_parser(__doc__).parse_args()
cfg, paths, model, samples = setup(args)

res = forward_with_intermediates(model, samples)
print("NORMS", {k: round(v, 3) for k, v in norm_report(res, res["passage_mask"]).items()})
del res
torch.cuda.empty_cache()

experiments = [  # (precision, loss scale, pair reduction, inject mode)
    ("fp32", 1.0, "sum", "raw"),
    ("fp32", 1.0, "mean", "raw"),
    ("fp16", 1.0, "sum", "raw"),
    ("fp16", 1024.0, "sum", "raw"),
    ("fp16", 1024.0, "mean", "raw"),
    ("fp16", 65536.0, "mean", "raw"),
    ("fp16", 1024.0, "mean", "unit"),
    ("fp16", 65536.0, "mean", "unit"),
]
keys = ["precision", "loss_scale", "pair_reduction", "inject_mode", "loss", "encoder_bad_params",
        "reranker_bad_params", "encoder_max_abs_grad", "reranker_max_abs_grad",
        "encoder_grad_norm", "reranker_grad_norm"]
print(" | ".join(keys))
for precision, scale, reduction, inject in experiments:
    model.cfg.pair_reduction, model.cfg.inject_mode = reduction, inject
    try:
        r = grad_finite_report(model, samples, precision, scale)
        print(" | ".join(str(r[k]) for k in keys))
    except torch.OutOfMemoryError:
        print(f"{precision} | {scale} | {reduction} | {inject} | OUT OF MEMORY")
        torch.cuda.empty_cache()
