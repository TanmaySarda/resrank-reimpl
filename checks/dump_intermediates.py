"""Save every component's output for a few samples to data/07_debug/<name>/ and print the norm report.

    python checks/dump_intermediates.py --train-file stage1 --n 2 --name first_look
The norm report tells you whether the injected passage vector or the reranker hidden state dominates
r = h + e; use it to choose model.inject_mode / model.inject_scale in configs/base.yaml.
"""
from _common import base_parser, setup, show

from resrank.model.diagnostics import dump_intermediates

ap = base_parser(__doc__)
ap.add_argument("--name", default="debug")
args = ap.parse_args()
cfg, paths, model, samples = setup(args)
meta = dump_intermediates(model, samples, paths.debug_dir(args.name))
show(meta["norms"])
print("saved to", paths.debug_dir(args.name))
