"""Confirm both models share a hidden size, then run one forward pass and print every shape.

    python checks/check_shapes.py --train-file stage1
"""
from _common import base_parser, setup, show

from resrank.model.diagnostics import forward_with_intermediates

ap = base_parser(__doc__)
cfg, paths, model, samples = setup(ap.parse_args())
assert model.encoder.hidden_size == model.reranker.hidden_size, "hidden sizes differ"
res = forward_with_intermediates(model, samples)
show({
    "hidden_size": model.encoder.hidden_size,
    "layouts": [l.to_dict() for l in res["layouts"]],
    **{k: list(res[k].shape) for k in ("e", "inputs_embeds", "H", "h_p", "h_eos", "r", "scores")},
})
