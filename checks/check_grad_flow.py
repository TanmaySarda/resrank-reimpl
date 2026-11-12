"""One backward pass: do gradients reach the encoder and the reranker, and does one step change the
encoder's weights?   python checks/check_grad_flow.py --train-file stage1 --n 2"""
from _common import base_parser, setup, show

from resrank.model.diagnostics import grad_flow_report

cfg, paths, model, samples = setup(base_parser(__doc__).parse_args())
report = grad_flow_report(model, samples)
show(report)
assert report["encoder"]["grad_norm"] > 0, "encoder received no gradient"
assert report["encoder_weights_changed"], "encoder weights did not change after a step"
print("OK: gradients reach both models")
