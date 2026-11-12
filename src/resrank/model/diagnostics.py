"""Debug helpers behind the scripts in checks/: norms, gradient flow, intermediate dumps, overfit test."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from resrank.io_utils import write_json
from resrank.model.resrank import ResRankModel
from resrank.schemas import RankingSample
from resrank.torch_utils import autocast, backward, make_grad_scaler, optimizer_step


def norm_report(inter: dict[str, Any], passage_mask: torch.Tensor) -> dict[str, float]:
    """Compare the size of the injected passage vector e_i and the reranker hidden state h_i.

    If the means differ a lot, one side dominates r = h + e. Decide on `inject_mode` / `inject_scale`
    from this.
    """
    m = passage_mask
    e_norm = inter["e"].float().norm(dim=-1)[m]
    h_norm = inter["h_p"].float().norm(dim=-1)[m]
    return {
        "e_norm_mean": e_norm.mean().item(),
        "h_norm_mean": h_norm.mean().item(),
        "h_over_e": (h_norm.mean() / e_norm.mean().clamp_min(1e-12)).item(),
        "h_eos_norm_mean": inter["h_eos"].float().norm(dim=-1).mean().item(),
    }


def grad_flow_report(model: ResRankModel, samples: list[RankingSample],
                     take_step: bool = True, lr: float = 1e-3) -> dict[str, Any]:
    """Run one loss + backward and report whether gradients reach both models.

    With take_step=True it also does one SGD step and checks that encoder weights actually changed
    (the 'embedding injection is not detached from the graph' check).
    """
    model.train()
    model.zero_grad(set_to_none=True)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    rr_params = [p for p in model.reranker.parameters() if p.requires_grad]
    before = enc_params[0].detach().clone() if enc_params else None

    losses = model.training_losses(samples)
    losses["total"].backward()

    def stats(params):
        grads = [p.grad for p in params if p.grad is not None]
        return {
            "params_with_grad": len(grads),
            "params_total": len(params),
            "grad_norm": float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else 0.0,
        }

    report: dict[str, Any] = {
        "losses": {k: float(v.detach()) for k, v in losses.items()},
        "encoder": stats(enc_params),
        "reranker": stats(rr_params),
    }
    if take_step:
        opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr)
        opt.step()
        opt.zero_grad(set_to_none=True)
        report["encoder_weights_changed"] = bool(
            before is not None and not torch.equal(before, enc_params[0].detach())
        )
    return report


@torch.no_grad()
def forward_with_intermediates(model: ResRankModel, samples: list[RankingSample]) -> dict[str, Any]:
    """One inference forward pass (eval mode, no gradients) that also returns every intermediate tensor."""
    model.eval()
    return model.rerank_from_embeddings(
        [s.query for s in samples],
        model.encode_passages([s.passages for s in samples]),
        return_intermediates=True,
    )


@torch.no_grad()
def dump_intermediates(model: ResRankModel, samples: list[RankingSample],
                       out_dir: str | Path) -> dict[str, Any]:
    """Save every component's output for one small batch (see plan, section 6)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = forward_with_intermediates(model, samples)
    for name in ("e", "inputs_embeds", "attention_mask", "H", "h_p", "h_eos", "r", "scores",
                 "passage_mask"):
        torch.save(res[name].detach().cpu(), out / f"{name}.pt")
    meta = {
        "layouts": [layout.to_dict() for layout in res["layouts"]],
        "qids": [s.qid for s in samples],
        "dtype": str(res["H"].dtype),
        "shapes": {k: list(res[k].shape) for k in ("e", "inputs_embeds", "H", "h_p", "r", "scores")},
        "norms": norm_report(res, res["passage_mask"]),
    }
    write_json(out / "meta.json", meta)
    return meta


def train_steps(model: ResRankModel, samples: list[RankingSample], steps: int = 2, lr: float = 1e-5,
                optimizer: str = "adamw", precision: str = "fp32",
                max_grad_norm: float = 1.0, init_loss_scale: float = 65536.0) -> dict[str, Any]:
    """Run `steps` optimizer steps on the same samples, mirroring the real trainer's optimizer and
    autocast choices. Returns the loss history and, on a GPU, the peak memory in GB (which includes
    optimizer state, so it is the number that says whether a config fits)."""
    from resrank.train.trainer import TrainConfig, make_optimizer

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, TrainConfig(lr=lr, optimizer=optimizer))
    on_cuda = model.reranker.device.type == "cuda"
    scaler = make_grad_scaler(precision, on_cuda, init_loss_scale)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    history, times, skipped = [], [], 0
    for _ in range(steps):
        if on_cuda:
            torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with autocast(precision, on_cuda):
            loss = model.training_losses(samples)["total"]
        backward(loss, scaler)
        _, was_skipped = optimizer_step(opt, scaler, params, max_grad_norm)
        skipped += int(was_skipped)
        if on_cuda:
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        history.append(float(loss.detach()))
    steady = times[1:] if len(times) > 1 else times      # the first step includes warm-up
    return {
        "history": history,
        "skipped_steps": skipped,
        "sec_per_sample": round(sum(steady) / len(steady) / max(1, len(samples)), 3),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if on_cuda else None,
        "trainable_params_m": round(sum(p.numel() for p in params) / 1e6, 1),
    }


def overfit(model: ResRankModel, samples: list[RankingSample], steps: int = 50, lr: float = 1e-4,
            optimizer: str = "adamw", precision: str = "fp32") -> list[float]:
    """Train on a handful of samples and return the loss per step. It should go close to zero."""
    return train_steps(model, samples, steps, lr, optimizer, precision)["history"]


def grad_finite_report(model: ResRankModel, samples: list[RankingSample], precision: str = "fp32",
                       loss_scale: float = 1.0) -> dict[str, Any]:
    """One forward/backward, then check every parameter's gradient for inf/NaN, separately for the
    encoder and the reranker. Gradient sizes are divided by `loss_scale` so runs are comparable.
    This separates "the loss scale is too high for fp16" from "gradients are NaN regardless"."""
    model.train()
    model.zero_grad(set_to_none=True)
    on_cuda = model.reranker.device.type == "cuda"
    with autocast(precision, on_cuda):
        loss = model.training_losses(samples)["total"]
    (loss * loss_scale).backward()
    out: dict[str, Any] = {
        "precision": precision, "loss_scale": loss_scale,
        "pair_reduction": model.cfg.pair_reduction, "inject_mode": model.cfg.inject_mode,
        "loss": round(float(loss.detach()), 4),
    }
    first_bad: list[str] = []
    for part in ("encoder", "reranker"):
        n_bad, max_abs, sq = 0, 0.0, 0.0
        for name, p in getattr(model, part).named_parameters():
            if p.grad is None:
                continue
            g = p.grad.float()
            if bool(torch.isfinite(g).all()):
                max_abs = max(max_abs, float(g.abs().max()) / loss_scale)
                sq += float((g ** 2).sum()) / loss_scale ** 2
            else:
                n_bad += 1
                if len(first_bad) < 3:
                    first_bad.append(f"{part}.{name}")
        out[f"{part}_bad_params"] = n_bad
        out[f"{part}_max_abs_grad"] = round(max_abs, 4)
        out[f"{part}_grad_norm"] = round(sq ** 0.5, 4)
    out["first_bad"] = first_bad
    model.zero_grad(set_to_none=True)
    return out
