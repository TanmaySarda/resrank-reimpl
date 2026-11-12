"""Torch helpers shared by training, diagnostics, loading and evaluation (one place for precision handling)."""
from __future__ import annotations

from typing import Optional

import torch

# Weight dtypes accepted by load_model (and --weights-dtype in checks/).
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def amp_dtype(precision: str) -> Optional[torch.dtype]:
    """Autocast dtype for a precision setting; None means no autocast (fp32)."""
    return {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)


def autocast(precision: str, on_cuda: bool):
    """Autocast context for the precision setting (a no-op off the GPU or for fp32)."""
    dtype = amp_dtype(precision)
    return torch.autocast("cuda", dtype=dtype, enabled=dtype is not None and on_cuda)


def make_grad_scaler(precision: str, on_cuda: bool, init_scale: float) -> Optional["torch.amp.GradScaler"]:
    """GradScaler for fp16 on a GPU, otherwise None."""
    if precision == "fp16" and on_cuda:
        return torch.amp.GradScaler("cuda", init_scale=init_scale)
    return None


def backward(loss: torch.Tensor, scaler) -> None:
    (scaler.scale(loss) if scaler else loss).backward()


def optimizer_step(opt, scaler, params, max_grad_norm: float) -> tuple[float, bool]:
    """Unscale (fp16), clip, step and update the scaler. Returns (grad_norm before clipping, skipped), where
    skipped is True when the fp16 scaler dropped the step because of inf/NaN gradients."""
    if scaler:
        scaler.unscale_(opt)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm))
    skipped = False
    if scaler:
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < before      # the scale drops exactly when the step was skipped
    else:
        opt.step()
    return grad_norm, skipped
