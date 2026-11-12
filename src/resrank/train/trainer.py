"""Joint training loop for ResRank (both models trained together, full parameters).

Plain PyTorch: gradient accumulation, optional autocast (fp16 with GradScaler, or bf16),
optional 8-bit AdamW, linear warmup then linear decay, periodic saves, JSONL logging.
The paper does not specify the schedule, clipping or weight decay; those defaults are ASSUMED.
Resuming: every checkpoint folder also holds `train_state.json` (step, loss scale, seed, data settings).
`train(..., start_step=N)` skips the first N optimizer steps' worth of data (the per-epoch order is a pure
function of the seed and epoch) and continues the learning-rate schedule from step N. The optimizer's moment
estimates are NOT saved (8-bit Adam state is large), so they restart from zero; a short learning-rate ramp
after resuming softens that. Runs are not bit-for-bit repeatable on a GPU (non-deterministic kernels, fp16).
"""
from __future__ import annotations

import gc
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from resrank import hub
from resrank.io_utils import read_jsonl
from resrank.model.resrank import ResRankModel
from resrank.schemas import RankingSample
from resrank.torch_utils import autocast, backward, make_grad_scaler, optimizer_step
from resrank.util import say


@dataclass
class TrainConfig:
    epochs: int = 1
    micro_batch_size: int = 1
    grad_accum: int = 128
    lr: float = 6e-6
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    precision: str = "fp32"          # fp32 | fp16 | bf16
    optimizer: str = "adamw"         # adamw | adamw8bit | sgd
    log_every: int = 1
    save_every: int = 0              # optimizer steps; 0 = off
    save_every_minutes: float = 0    # wall-clock minutes between snapshots; 0 = off (either trigger saves)
    max_steps: Optional[int] = None
    seed: int = 42
    init_loss_scale: float = 65536.0   # fp16 GradScaler start value; lower it if early steps are skipped
    progress: bool = True              # show a progress bar (percentage, elapsed/remaining time, live loss)
    hub_repo: Optional[str] = None     # e.g. 'you/resrank-ckpts': push `latest` at each save and the final model (needs HF_TOKEN)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _push(cfg: "TrainConfig", folder: Path, path_in_repo: str, message: str, ignore=None) -> None:
    """Back up to the Hub if configured. A network failure is reported but never stops training."""
    if not cfg.hub_repo:
        return
    try:
        hub.push_folder(folder, cfg.hub_repo, path_in_repo, ignore_patterns=ignore, message=message)
        print(f"[hub] pushed {path_in_repo} ({message})", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[hub] push of {path_in_repo} failed, continuing: {e}", flush=True)


def load_samples(path: str | Path, limit: Optional[int] = None) -> list[RankingSample]:
    return [RankingSample.from_dict(r) for r in read_jsonl(path, limit=limit)]


def make_optimizer(params, cfg: TrainConfig):
    if cfg.optimizer == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr)
    raise ValueError("optimizer must be 'adamw', 'adamw8bit' or 'sgd'")


_make_optimizer = make_optimizer  # old name


def load_train_state(ckpt_dir: str | Path) -> dict:
    p = Path(ckpt_dir) / "train_state.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; this checkpoint was saved without training state")
    return json.loads(p.read_text())


def check_resume_state(state: dict, cfg: TrainConfig, n_samples: int) -> None:
    """Refuse to resume if the data or schedule settings differ from the interrupted run, since the
    skipped-data bookkeeping would then be wrong."""
    mismatches = {
        k: (state.get(k), v) for k, v in {
            "seed": cfg.seed, "micro_batch_size": cfg.micro_batch_size, "grad_accum": cfg.grad_accum,
            "epochs": cfg.epochs, "n_samples": n_samples, "lr": cfg.lr,
        }.items() if state.get(k) != v
    }
    if mismatches:
        raise ValueError(f"cannot resume: settings differ from the interrupted run (saved, now): {mismatches}")


def _model_bytes(model: ResRankModel) -> int:
    """Size of the weights as saved (shared tensors counted once)."""
    seen: set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel() * p.element_size()
    return total


def _free_bytes(path: Path) -> int:
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


DISK_MARGIN_BYTES = 1_000_000_000      # keep about 1 GB free for logs, uploads and temporary files


def _has_room_for(model: ResRankModel, path: Path) -> bool:
    """Is there room for one more checkpoint (plus a safety margin) on the disk holding `path`?"""
    return _free_bytes(path) >= int(_model_bytes(model) * 1.02) + DISK_MARGIN_BYTES


def _write_weights_and_state(model: ResRankModel, folder: Path, state: dict) -> None:
    """Save weights and train_state.json to `folder`; on any failure remove the half-written folder."""
    try:
        model.save_pretrained(folder)
        (folder / "train_state.json").write_text(json.dumps(state))
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)       # never leave a half-written copy eating the disk
        raise


def _save_snapshot(model: ResRankModel, folder: Path, state: dict) -> None:
    """Write weights + training state to `folder`.

    Normally the new snapshot is written to a temp folder and swapped in, so an interrupted save never destroys
    the previous one. That needs room for two copies. When the disk is too tight for that (Kaggle's working
    folder is about 19.5 GB and one checkpoint is about 4.8 GB), the old snapshot is deleted first and the new one
    is written in place: not atomic, but the copy already uploaded elsewhere is the fallback."""
    tmp = folder.with_name(folder.name + ".tmp")
    old = folder.with_name(folder.name + ".old")
    for d in (tmp, old):
        shutil.rmtree(d, ignore_errors=True)
    if _has_room_for(model, folder.parent):
        _write_weights_and_state(model, tmp, state)
        if folder.exists():
            folder.rename(old)
        tmp.rename(folder)
        shutil.rmtree(old, ignore_errors=True)
    else:
        say("[train] disk is tight: replacing the snapshot in place (not atomic)")
        shutil.rmtree(folder, ignore_errors=True)
        _write_weights_and_state(model, folder, state)


def train(model: ResRankModel, samples: list[RankingSample], cfg: TrainConfig,
          out_dir: str | Path,
          on_save: Optional[Callable[[Path, bool], None]] = None,
          start_step: int = 0, start_loss_scale: Optional[float] = None, save_final: bool = True,
          resume_ramp_steps: int = 10) -> list[dict]:
    """Train and save the final model to `out_dir`. Returns the logged records.

    `on_save(out_dir, final)` is called after every periodic save (final=False) and after the final save
    (final=True); use it to copy checkpoints somewhere durable (see train/publish.py).
    `start_step` > 0 resumes an interrupted run (see the module docstring); `start_loss_scale` restores the
    fp16 loss scale from the saved state."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = model.reranker.device
    use_cuda = device.type == "cuda"
    scaler = make_grad_scaler(cfg.precision, use_cuda, start_loss_scale or cfg.init_loss_scale)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(params, cfg)

    per_step = cfg.micro_batch_size * cfg.grad_accum
    steps_per_epoch = max(1, len(samples) // per_step)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_steps:
        total_steps = min(total_steps, cfg.max_steps)
    if start_step > total_steps:
        raise ValueError(f"start_step {start_step} is beyond the planned {total_steps} steps")
    warmup = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_at(step: int) -> float:
        if step < warmup:
            lr = cfg.lr * (step + 1) / warmup
        else:
            lr = cfg.lr * max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        if start_step and resume_ramp_steps and step >= start_step:   # optimizer moments restarted at zero
            lr *= min(1.0, (step - start_step + 1) / resume_ramp_steps)
        return lr

    def state_at(step: int) -> dict:
        return {"step": step, "total_steps": total_steps,
                "loss_scale": float(scaler.get_scale()) if scaler else None, "seed": cfg.seed,
                "micro_batch_size": cfg.micro_batch_size, "grad_accum": cfg.grad_accum,
                "epochs": cfg.epochs, "n_samples": len(samples), "lr": cfg.lr}

    log: list[dict] = []
    log_f = open(out_dir / "train_log.jsonl", "a" if start_step else "w", encoding="utf-8")
    model.train()
    step, chunk_no, start = start_step, 0, time.time()
    last_save = start
    oom_skipped = 0
    bar = None
    if cfg.progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total_steps, initial=start_step, desc="training", unit="step", dynamic_ncols=True,
                       smoothing=0.05, bar_format="{l_bar}{bar:25}| {n_fmt}/{total_fmt} "
                                                  "[{elapsed}<{remaining}, {rate_fmt}{postfix}]")
        except ImportError:
            bar = None
    try:
        for epoch in range(cfg.epochs):
            order = list(range(len(samples)))
            random.Random(f"{cfg.seed}-{epoch}").shuffle(order)   # order depends only on (seed, epoch)
            for s0 in range(0, len(order) - per_step + 1 if len(order) >= per_step else 1, per_step):
                chunk_no += 1
                if chunk_no <= start_step:
                    continue                                       # already trained on before the interruption
                if step >= total_steps:
                    break
                chunk = order[s0:s0 + per_step]
                sums = {"total": 0.0, "infonce": 0.0, "ranknet": 0.0}
                n_micro = 0
                oom = False
                # Number of micro-batches in this step. ceil, not floor: with a short last micro-batch, floor
                # would divide by one too few and the accumulated gradient would be too large.
                n_acc = math.ceil(len(chunk) / cfg.micro_batch_size)
                try:
                    for m0 in range(0, len(chunk), cfg.micro_batch_size):
                        batch = [samples[i] for i in chunk[m0:m0 + cfg.micro_batch_size]]
                        with autocast(cfg.precision, use_cuda):
                            losses = model.training_losses(batch)
                        scaled = losses["total"] / n_acc
                        backward(scaled, scaler)
                        for k in sums:
                            sums[k] += float(losses[k].detach())
                        n_micro += 1
                except torch.OutOfMemoryError:
                    oom = True                      # handled below, once the traceback no longer holds tensors
                if oom:
                    # A batch with very long passages can exceed GPU memory. A failed backward may leave partial
                    # gradients, so drop the whole accumulation step instead of updating on corrupt gradients.
                    losses = scaled = None
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    if use_cuda:
                        torch.cuda.empty_cache()
                    oom_skipped += 1
                    step += 1
                    say(f"[train] WARNING: out of memory at step {step}; skipped it "
                        f"({oom_skipped} skipped so far). Consider a smaller max_passage_len or micro-batch.")
                    if bar:
                        bar.update(1)
                        bar.set_postfix(skipped=oom_skipped, refresh=True)
                    rec = {"step": step, "oom_skipped": True, "elapsed_s": round(time.time() - start, 1)}
                    log.append(rec)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    continue
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                grad_norm, _ = optimizer_step(opt, scaler, params, cfg.max_grad_norm)
                opt.zero_grad(set_to_none=True)
                step += 1
                if bar:
                    bar.update(1)
                    bar.set_postfix(loss=f"{sums['total'] / n_micro:.3f}", rank=f"{sums['ranknet'] / n_micro:.3f}",
                                    nce=f"{sums['infonce'] / n_micro:.3f}", lr=f"{lr_at(step - 1):.1e}",
                                    skipped=oom_skipped, refresh=True)
                if step % cfg.log_every == 0 or step == total_steps:
                    rec = {"step": step, "lr": lr_at(step - 1), "grad_norm": grad_norm,
                           "loss_scale": scaler.get_scale() if scaler else None,
                           "elapsed_s": round(time.time() - start, 1),
                           **{k: v / n_micro for k, v in sums.items()}}
                    log.append(rec)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                now = time.time()
                due_steps = bool(cfg.save_every) and step % cfg.save_every == 0
                # A time-based snapshot on the very last step would just duplicate the final save.
                due_time = (bool(cfg.save_every_minutes) and step < total_steps
                            and now - last_save >= cfg.save_every_minutes * 60)
                if due_steps or due_time:
                    last_save = now                  # cadence is measured trigger to trigger, uploads included
                    try:
                        _save_snapshot(model, out_dir / "latest", state_at(step))
                        snapshot_ok = True
                    except Exception as e:  # noqa: BLE001 - a full disk must not kill hours of training
                        snapshot_ok = False
                        say(f"[train] WARNING: could not save a snapshot at step {step} ({type(e).__name__}: {e}). "
                            f"Training continues; free some disk space (check df -h /kaggle/working).")
                    if snapshot_ok:
                        if on_save:
                            on_save(out_dir, False)
                        _push(cfg, out_dir / "latest", f"{out_dir.name}/latest", f"step {step}")
            if step >= total_steps:
                break
    finally:
        log_f.close()
        if bar:
            bar.close()
    if oom_skipped:
        say(f"[train] {oom_skipped} of {step - start_step} steps were skipped because of out-of-memory errors")
    if save_final:
        latest = out_dir / "latest"
        if latest.exists() and not _has_room_for(model, out_dir):
            say("[train] disk is tight: removing the periodic snapshot to make room for the final checkpoint")
            shutil.rmtree(latest, ignore_errors=True)
        model.save_pretrained(out_dir)
    (out_dir / "train_state.json").write_text(json.dumps(state_at(step)))
    if save_final:
        _push(cfg, out_dir, out_dir.name, "final", ignore=["latest/*"])
        if on_save:
            on_save(out_dir, True)
    return log
