"""CLI for stages s04 (stage 1) and s05 (stage 2).

    python -m resrank.train.run_training --train-config configs/stage1_small.yaml --auto-resume

--auto-resume starts fresh on a first run, continues from the newest snapshot after an interruption, and does
nothing if the run already finished, so the same command works every time.
"""
from __future__ import annotations

import argparse
import json

import torch

from resrank.config import load_config
from resrank.model.loading import load_model
from resrank.paths import Paths
from resrank.torch_utils import default_device
from resrank.train.trainer import (TrainConfig, check_resume_state, load_samples, load_train_state,
                                   train)
from resrank.util import say


def count_lines(path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def resolve_init(paths: Paths, init: str) -> str:
    """Checkpoint name to initialize from. A run that was stopped before it finished (e.g. stage 1 stopped at a
    plateau) has no final weights in checkpoints/<init>/, only the periodic snapshot in <init>/latest; use that
    snapshot instead of failing, and say so."""
    if (paths.stage_ckpt(init) / "encoder").is_dir():
        return init
    snap = f"{init}/latest"
    if (paths.stage_ckpt(snap) / "encoder").is_dir():
        st = paths.stage_ckpt(snap) / "train_state.json"
        where = ""
        if st.exists():
            state = load_train_state(paths.stage_ckpt(snap))
            where = f" (step {state.get('step')} of {state.get('total_steps')})"
        say(f"init_from: {init} has no final checkpoint; using its last snapshot {snap}{where}")
        return snap
    raise SystemExit(f"init_from: no checkpoint at {paths.stage_ckpt(init)} (neither final weights nor latest/). "
                     f"Restore it first (e.g. copy it from your Kaggle dataset).")


def quiet_libraries() -> None:
    """Silence transformers' progress bars and info messages (the training progress bar is the one to watch)."""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--train-config", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training samples")
    ap.add_argument("--train-file", default=None, help="override train_file (a name under data/02_train/)")
    ap.add_argument("--no-publish", action="store_true", help="do not upload checkpoints (e.g. for trial runs)")
    ap.add_argument("--no-save", action="store_true",
                    help="do not save weights at the end (timing runs: keeps the log and state, saves about 4.8 GB of disk)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from checkpoints/<out_name>/latest (weights, step, data position)")
    ap.add_argument("--auto-resume", action="store_true",
                    help="resume if a snapshot exists, skip if the run already finished, otherwise start fresh")
    ap.add_argument("--out-name", default=None, help="override out_name (e.g. timing_run) so a trial run does not overwrite a real checkpoint")
    ap.add_argument("--hub-repo", default=None, help="back up checkpoints to this Hugging Face repo (needs HF_TOKEN)")
    a = ap.parse_args()
    quiet_libraries()

    cfg, tcfg_raw = load_config(a.config), load_config(a.train_config)
    paths = Paths.from_env(a.root or cfg.get("root")).ensure()
    tcfg = TrainConfig.from_dict({"seed": cfg.get("seed", 42), **tcfg_raw})
    if a.hub_repo:
        tcfg.hub_repo = a.hub_repo
    out = paths.stage_ckpt(a.out_name or tcfg_raw["out_name"])
    train_path = paths.train_file(a.train_file or tcfg_raw["train_file"])
    if not train_path.exists():
        raise SystemExit(f"training file not found: {train_path}. Build it with resrank.data.s02_build_train first.")

    publishing = bool(tcfg_raw.get("publish_dataset")) and not a.no_publish
    if publishing:
        from resrank.train.publish import check_auth
        check_auth()   # fail now, not after hours of training

    n_samples = count_lines(train_path)
    if a.limit:
        n_samples = min(n_samples, a.limit)

    if a.auto_resume:
        final = out / "train_state.json"
        if final.exists():
            st = load_train_state(out)
            if st["step"] >= st["total_steps"] and st.get("n_samples") == n_samples:
                say(f"{out.name} already finished ({st['step']} steps). Nothing to do.")
                return
        a.resume = any((out / d / "train_state.json").exists() for d in ("latest", "latest.old"))
        say("snapshot found: resuming" if a.resume else "no snapshot found: starting a fresh run")

    resume_dir, state = None, None
    if a.resume:
        resume_dir = next((d for d in (out / "latest", out / "latest.old") if (d / "train_state.json").exists()), None)
        if resume_dir is None:
            raise SystemExit(f"--resume: no snapshot with train_state.json under {out}. Restore the checkpoint "
                             f"folder first (e.g. copy it from your Kaggle dataset).")
        state = load_train_state(resume_dir)
        check_resume_state(state, tcfg, n_samples)       # fail before the model is loaded
        say(f"resuming from {resume_dir} at step {state['step']} of {state['total_steps']}")

    init = tcfg_raw.get("init_from")
    init_ckpt = resolve_init(paths, init) if (init and not resume_dir) else init
    # fp16 autocast needs fp32 master weights, so weights stay in fp32 unless precision is bf16.
    weights = "bf16" if tcfg.precision == "bf16" else "fp32"
    model = load_model(cfg, resume_dir if resume_dir else (paths.stage_ckpt(init_ckpt) if init else None),
                       device=default_device(), weights_dtype=weights)
    samples = load_samples(train_path, a.limit)
    say(f"training on {len(samples)} samples -> {out}")

    on_save = None
    if publishing:
        from resrank.train.publish import make_save_callback
        # publish_init: false keeps the init checkpoint out of this dataset (e.g. it has its own dataset)
        init_name = init if tcfg_raw.get("publish_init", True) else None
        names = [n for n in (out.name, init_name, *(tcfg_raw.get("publish_include") or [])) if n]
        names = list(dict.fromkeys(names))
        on_save = make_save_callback(paths.checkpoints, tcfg_raw["publish_dataset"], names,
                                     int(tcfg_raw.get("publish_every_saves", 0)))
    log = train(model, samples, tcfg, out, on_save=on_save,
                start_step=state["step"] if state else 0,
                start_loss_scale=state.get("loss_scale") if state else None,
                save_final=not a.no_save)
    if torch.cuda.is_available():
        say(f"peak GPU memory this session: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    done = [r for r in log if not r.get("oom_skipped")]
    if done:
        say("finished. last logged: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                     for k, v in done[-1].items()}))


if __name__ == "__main__":
    main()
