import json
from pathlib import Path

import pytest

from conftest import make_model, make_sample
from resrank.train import publish
from resrank.train.trainer import (TrainConfig, _save_snapshot, check_resume_state, load_train_state,
                                   train)


def samples(n=12):
    return [make_sample(4, seed=i, qid=f"q{i}") for i in range(n)]


def recorded(model):
    """Wrap training_losses so we can see which samples each micro-batch used."""
    seen = []
    original = model.training_losses

    def wrapper(batch):
        seen.append([s.qid for s in batch])
        return original(batch)

    model.training_losses = wrapper
    return seen


CFG = dict(epochs=2, micro_batch_size=2, grad_accum=2, lr=1e-3, log_every=1, save_every=2, seed=7)
# 12 samples / (2*2) = 3 steps per epoch, 6 steps in total.


def test_interrupted_then_resumed_run_sees_the_same_batches_as_an_uninterrupted_run(tmp_path):
    full_model = make_model()
    full_seen = recorded(full_model)
    train(full_model, samples(), TrainConfig(**CFG), tmp_path / "full")
    assert len(full_seen) == 12                                         # 6 steps x 2 micro-batches

    class Disconnect(Exception):
        pass

    def die_after_first_snapshot(out_dir, final):
        raise Disconnect()

    crash_model = make_model()
    crash_seen = recorded(crash_model)
    with pytest.raises(Disconnect):
        train(crash_model, samples(), TrainConfig(**CFG), tmp_path / "run", on_save=die_after_first_snapshot)
    state = load_train_state(tmp_path / "run" / "latest")
    assert state["step"] == 2 and state["total_steps"] == 6 and state["seed"] == 7
    assert len(crash_seen) == 4                                         # steps 1 and 2 only

    resumed_model = make_model()
    resumed_seen = recorded(resumed_model)
    check_resume_state(state, TrainConfig(**CFG), 12)
    log = train(resumed_model, samples(), TrainConfig(**CFG), tmp_path / "run", start_step=state["step"])
    assert crash_seen + resumed_seen == full_seen                       # nothing repeated, nothing skipped
    assert [r["step"] for r in log] == [3, 4, 5, 6]
    # The log file continues instead of being overwritten.
    steps_in_file = [json.loads(l)["step"] for l in (tmp_path / "run" / "train_log.jsonl").read_text().splitlines()]
    assert steps_in_file == [1, 2, 3, 4, 5, 6]


def test_data_order_depends_only_on_seed_and_epoch(tmp_path):
    seen_a, seen_b, seen_c = [], [], []
    for seen, seed in ((seen_a, 1), (seen_b, 1), (seen_c, 2)):
        m = make_model()
        rec = recorded(m)
        train(m, samples(), TrainConfig(**{**CFG, "seed": seed, "save_every": 0}), tmp_path / f"r{seed}{len(seen_a)}{len(seen_b)}")
        seen.extend(rec)
    assert seen_a == seen_b and seen_a != seen_c


def test_learning_rate_schedule_continues_and_ramps_after_resume(tmp_path):
    cfg = TrainConfig(**{**CFG, "save_every": 0})
    full_log = train(make_model(), samples(), cfg, tmp_path / "full")
    resumed = train(make_model(), samples(), cfg, tmp_path / "res", start_step=3, resume_ramp_steps=3)
    assert [r["step"] for r in resumed] == [4, 5, 6]
    full_lr = {r["step"]: r["lr"] for r in full_log}
    # Same schedule position, scaled by the 1/3, 2/3, 3/3 ramp.
    for step, frac in ((4, 1 / 3), (5, 2 / 3), (6, 1.0)):
        assert {r["step"]: r["lr"] for r in resumed}[step] == pytest.approx(full_lr[step] * frac)


def test_resume_refuses_mismatched_settings():
    state = {"seed": 7, "micro_batch_size": 2, "grad_accum": 2, "epochs": 2, "n_samples": 12, "lr": 1e-3}
    check_resume_state(state, TrainConfig(**CFG), 12)
    with pytest.raises(ValueError, match="n_samples"):
        check_resume_state(state, TrainConfig(**CFG), 20)
    with pytest.raises(ValueError, match="seed"):
        check_resume_state(state, TrainConfig(**{**CFG, "seed": 8}), 12)


def test_start_step_beyond_plan_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="beyond"):
        train(make_model(), samples(), TrainConfig(**CFG), tmp_path / "x", start_step=99)


def test_final_checkpoint_also_has_state(tmp_path):
    train(make_model(), samples(), TrainConfig(**{**CFG, "save_every": 0}), tmp_path / "x")
    assert load_train_state(tmp_path / "x")["step"] == 6


def test_snapshot_swap_keeps_previous_until_new_one_is_complete(tmp_path):
    m = make_model()
    folder = tmp_path / "latest"
    _save_snapshot(m, folder, {"step": 1})
    assert load_train_state(folder)["step"] == 1

    class Boom(Exception):
        pass

    def failing_save(path):
        raise Boom()

    original = m.save_pretrained
    m.save_pretrained = failing_save
    with pytest.raises(Boom):
        _save_snapshot(m, folder, {"step": 2})
    m.save_pretrained = original
    assert load_train_state(folder)["step"] == 1                        # the old snapshot is untouched
    _save_snapshot(m, folder, {"step": 3})
    assert load_train_state(folder)["step"] == 3
    assert not (tmp_path / "latest.tmp").exists() and not (tmp_path / "latest.old").exists()


def test_publish_never_stages_temp_or_old_snapshots(tmp_path):
    root = tmp_path / "ck"
    for rel in ("run/encoder/w.bin", "run/latest/encoder/w.bin", "run/latest.tmp/encoder/w.bin",
                "run/latest.old/encoder/w.bin"):
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    staged = publish.stage_checkpoints(root, tmp_path / "s", ["run"], include_latest=True)
    rels = sorted(str(p.relative_to(tmp_path / "s")) for p in staged)
    assert rels == ["run/encoder/w.bin", "run/latest/encoder/w.bin"]


def test_out_of_memory_skips_the_step_instead_of_crashing(tmp_path, capsys):
    import torch

    model = make_model()
    original = model.training_losses
    calls = {"n": 0}

    def flaky(batch):
        calls["n"] += 1
        if calls["n"] == 3:                       # the first micro-batch of step 2
            raise torch.OutOfMemoryError("CUDA out of memory (simulated)")
        return original(batch)

    model.training_losses = flaky
    log = train(model, samples(), TrainConfig(**{**CFG, "save_every": 0}), tmp_path / "run")
    assert [r["step"] for r in log] == [1, 2, 3, 4, 5, 6]               # the run finished all 6 steps
    skipped = [r for r in log if r.get("oom_skipped")]
    assert [r["step"] for r in skipped] == [2]
    assert all("total" in r for r in log if not r.get("oom_skipped"))
    assert load_train_state(tmp_path / "run")["step"] == 6              # resume arithmetic still holds
    assert "skipped it" in capsys.readouterr().out


def test_time_based_snapshots(tmp_path):
    base = {**CFG, "save_every": 0}
    # A tiny interval means every step is "due", except the last one (the final save covers it).
    calls = []
    train(make_model(), samples(), TrainConfig(**{**base, "save_every_minutes": 1e-9}), tmp_path / "a",
          on_save=lambda d, final: calls.append(final))
    assert calls == [False] * 5 + [True]                  # 6 steps: snapshots after steps 1-5, then the final save
    assert load_train_state(tmp_path / "a" / "latest")["step"] == 5
    # A long interval means no periodic snapshot at all.
    calls = []
    train(make_model(), samples(), TrainConfig(**{**base, "save_every_minutes": 1000}), tmp_path / "b",
          on_save=lambda d, final: calls.append(final))
    assert calls == [True] and not (tmp_path / "b" / "latest").exists()


def test_step_and_time_triggers_save_once_per_step(tmp_path):
    calls = []
    cfg = TrainConfig(**{**CFG, "save_every": 2, "save_every_minutes": 1e-9})
    train(make_model(), samples(), cfg, tmp_path / "c", on_save=lambda d, final: calls.append(final))
    # Steps 1-5 are due by time, step 2 and 4 also by steps, and step 6 by steps (it is not excluded for step-based saves).
    assert calls == [False] * 6 + [True]
