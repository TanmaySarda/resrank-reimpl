import json

import pytest

from conftest import make_model, make_sample
from resrank.train.plan import plan_samples
from resrank.train.trainer import TrainConfig, load_train_state, train


def test_plan_matches_hand_calculation():
    p = plan_samples(1.45, 27)
    eff = 1.45 * (1 + 80 / 900) * 1.10                 # per-sample cost with upload overhead and safety margin
    assert p["effective_sec_per_sample"] == pytest.approx(eff, abs=1e-3)
    assert p["n_samples"] % 1000 == 0
    assert p["n_samples"] * eff / 3600 <= 27 - 4.7 + 1e-9        # never exceeds the training budget
    assert (p["n_samples"] + 1000) * eff / 3600 > 27 - 4.7       # and uses nearly all of it
    assert p["optimizer_steps"] == p["n_samples"] // 16


def test_plan_scales_and_caps():
    assert plan_samples(2.9, 27)["n_samples"] < plan_samples(1.45, 27)["n_samples"]
    assert plan_samples(0.01, 27)["n_samples"] == 232_000        # capped at the size of the PE-Rank data
    with pytest.raises(ValueError):
        plan_samples(1.0, 4.0)                                    # budget smaller than the reserve


def run(tmp_path, **kw):
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(12)]
    cfg = TrainConfig(epochs=2, micro_batch_size=2, grad_accum=2, lr=1e-3, **kw)
    return train(make_model(), samples, cfg, tmp_path)


def test_progress_bar_shows_percentage_and_steps(tmp_path, capfd):
    run(tmp_path / "a", progress=True)
    err = capfd.readouterr().err
    assert "training:" in err and "100%" in err and "6/6" in err and "loss=" in err and "skipped=0" in err


def test_progress_can_be_turned_off(tmp_path, capfd):
    run(tmp_path / "b", progress=False)
    assert "training:" not in capfd.readouterr().err        # (model saving prints its own shard bars)


def test_resumed_bar_starts_at_the_resumed_step(tmp_path, capfd):
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(12)]
    cfg = TrainConfig(epochs=2, micro_batch_size=2, grad_accum=2, lr=1e-3, progress=True)
    train(make_model(), samples, cfg, tmp_path / "c", start_step=4)
    err = capfd.readouterr().err
    assert "5/6" in err and "6/6" in err and "1/6" not in err
