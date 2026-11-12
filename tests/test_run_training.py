import json
import sys

import pytest
import yaml

from conftest import make_model, make_sample
from resrank.io_utils import write_jsonl
from resrank.paths import Paths
from resrank.train import run_training as rt
from resrank.train.trainer import load_train_state

N_SAMPLES = 8       # micro-batch 1 x grad_accum 2 -> 4 optimizer steps


@pytest.fixture
def env(tmp_path, monkeypatch):
    paths = Paths.from_env(tmp_path).ensure()
    write_jsonl(paths.train_file("stage1"), (make_sample(4, seed=i, qid=f"q{i}").to_dict() for i in range(N_SAMPLES)))
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"root": str(tmp_path), "seed": 5}))
    tcfg = tmp_path / "train.yaml"
    tcfg.write_text(yaml.safe_dump({
        "train_file": "stage1", "out_name": "run", "init_from": None, "epochs": 1, "micro_batch_size": 1,
        "grad_accum": 2, "lr": 1e-3, "precision": "fp32", "optimizer": "adamw", "log_every": 1,
        "save_every": 0, "save_every_minutes": 1e-9, "progress": False}))
    monkeypatch.setattr(rt, "load_model", lambda cfg, ckpt_dir=None, **kw: make_model())
    calls = {"train": 0}
    real_train = rt.train

    def counting_train(*a, **k):
        calls["train"] += 1
        return real_train(*a, **k)

    monkeypatch.setattr(rt, "train", counting_train)

    def run(*extra):
        monkeypatch.setattr(sys, "argv", ["run_training", "--config", str(base), "--train-config", str(tcfg), *extra])
        rt.main()

    return paths, run, calls, real_train, monkeypatch


def test_auto_resume_fresh_then_finished(env, capsys):
    paths, run, calls, *_ = env
    run("--auto-resume")
    assert "starting a fresh run" in capsys.readouterr().out
    assert load_train_state(paths.stage_ckpt("run"))["step"] == 4 and calls["train"] == 1
    run("--auto-resume")                                   # nothing left to do
    assert "already finished" in capsys.readouterr().out and calls["train"] == 1


def test_auto_resume_continues_after_an_interruption(env, capsys):
    paths, run, calls, real_train, monkeypatch = env

    class Disconnect(Exception):
        pass

    def die(out_dir, final):
        raise Disconnect()

    monkeypatch.setattr(rt, "train", lambda *a, **k: real_train(*a, **{**k, "on_save": die}))
    with pytest.raises(Disconnect):
        run("--auto-resume")
    out = paths.stage_ckpt("run")
    assert load_train_state(out / "latest")["step"] == 1 and not (out / "train_state.json").exists()

    monkeypatch.setattr(rt, "train", lambda *a, **k: real_train(*a, **k))
    capsys.readouterr()
    run("--auto-resume")
    assert "resuming from" in capsys.readouterr().out
    assert load_train_state(out)["step"] == 4
    log_steps = [json.loads(l)["step"] for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert log_steps == [1, 2, 3, 4]


def test_train_file_override_and_limit_and_missing_file(env):
    paths, run, calls, *_ = env
    write_jsonl(paths.train_file("timing"), (make_sample(4, seed=i, qid=f"t{i}").to_dict() for i in range(4)))
    run("--train-file", "timing", "--out-name", "timing_run", "--no-publish")
    assert load_train_state(paths.stage_ckpt("timing_run"))["n_samples"] == 4
    with pytest.raises(SystemExit, match="not found"):
        run("--train-file", "nope")


def test_changed_sample_count_refuses_to_resume(env):
    paths, run, calls, real_train, monkeypatch = env

    class Disconnect(Exception):
        pass

    monkeypatch.setattr(rt, "train", lambda *a, **k: real_train(*a, **{**k, "on_save": lambda d, f: (_ for _ in ()).throw(Disconnect())}))
    with pytest.raises(Disconnect):
        run("--auto-resume")
    with pytest.raises(ValueError, match="n_samples"):
        run("--auto-resume", "--limit", "6")               # a different sample count than the interrupted run
