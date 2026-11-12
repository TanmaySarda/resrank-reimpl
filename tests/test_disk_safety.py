import logging
import sys
import types
from pathlib import Path

import pytest

from conftest import make_model, make_sample
from resrank.train import publish
from resrank.train import trainer as T
from resrank.train.trainer import TrainConfig, load_train_state, train

CFG = dict(epochs=1, micro_batch_size=2, grad_accum=2, lr=1e-3, progress=False)      # 12 samples -> 3 steps


def samples(n=12):
    return [make_sample(4, seed=i, qid=f"q{i}") for i in range(n)]


def test_snapshot_uses_atomic_swap_when_there_is_room(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_free_bytes", lambda p: 10**12)
    m = make_model()
    T._save_snapshot(m, tmp_path / "latest", {"step": 1})
    T._save_snapshot(m, tmp_path / "latest", {"step": 2})
    assert load_train_state(tmp_path / "latest")["step"] == 2
    assert not (tmp_path / "latest.tmp").exists() and not (tmp_path / "latest.old").exists()


def test_snapshot_replaces_in_place_when_disk_is_tight(tmp_path, monkeypatch, capsys):
    m = make_model()
    monkeypatch.setattr(T, "_free_bytes", lambda p: 10**12)
    T._save_snapshot(m, tmp_path / "latest", {"step": 1})
    monkeypatch.setattr(T, "_free_bytes", lambda p: 0)                    # no room for a second copy
    seen_during_save = []
    original = m.save_pretrained

    def spy(path):
        seen_during_save.append(sorted(p.name for p in tmp_path.iterdir()))
        return original(path)

    m.save_pretrained = spy
    T._save_snapshot(m, tmp_path / "latest", {"step": 2})
    assert "disk is tight" in capsys.readouterr().out
    assert seen_during_save == [[]]                                         # the old copy was deleted before writing
    assert load_train_state(tmp_path / "latest")["step"] == 2 and (tmp_path / "latest/encoder/config.json").exists()


def test_failed_save_leaves_no_partial_copy(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_free_bytes", lambda p: 10**12)
    m = make_model()
    T._save_snapshot(m, tmp_path / "latest", {"step": 1})
    original = m.save_pretrained

    def partial_then_fail(path):
        original(path)                                                      # writes the big files...
        raise OSError(28, "No space left on device")                        # ...then fails like the real run did

    m.save_pretrained = partial_then_fail
    with pytest.raises(OSError):
        T._save_snapshot(m, tmp_path / "latest", {"step": 2})
    assert not (tmp_path / "latest.tmp").exists()                           # the half-written copy is removed
    assert load_train_state(tmp_path / "latest")["step"] == 1               # the previous snapshot survives


def test_snapshot_failure_does_not_stop_training(tmp_path, monkeypatch, capsys):
    real = T._save_snapshot
    calls = {"n": 0}

    def flaky(model, folder, state):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(28, "No space left on device")
        return real(model, folder, state)

    monkeypatch.setattr(T, "_save_snapshot", flaky)
    saved = []
    log = train(make_model(), samples(), TrainConfig(**CFG, save_every=1), tmp_path / "run",
                on_save=lambda d, final: saved.append(final))
    out = capsys.readouterr().out
    assert [r["step"] for r in log] == [1, 2, 3]                            # all steps still ran
    assert "could not save a snapshot at step 1" in out and "No space left" in out
    assert saved == [False, False, True]                                    # step 1's failed snapshot was not uploaded
    assert load_train_state(tmp_path / "run")["step"] == 3


def test_final_save_frees_the_periodic_snapshot_when_disk_is_tight(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "_free_bytes", lambda p: 0)
    train(make_model(), samples(), TrainConfig(**CFG, save_every=1), tmp_path / "run")
    assert not (tmp_path / "run" / "latest").exists()
    assert (tmp_path / "run/encoder/config.json").exists() and load_train_state(tmp_path / "run")["step"] == 3


def test_no_save_keeps_log_and_state_but_no_weights(tmp_path):
    saved = []
    train(make_model(), samples(), TrainConfig(**CFG), tmp_path / "run", save_final=False,
          on_save=lambda d, final: saved.append(final))
    assert load_train_state(tmp_path / "run")["step"] == 3 and (tmp_path / "run/train_log.jsonl").exists()
    assert not (tmp_path / "run/encoder").exists() and not (tmp_path / "run/reranker").exists() and saved == []


def test_default_uploader_silences_kagglehub_logging_and_restores_it(monkeypatch, capsys):
    logger = logging.getLogger("fake_kagglehub")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))          # bound to the stream at creation, like kagglehub's
    fake = types.ModuleType("kagglehub")

    def dataset_upload(handle, folder, version_notes=""):
        logger.info("Starting upload for file x")
        if handle == "bad/ds":
            raise ConnectionError("network down")

    fake.dataset_upload = dataset_upload
    monkeypatch.setitem(sys.modules, "kagglehub", fake)
    upload = publish._default_uploader()
    upload("me/ds", "/tmp/x", version_notes="n")
    assert "Starting upload" not in capsys.readouterr().out
    assert logging.root.manager.disable == logging.NOTSET
    with pytest.raises(RuntimeError, match="network down"):
        upload("bad/ds", "/tmp/x")
    assert logging.root.manager.disable == logging.NOTSET                    # restored even on failure
    logger.info("visible again")
    assert "visible again" in capsys.readouterr().out
