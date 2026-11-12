import os
from pathlib import Path

import pytest

from conftest import make_model, make_sample
from resrank.train import publish
from resrank.train.trainer import TrainConfig, train


def make_ckpts(root: Path):
    for name in ("stage1", "stage2", "smoke_test"):
        for rel in ("encoder/model.safetensors", "reranker/model.safetensors", "resrank_config.json",
                    "latest/encoder/model.safetensors"):
            f = root / name / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"{name}/{rel}")
    return root


def test_stage_hardlinks_selected_folders_and_skips_latest(tmp_path):
    root = make_ckpts(tmp_path / "ck")
    staged = publish.stage_checkpoints(root, tmp_path / "stage", names=["stage1", "stage2"])
    rels = sorted(str(p.relative_to(tmp_path / "stage")) for p in staged)
    assert "stage1/encoder/model.safetensors" in rels and "stage2/resrank_config.json" in rels
    assert not any("latest" in r or "smoke_test" in r for r in rels)
    # Hard links: same inode, so no extra disk space is used.
    assert os.stat(root / "stage1/resrank_config.json").st_ino == os.stat(tmp_path / "stage/stage1/resrank_config.json").st_ino
    with_latest = publish.stage_checkpoints(root, tmp_path / "stage2", names=["stage1"], include_latest=True)
    assert any("latest" in str(p) for p in with_latest)


def test_stage_errors(tmp_path):
    root = make_ckpts(tmp_path / "ck")
    with pytest.raises(FileNotFoundError):
        publish.stage_checkpoints(root, tmp_path / "s", names=["nope"])


def test_upload_calls_uploader_and_cleans_stage(tmp_path):
    root = make_ckpts(tmp_path / "ck")
    calls = []

    def fake(handle, folder, version_notes=""):
        calls.append((handle, sorted(p.name for p in Path(folder).iterdir()), version_notes))

    n = publish.upload_checkpoints(root, "me/ds", names=["stage1"], version_notes="hi", uploader=fake)
    assert calls == [("me/ds", ["stage1"], "hi")] and n == 3
    assert not (tmp_path / "_upload_stage").exists()
    with pytest.raises(ValueError):
        publish.upload_checkpoints(root, "no-slash", uploader=fake)


def test_check_auth(monkeypatch, tmp_path):
    for var in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="KAGGLE_API_TOKEN"):
        publish.check_auth()
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    publish.check_auth()


def test_callback_frequency_and_final(tmp_path):
    root = make_ckpts(tmp_path / "ck")
    seen = []
    cb = publish.make_save_callback(root, "me/ds", ["stage1"], every_saves=2,
                                    uploader=lambda h, f, version_notes="": seen.append(version_notes))
    for _ in range(5):
        cb(root / "stage1", False)
    cb(root / "stage1", True)
    assert seen == ["snapshot: stage1", "snapshot: stage1", "final: stage1"]   # saves 2 and 4, then final
    only_final = []
    cb2 = publish.make_save_callback(root, "me/ds", ["stage1"], 0,
                                     uploader=lambda h, f, version_notes="": only_final.append(1))
    cb2(root / "stage1", False); cb2(root / "stage1", True)
    assert only_final == [1]


def test_callback_swallows_upload_errors(tmp_path, capsys):
    root = make_ckpts(tmp_path / "ck")

    def boom(*a, **k):
        raise ConnectionError("network down")

    publish.make_save_callback(root, "me/ds", ["stage1"], 0, uploader=boom)(root / "stage1", True)
    assert "upload failed" in capsys.readouterr().out


def test_trainer_calls_on_save(tmp_path):
    model = make_model()
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(8)]
    calls = []
    cfg = TrainConfig(epochs=2, micro_batch_size=2, grad_accum=2, lr=1e-3, save_every=2)
    train(model, samples, cfg, tmp_path / "ck", on_save=lambda d, final: calls.append(final))
    assert calls == [False, False, True]      # 4 steps, saves at 2 and 4, then the final save


def test_callback_uploads_only_folders_that_exist_and_warns_once(tmp_path, capsys):
    root = make_ckpts(tmp_path / "ck")
    seen = []
    cb = publish.make_save_callback(root, "me/ds", ["stage1", "not_restored"], 0,
                                    uploader=lambda h, f, version_notes="": seen.append(sorted(p.name for p in Path(f).iterdir())))
    cb(root / "stage1", True)
    cb(root / "stage1", True)
    assert seen == [["stage1"], ["stage1"]]
    out = capsys.readouterr().out
    assert out.count("not uploading folders that are missing") == 1 and "not_restored" in out


def test_callback_with_nothing_to_upload_does_not_raise(tmp_path, capsys):
    root = make_ckpts(tmp_path / "ck")
    calls = []
    cb = publish.make_save_callback(root, "me/ds", ["missing"], 0, uploader=lambda *a, **k: calls.append(1))
    cb(root / "stage1", True)
    assert calls == [] and "nothing to upload" in capsys.readouterr().out
