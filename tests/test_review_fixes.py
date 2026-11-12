"""Tests for the fixes from the code review (language filter, publishing, log summary, init_from, decontamination,
gradient accumulation)."""
import json

import pytest
import torch

from conftest import make_model, make_sample
from resrank.data.filters import is_english_sample, non_latin_letter_ratio
from resrank.data.s02_build_train import build_stage2, load_eval_queries, normalize_query
from resrank.io_utils import read_jsonl, write_jsonl
from resrank.model.sequence import length_mask, pad_stack
from resrank.paths import Paths
from resrank.schemas import RankingSample
from resrank.train import publish
from resrank.train.log_summary import summarize
from resrank.train.run_training import resolve_init
from resrank.train.trainer import TrainConfig, train
from test_data_utils import e2rank_row


def _sample(query, passages):
    return RankingSample(query=query, passages=passages, ranking=list(range(len(passages))), pos_index=0)


def test_non_latin_scripts_are_not_english():
    english = _sample("what is the half-life of caffeine", ["Caffeine has a half-life of about 5 hours.",
                                                             "Dosage in μg/kg, after Müller et al."])
    assert is_english_sample(english)
    for foreign in ("что такое кофеин", "ما هو الكافيين", "คาเฟอีนคืออะไร", "कैफीन क्या है"):
        assert not is_english_sample(_sample(foreign, ["English passage."]))
        assert not is_english_sample(_sample("an english query", ["fine", foreign]))
    assert non_latin_letter_ratio("1234 !!") == 0.0


def test_latest_snapshot_is_uploaded_when_it_is_the_only_copy(tmp_path):
    root = tmp_path / "ck"
    for rel in ("stage1_full/latest/encoder/w", "stage1_full/train_log.jsonl",          # stopped: no final weights
                "stage2/encoder/w", "stage2/latest/encoder/w"):                         # finished
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(rel)
    staged = publish.stage_checkpoints(root, tmp_path / "stage", names=["stage1_full", "stage2"])
    rels = {str(p.relative_to(tmp_path / "stage")) for p in staged}
    assert "stage1_full/latest/encoder/w" in rels          # only copy of stage 1: kept
    assert "stage2/encoder/w" in rels and "stage2/latest/encoder/w" not in rels   # superseded: skipped


def test_log_summary_sums_time_over_resumed_sessions(tmp_path):
    recs = [{"step": s, "total": 1.0, "infonce": 1.0, "ranknet": 1.0, "grad_norm": 1.0, "loss_scale": None,
             "elapsed_s": e} for s, e in [(1, 10.0), (2, 20.0), (3, 10.0), (4, 20.0), (5, 30.0)]]
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (tmp_path / "train_state.json").write_text(json.dumps({"micro_batch_size": 1, "grad_accum": 1}))
    s = summarize(tmp_path)
    assert s["sessions"] == 2 and s["elapsed_s_total"] == 50.0 and s["sec_per_sample_incl_warmup"] == 10.0


def test_resolve_init_falls_back_to_the_snapshot_of_a_stopped_run(tmp_path, capsys):
    paths = Paths.from_env(tmp_path).ensure()
    (paths.stage_ckpt("s1/latest") / "encoder").mkdir(parents=True)
    (paths.stage_ckpt("s1/latest") / "train_state.json").write_text(json.dumps({"step": 7, "total_steps": 9}))
    assert resolve_init(paths, "s1") == "s1/latest" and "step 7 of 9" in capsys.readouterr().out
    (paths.stage_ckpt("s1") / "encoder").mkdir()
    assert resolve_init(paths, "s1") == "s1"
    with pytest.raises(SystemExit):
        resolve_init(paths, "missing")


def test_exclude_queries_drops_evaluation_queries(tmp_path):
    rows = [{**e2rank_row(chinese=False), "query": q} for q in ("What is BM25?", "an unrelated question", "what is bm25")]
    write_jsonl(tmp_path / "e2.jsonl", rows)
    write_jsonl(tmp_path / "queries.jsonl", [{"_id": "1", "text": "what is BM25"}])
    exclude = load_eval_queries([tmp_path / "queries.jsonl"])
    assert exclude == {normalize_query("What is BM25?")}
    stats = build_stage2(tmp_path / "e2.jsonl", tmp_path / "out.jsonl", 10, exclude_queries=exclude)
    assert stats["dropped_eval_overlap"] == 2 and stats["written"] == 1
    assert [r["query"] for r in read_jsonl(tmp_path / "out.jsonl")] == ["an unrelated question"]
    plain = build_stage2(tmp_path / "e2.jsonl", tmp_path / "out2.jsonl", 10)        # default: nothing removed
    assert plain["written"] == 3 and "dropped_eval_overlap" not in plain


def test_masks_and_padding():
    m = length_mask([2, 0, 3], 3)
    assert m.tolist() == [[True, True, False], [False, False, False], [True, True, True]]
    out = pad_stack([torch.ones(2, 4), torch.ones(1, 4)])
    assert out.shape == (2, 2, 4) and out[1, 1].abs().sum() == 0


def test_short_last_micro_batch_does_not_inflate_the_gradient(tmp_path, monkeypatch):
    """3 samples, micro-batch 2: the step has 2 micro-batches, so each micro loss is divided by 2 (not 1)."""
    model = make_model()
    seen = []
    real = model.training_losses

    def spy(batch):
        out = real(batch)
        seen.append(out["total"])
        return out

    monkeypatch.setattr(model, "training_losses", spy)
    divisors = []
    real_backward = torch.Tensor.backward

    def track(self, *a, **k):
        divisors.append(float(seen[-1].detach()) / float(self.detach()))
        return real_backward(self, *a, **k)

    monkeypatch.setattr(torch.Tensor, "backward", track)
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(3)]
    train(model, samples, TrainConfig(micro_batch_size=2, grad_accum=2, lr=1e-4, progress=False), tmp_path / "o",
          save_final=False)
    assert divisors == pytest.approx([2.0, 2.0])


def test_source_report_separates_latin_script_languages(tmp_path):
    from resrank.data.e2rank import source_report, suggest_english_sources
    from resrank.data.filters import english_word_rate
    en = "The capital of France is Paris, which is on the Seine and is known for the Louvre."
    fi = "Ranskan pääkaupunki on Pariisi, joka sijaitsee Seinen varrella ja tunnetaan Louvresta."
    sw = "Mji mkuu wa Ufaransa ni Paris, ulio kando ya mto Seine na unajulikana kwa Louvre."
    assert english_word_rate(en) > 0.25 and english_word_rate(fi) < 0.1 and english_word_rate(sw) < 0.1
    rows = []
    for src, text in (("msmarco", en), ("mr_tydi_finnish", fi), ("miracl_sw", sw), ("cMedQAv2", None)):
        row = e2rank_row(chinese=text is None, source=src)
        if text:
            row = {**row, "query": text, "document": [text] * 16}
        rows += [row, row]
    write_jsonl(tmp_path / "e2.jsonl", rows)
    rep = source_report(tmp_path / "e2.jsonl")
    assert rep["mr_tydi_finnish"]["frac_pass_script_filter"] == 1.0      # the script filter cannot catch it...
    assert suggest_english_sources(rep) == ["msmarco"]                  # ...the word rate does


def test_publish_init_false_keeps_init_out_of_the_upload(tmp_path, monkeypatch):
    import sys
    import yaml
    from resrank.train import run_training as rt
    from resrank.train import publish as pub
    paths = Paths.from_env(tmp_path).ensure()
    write_jsonl(paths.train_file("s2"), (make_sample(4, seed=i, qid=f"q{i}").to_dict() for i in range(4)))
    (paths.stage_ckpt("init_ck") / "encoder").mkdir(parents=True)
    names_seen = []
    monkeypatch.setattr(rt, "load_model", lambda cfg, ckpt_dir=None, **kw: make_model())
    monkeypatch.setattr(pub, "check_auth", lambda: None)
    monkeypatch.setattr(pub, "make_save_callback", lambda root, handle, names, every: names_seen.append(names) or None)
    base = tmp_path / "base.yaml"; base.write_text(yaml.safe_dump({"root": str(tmp_path)}))
    for flag, expected in ((None, ["out", "init_ck"]), (False, ["out"])):
        cfg = {"train_file": "s2", "out_name": "out", "init_from": "init_ck", "micro_batch_size": 1, "grad_accum": 2,
               "lr": 1e-3, "progress": False, "publish_dataset": "me/ds"}
        if flag is not None:
            cfg["publish_init"] = flag
        t = tmp_path / "t.yaml"; t.write_text(yaml.safe_dump(cfg))
        monkeypatch.setattr(sys, "argv", ["x", "--config", str(base), "--train-config", str(t), "--no-save"])
        rt.main()
        assert names_seen[-1] == expected
