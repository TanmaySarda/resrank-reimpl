import json
import py_compile
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import make_model, make_sample
from resrank.config import load_config
from resrank.data.bm25 import BM25
from resrank.data.pe_rank import (inspect_pe_rank_row, iter_pe_rank_samples, parse_user_prompt,
                                  pe_rank_row_to_sample)
from resrank.data.s02_build_train import build_stage1, build_stage2
from resrank.data.s03_bm25 import run_bm25
from resrank.data.sampling import shuffle_candidates, subsample_candidates
from resrank.eval.s06_encode_corpus import build_index
from resrank.eval.s07_rerank import rerank_run
from resrank.eval.s08_evaluate import evaluate_files, write_results
from resrank.io_utils import read_json, read_jsonl, write_jsonl, write_trec_run
from resrank.model.loading import resrank_config_from_cfg
from resrank.schemas import RankingSample, SchemaError
from resrank.train.trainer import TrainConfig, load_samples, train
from test_data_utils import e2rank_row

ROOT = Path(__file__).resolve().parent.parent


# ---- PE-Rank ---------------------------------------------------------------
def pe_row(n=3, postamble=False):
    passages = [f"{w} text here." for w in ("Alpha", "Beta", "Gamma", "Delta", "Epsilon")[:n]]
    body = "".join(f"\n\nPassage {i + 1}: [<PLACEHOLDER>] {p}" for i, p in enumerate(passages))
    if postamble:
        body += f"\n\nRank the {n} passages above based on their relevance."
    user = ("I will provide you with {} passages, each with a special token representing the passage "
            "enclosed in [], followed by original text.\nRank the passages based on their relevance to "
            "the search query: define extreme..{}").format(n, body)
    ranking = list(range(n, 0, -1))
    return {
        "messages_w_content": [{"role": "user", "content": user},
                               {"role": "assistant", "content": " > ".join(f"[{r}]" for r in ranking)}],
        "messages_wo_content": [],
        "extra_texts": passages + passages,
        "ranking": ranking,
    }


def test_parse_user_prompt_and_postamble():
    q, ps = parse_user_prompt(pe_row(3, postamble=True)["messages_w_content"][0]["content"])
    assert q == "define extreme"
    assert ps == ["Alpha text here.", "Beta text here.", "Gamma text here."]


def test_pe_rank_row_to_sample():
    s = pe_rank_row_to_sample(pe_row(3))
    assert s.ranking == [2, 1, 0] and s.pos_index == 2        # positive = top-ranked passage
    assert s.passages[0] == "Alpha text here."
    assert pe_rank_row_to_sample(pe_row(3), passage_source="extra_second").passages == s.passages


def test_pe_rank_bad_rows_rejected():
    bad = pe_row(3)
    bad["messages_w_content"][0]["content"] = "no passages here"
    with pytest.raises(SchemaError):
        pe_rank_row_to_sample(bad)
    bad2 = pe_row(3)
    bad2["ranking"] = [1, 1, 2]
    with pytest.raises(SchemaError):
        pe_rank_row_to_sample(bad2)


def test_inspect_reports_which_extra_view_matches():
    info = inspect_pe_rank_row(pe_row(3))
    assert info["n_passages"] == 3 and info["ranking_is_permutation_1_to_n"]
    assert info["assistant_matches_ranking"] is True
    assert info["extra_texts_matches_parsed"]["extra_first"] and info["extra_texts_matches_parsed"]["extra_second"]
    assert not info["extra_texts_matches_parsed"]["extra_even"]


def test_iter_pe_rank_counts_mismatch_and_invalid(tmp_path):
    rows = [pe_row(3), pe_row(3), {"messages_w_content": []}]
    rows[1]["messages_w_content"][1]["content"] = "[1] > [2] > [3]"     # disagrees with `ranking`
    write_jsonl(tmp_path / "pe.jsonl", rows)
    stats = {}
    kept = list(iter_pe_rank_samples(tmp_path / "pe.jsonl", stats=stats))
    assert len(kept) == 2
    assert stats == {"read": 3, "invalid": 1, "ranking_text_mismatch": 1, "kept": 2}


# ---- sampling --------------------------------------------------------------
def test_subsample_keeps_positive_and_relative_ranking():
    s = make_sample(10, seed=4)
    sub = subsample_candidates(s, 4, random.Random(0))
    assert sub.n == 4 and sorted(sub.ranking) == [0, 1, 2, 3]
    assert sub.passages[sub.pos_index] == s.passages[s.pos_index]
    assert sub.ranking[0] == sub.pos_index          # make_sample puts the positive first in the ranking
    kept_in_order = [p for p in s.passages if p in set(sub.passages)]
    assert sub.passages == kept_in_order            # original candidate order preserved
    assert subsample_candidates(s, 20, random.Random(0)) is s


def test_shuffle_follows_passages():
    s = make_sample(8, seed=5)
    sh = shuffle_candidates(s, random.Random(1))
    assert sh.passages != s.passages and sorted(sh.passages) == sorted(s.passages)
    assert [sh.passages[i] for i in sh.ranking] == [s.passages[i] for i in s.ranking]
    assert sh.passages[sh.pos_index] == s.passages[s.pos_index]


# ---- build stages ----------------------------------------------------------
def test_build_stage1_and_stage2(tmp_path):
    write_jsonl(tmp_path / "pe.jsonl", [pe_row(5) for _ in range(6)])
    stats = build_stage1(tmp_path / "pe.jsonl", tmp_path / "s1.jsonl", n_samples=4, n_candidates=3, shuffle=True)
    assert stats["written"] == 4 and stats["kept"] == 6
    rows = list(read_jsonl(tmp_path / "s1.jsonl"))
    assert all(len(r["passages"]) == 3 for r in rows)
    assert all(RankingSample.from_dict(r).n == 3 for r in rows)
    assert (tmp_path / "s1.stats.json").exists()

    e2 = [e2rank_row(chinese=False, source="msmarco") for _ in range(3)] + [e2rank_row() for _ in range(3)]
    write_jsonl(tmp_path / "e2.jsonl", e2)
    stats2 = build_stage2(tmp_path / "e2.jsonl", tmp_path / "s2.jsonl", n_samples=10)
    assert stats2["written"] == 3 and stats2["dropped_language"] == 3


# ---- BM25 + stage s03 ------------------------------------------------------
def test_bm25_ranks_matching_doc_first():
    corpus = {"d1": "apple pie recipe with cinnamon", "d2": "stock market news today",
              "d3": "how to bake an apple tart", "d4": "football scores"}
    bm = BM25().fit(list(corpus), list(corpus.values()))
    hits = bm.search("apple pie", k=3)
    assert hits[0][0] == "d1" and {d for d, _ in hits} == {"d1", "d3"}   # zero-score docs are dropped
    assert bm.search("zzzz unknown", k=3) == []
    assert len(bm.search("apple", k=1)) == 1


def test_run_bm25_returns_run():
    run = run_bm25({"a": "red car", "b": "blue boat"}, {"q1": "red", "q2": "boat"}, k=5)
    assert run["q1"][0][0] == "a" and run["q2"][0][0] == "b"


# ---- trainer ---------------------------------------------------------------
def test_trainer_runs_logs_and_saves(tmp_path):
    model = make_model()
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(8)]
    cfg = TrainConfig(epochs=2, micro_batch_size=2, grad_accum=2, lr=3e-3, log_every=1, save_every=2)
    before = next(model.encoder.parameters()).detach().clone()
    log = train(model, samples, cfg, tmp_path / "ck")
    assert len(log) == 4 and log[-1]["step"] == 4
    assert log[-1]["total"] < log[0]["total"]
    assert not torch.equal(before, next(model.encoder.parameters()).detach())
    assert (tmp_path / "ck/train_log.jsonl").exists()
    assert (tmp_path / "ck/latest/encoder/config.json").exists()
    assert (tmp_path / "ck/encoder/config.json").exists()
    assert len(list(read_jsonl(tmp_path / "ck/train_log.jsonl"))) == 4


def test_train_config_from_dict_ignores_other_keys():
    cfg = TrainConfig.from_dict({"lr": 1e-4, "train_file": "x", "out_name": "y", "init_from": None})
    assert cfg.lr == 1e-4


# ---- eval stages (tiny model) ----------------------------------------------
def test_encode_rerank_evaluate_roundtrip(tmp_path):
    model = make_model()
    corpus = {f"d{i}": f"document number {i} about subject {i % 3}" for i in range(10)}
    ids, emb = build_index(model, corpus, batch_size=4)
    assert ids == list(corpus) and emb.shape == (10, 32)
    # Row order must be restored after the length-sorted batching.
    single = build_index(model, corpus, doc_ids=["d7"], batch_size=1)[1][0]
    assert np.allclose(emb[7], single, atol=1e-4)

    queries = {"q1": "subject 1", "q2": "subject 2"}
    cands = {"q1": [(f"d{i}", 10 - i) for i in range(6)], "q2": [(f"d{i}", 10 - i) for i in range(2, 8)]}
    orig = rerank_run(model, queries, cands, ids, emb, "original", batch_size=2)
    rev = rerank_run(model, queries, cands, ids, emb, "reversed")
    rnd = rerank_run(model, queries, cands, ids, emb, "random", seed=3)
    for run in (orig, rev, rnd):
        assert set(run) == {"q1", "q2"}
        assert {d for d, _ in run["q1"]} == {f"d{i}" for i in range(6)}

    # Precomputed-embedding scores equal encoding the passages on the fly.
    with torch.no_grad():
        direct = model.rerank(["subject 1"], [[corpus[f"d{i}"] for i in range(6)]])["scores"][0]
    assert np.allclose([s for _, s in orig["q1"]], direct.numpy(), atol=1e-4)

    # Write runs and score them.
    write_trec_run(tmp_path / "a.run", orig)
    write_trec_run(tmp_path / "b.run", cands)
    (tmp_path / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td0\t1\nq2\td2\t1\n")
    rows = evaluate_files("toy", tmp_path / "qrels.tsv", {"resrank": tmp_path / "a.run", "bm25": tmp_path / "b.run"})
    assert {r["label"] for r in rows} == {"resrank", "bm25"}
    bm25_row = next(r for r in rows if r["label"] == "bm25")
    assert bm25_row["value"] == 1.0 and bm25_row["n_queries"] == 2
    write_results(tmp_path / "out.csv", rows)
    assert (tmp_path / "out.csv").read_text().startswith("dataset,label,metric,k,value,n_queries")


def test_rerank_run_reports_missing_index_entries():
    model = make_model()
    ids, emb = build_index(model, {"d0": "alpha", "d1": "beta"})
    with pytest.raises(KeyError):
        rerank_run(model, {"q": "x"}, {"q": [("d0", 1.0), ("zzz", 0.5)]}, ids, emb)


# ---- configs and scripts ---------------------------------------------------
def test_configs_load_and_map_to_model_config():
    cfg = load_config(ROOT / "configs/base.yaml")
    rr = resrank_config_from_cfg(cfg)
    assert (rr.tau_infonce, rr.tau_ranknet, rr.lam) == (0.05, 0.05, 0.1)
    assert rr.fusion_mode == "residual" and rr.passage_chunk_size == 16
    assert rr.pair_reduction == "mean" and rr.inject_mode == "raw"
    for name in ("stage1_small", "stage2_small"):
        t = TrainConfig.from_dict(load_config(ROOT / f"configs/{name}.yaml"))
        assert (t.micro_batch_size, t.grad_accum, t.init_loss_scale) == (1, 16, 1024.0)
        assert t.lr == pytest.approx(6e-6)


@pytest.mark.parametrize("script", sorted(p.name for p in (ROOT / "checks").glob("*.py")))
def test_check_scripts_compile(script):
    py_compile.compile(str(ROOT / "checks" / script), doraise=True)


@pytest.mark.parametrize("module", ["data/s01_download.py", "data/s02_build_train.py", "data/s03_bm25.py",
                                    "train/run_training.py", "eval/s06_encode_corpus.py",
                                    "eval/s07_rerank.py", "eval/s08_evaluate.py"])
def test_stage_scripts_compile(module):
    py_compile.compile(str(ROOT / "src/resrank" / module), doraise=True)


# ---- memory fallbacks ------------------------------------------------------
def test_freeze_embeddings_reduces_trainable_params_and_still_trains():
    from resrank.model.diagnostics import train_steps
    from resrank.model.loading import freeze_embeddings

    full = make_model()
    n_full = sum(p.numel() for p in full.parameters() if p.requires_grad)
    frozen_model = make_model()
    frozen = freeze_embeddings(frozen_model)
    n_frozen = sum(p.numel() for p in frozen_model.parameters() if p.requires_grad)
    assert frozen == 2 * 260 * 32 and n_full - n_frozen == frozen        # two tables: vocab 260 x hidden 32
    emb_before = frozen_model.reranker.model.get_input_embeddings().weight.detach().clone()
    out = train_steps(frozen_model, [make_sample(4, seed=1)], steps=3, lr=3e-3, optimizer="adamw")
    assert out["peak_gpu_gb"] is None and len(out["history"]) == 3
    assert out["trainable_params_m"] == round(n_frozen / 1e6, 1)
    assert torch.equal(emb_before, frozen_model.reranker.model.get_input_embeddings().weight.detach())


@pytest.mark.parametrize("opt", ["adamw", "sgd"])
def test_train_steps_optimizers(opt):
    from resrank.model.diagnostics import train_steps
    out = train_steps(make_model(), [make_sample(4, seed=2)], steps=2, lr=1e-3, optimizer=opt)
    assert len(out["history"]) == 2 and all(h == h for h in out["history"])
    assert out["skipped_steps"] == 0 and out["sec_per_sample"] > 0


def test_grad_finite_report_clean_on_healthy_model():
    from resrank.model.diagnostics import grad_finite_report
    model = make_model()
    samples = [make_sample(4, seed=1)]
    a = grad_finite_report(model, samples, "fp32", 1.0)
    b = grad_finite_report(model, samples, "fp32", 1024.0)
    assert a["encoder_bad_params"] == 0 and a["reranker_bad_params"] == 0
    assert a["encoder_grad_norm"] > 0 and a["reranker_grad_norm"] > 0
    # Reported gradient size is unscaled, so it does not depend on the loss scale.
    assert b["encoder_grad_norm"] == pytest.approx(a["encoder_grad_norm"], rel=1e-3)


def test_grad_finite_report_flags_nonfinite():
    from resrank.model.diagnostics import grad_finite_report
    model = make_model()
    with torch.no_grad():
        next(model.reranker.parameters()).fill_(float("inf"))
    r = grad_finite_report(model, [make_sample(4, seed=1)], "fp32", 1.0)
    assert r["encoder_bad_params"] + r["reranker_bad_params"] > 0 or r["loss"] != r["loss"]


def test_trainer_pushes_to_hub_when_configured(tmp_path, monkeypatch):
    from resrank import hub
    calls = []
    monkeypatch.setattr(hub, "push_folder", lambda folder, repo, path, **kw: calls.append((repo, path, kw.get("message"))))
    model = make_model()
    samples = [make_sample(4, seed=i, qid=f"q{i}") for i in range(4)]
    cfg = TrainConfig(epochs=1, micro_batch_size=2, grad_accum=1, lr=1e-3, save_every=1, hub_repo="me/ckpts")
    train(model, samples, cfg, tmp_path / "myrun")
    assert [c[1] for c in calls] == ["myrun/latest", "myrun/latest", "myrun"]   # 2 steps saved, then the final model
    assert all(c[0] == "me/ckpts" for c in calls)


def test_hub_failure_does_not_stop_training(tmp_path, monkeypatch):
    from resrank import hub

    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(hub, "push_folder", boom)
    model = make_model()
    cfg = TrainConfig(epochs=1, micro_batch_size=2, grad_accum=1, lr=1e-3, save_every=1, hub_repo="me/ckpts")
    log = train(model, [make_sample(4, seed=i) for i in range(2)], cfg, tmp_path / "run")
    assert len(log) == 1 and (tmp_path / "run/encoder/config.json").exists()


def test_no_hub_push_without_repo(tmp_path, monkeypatch):
    from resrank import hub
    monkeypatch.setattr(hub, "push_folder", lambda *a, **k: pytest.fail("should not push"))
    train(make_model(), [make_sample(4, seed=i) for i in range(2)],
          TrainConfig(epochs=1, micro_batch_size=2, grad_accum=1, lr=1e-3, save_every=1), tmp_path / "run")
