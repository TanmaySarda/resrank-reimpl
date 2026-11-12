import json

import numpy as np
import pytest

from resrank.data.e2rank import iter_e2rank_samples, reservoir_sample, source_report
from resrank.data.filters import cjk_ratio, contains_cjk, is_english_sample
from resrank.io_utils import (read_json, read_jsonl, read_qrels, read_trec_run, load_npy,
                              save_npy, write_json, write_jsonl, write_trec_run)
from resrank.paths import Paths
from resrank.schemas import RankingSample, SchemaError, ranking_to_ranks


# ---- schemas ---------------------------------------------------------------
def e2rank_row(n=16, ranking=None, pos_index=14, source="cMedQAv2", chinese=True):
    """Mirrors the pasted sample row: 16 passages, 1-indexed ranking and pos_index."""
    text = (lambda i: f"这是第{i}个段落") if chinese else (lambda i: f"This is passage number {i}")
    ranking = ranking or [14, 1, 7, 13, 4, 6, 10, 5, 15, 11, 12, 2, 3, 8, 9, 16]
    return {
        "query": "腰椎问题怎么办" if chinese else "what is an LLM",
        "document": [text(i) for i in range(1, n + 1)],
        "ranking": ranking,
        "pos_index": pos_index,
        "model_output": " > ".join(f"[{r}]" for r in ranking),
        "source": source,
    }


def test_from_e2rank_row_converts_to_zero_indexed():
    s = RankingSample.from_e2rank_row(e2rank_row())
    assert s.n == 16
    assert s.pos_index == 13
    assert s.ranking[0] == 13 and s.ranking[-1] == 15
    assert s.ranking[0] == s.pos_index          # in the pasted sample the labeler agrees with pos_index
    assert s.ranks[13] == 0 and s.ranks[15] == 15


def test_ranking_to_ranks():
    assert ranking_to_ranks([2, 0, 1]) == [1, 2, 0]
    with pytest.raises(SchemaError):
        ranking_to_ranks([0, 0, 1])


@pytest.mark.parametrize("bad", [
    dict(ranking=[1, 2, 3]),                       # wrong length
    dict(pos_index=17),                            # out of range
    dict(pos_index=0),                             # 1-indexed, so 0 is invalid
    dict(ranking=[1] * 16),                        # not a permutation
])
def test_invalid_rows_rejected(bad):
    row = e2rank_row()
    row.update(bad)
    with pytest.raises(SchemaError):
        RankingSample.from_e2rank_row(row)


def test_missing_key_rejected():
    row = e2rank_row()
    del row["ranking"]
    with pytest.raises(SchemaError):
        RankingSample.from_e2rank_row(row)


def test_sample_dict_roundtrip():
    s = RankingSample.from_e2rank_row(e2rank_row(), qid="q1")
    assert RankingSample.from_dict(s.to_dict()) == s


# ---- filters ---------------------------------------------------------------
def test_cjk_detection():
    assert contains_cjk("hello 世界") and not contains_cjk("hello world")
    assert cjk_ratio("") == 0.0
    assert cjk_ratio("ab 世界") == pytest.approx(0.5)


def test_is_english_sample():
    en = RankingSample.from_e2rank_row(e2rank_row(chinese=False, source="msmarco"))
    zh = RankingSample.from_e2rank_row(e2rank_row())
    assert is_english_sample(en) and not is_english_sample(zh)
    # One Chinese passage in an otherwise English list rejects it under the strict default...
    mixed = RankingSample.from_e2rank_row(e2rank_row(chinese=False))
    mixed.passages[3] = "混合语言"
    assert not is_english_sample(mixed)
    # ...but is tolerated if the allowed ratio is high enough.
    assert is_english_sample(mixed, max_ratio=1.0)


# ---- e2rank loader ---------------------------------------------------------
@pytest.fixture
def e2rank_file(tmp_path):
    rows = [
        e2rank_row(chinese=True, source="cMedQAv2"),
        e2rank_row(chinese=False, source="hotpotqa"),
        e2rank_row(chinese=False, source="msmarco"),
        e2rank_row(chinese=False, source="msmarco", ranking=[1, 1, 1]),  # invalid
    ]
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    return path


def test_loader_filters_language_and_invalid(e2rank_file):
    stats = {}
    kept = list(iter_e2rank_samples(e2rank_file, stats=stats))
    assert [s.source for s in kept] == ["hotpotqa", "msmarco"]
    assert stats == {"read": 4, "invalid": 1, "dropped_source": 0, "dropped_language": 1, "kept": 2}


def test_loader_source_filters_and_limit(e2rank_file):
    only = list(iter_e2rank_samples(e2rank_file, allow_sources=["msmarco"]))
    assert [s.source for s in only] == ["msmarco"]
    denied = list(iter_e2rank_samples(e2rank_file, deny_sources=["msmarco"], english_only=False))
    assert [s.source for s in denied] == ["cMedQAv2", "hotpotqa"]
    assert len(list(iter_e2rank_samples(e2rank_file, limit=1))) == 1


def test_source_report(e2rank_file):
    rep = source_report(e2rank_file)
    assert rep["cMedQAv2"]["frac_samples_with_cjk"] == 1.0
    assert rep["hotpotqa"]["frac_samples_with_cjk"] == 0.0
    assert rep["msmarco"]["count"] == 2


def test_reservoir_sample_is_deterministic_and_sized():
    a = reservoir_sample(range(1000), 10, seed=1)
    assert a == reservoir_sample(range(1000), 10, seed=1)
    assert len(a) == 10 and len(set(a)) == 10
    assert sorted(reservoir_sample(range(5), 10)) == list(range(5))


# ---- io_utils ---------------------------------------------------------------
def test_jsonl_and_json_roundtrip(tmp_path):
    rows = [{"a": 1, "t": "世界"}, {"a": 2, "t": "x"}]
    assert write_jsonl(tmp_path / "x.jsonl", rows) == 2
    assert list(read_jsonl(tmp_path / "x.jsonl")) == rows
    assert list(read_jsonl(tmp_path / "x.jsonl", limit=1)) == rows[:1]
    write_json(tmp_path / "y.json", {"k": [1, 2]})
    assert read_json(tmp_path / "y.json") == {"k": [1, 2]}


def test_atomic_write_leaves_no_temp_files_and_rejects_gz(tmp_path):
    write_jsonl(tmp_path / "x.jsonl", [{"a": 1}])
    assert [p.name for p in tmp_path.iterdir()] == ["x.jsonl"]
    with pytest.raises(ValueError):
        write_jsonl(tmp_path / "x.jsonl.gz", [{"a": 1}])


def test_npy_roundtrip(tmp_path):
    arr = np.random.randn(5, 3).astype("float32")
    save_npy(tmp_path / "e.npy", arr)
    assert np.array_equal(load_npy(tmp_path / "e.npy"), arr)


def test_trec_run_roundtrip_assigns_ranks_by_score(tmp_path):
    run = {"q1": [("d2", 0.3), ("d1", 0.9), ("d3", 0.5)], "q2": [("d9", 1.0)]}
    write_trec_run(tmp_path / "r.run", run, tag="t")
    back = read_trec_run(tmp_path / "r.run")
    assert [d for d, _ in back["q1"]] == ["d1", "d3", "d2"]
    assert back["q2"] == [("d9", 1.0)]


def test_read_qrels_skips_header(tmp_path):
    p = tmp_path / "q.tsv"
    p.write_text("query-id\tcorpus-id\tscore\nq1\t0\td1\t2\nq1\t0\td2\t0\nq2\t0\td3\t1\n")
    assert read_qrels(p) == {"q1": {"d1": 2, "d2": 0}, "q2": {"d3": 1}}


def test_read_qrels_beir_three_column_format(tmp_path):
    p = tmp_path / "beir.tsv"
    p.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td2\t0\nq2\td3\t2\n")
    assert read_qrels(p) == {"q1": {"d1": 1, "d2": 0}, "q2": {"d3": 2}}


# ---- paths ----------------------------------------------------------------
def test_paths_layout_and_ensure(tmp_path):
    p = Paths.from_env(tmp_path)
    assert p.train_file("stage2_e2rank_en") == tmp_path / "data/02_train/stage2_e2rank_en.jsonl"
    assert p.candidate_run("scifact") == tmp_path / "data/03_candidates/scifact.bm25.run"
    assert p.run_file("scifact", "a", "reversed") == tmp_path / "data/05_runs/scifact.a.reversed.run"
    assert p.encoder_ckpt("stage2") == tmp_path / "checkpoints/stage2/encoder"
    p.ensure()
    for d in ("01_raw", "02_train", "03_candidates", "04_index", "05_runs", "06_results", "07_debug"):
        assert (tmp_path / "data" / d).is_dir()
    assert (tmp_path / "checkpoints").is_dir()


def test_paths_root_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("RESRANK_ROOT", str(tmp_path / "elsewhere"))
    assert Paths.from_env().root == tmp_path / "elsewhere"
