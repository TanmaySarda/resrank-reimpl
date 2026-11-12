import zipfile

import pytest

from resrank.data.beir import read_beir_corpus, read_beir_queries
from resrank.data.s01_download_beir import download_beir, safe_extract
from resrank.data.s03_bm25 import restrict_queries, run_bm25
from resrank.io_utils import read_qrels
from resrank.paths import Paths


def make_beir_zip(path, name="toy"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{name}/corpus.jsonl", '{"_id": "d1", "title": "Apples", "text": "red fruit"}\n{"_id": "d2", "title": "", "text": "boats float"}\n')
        z.writestr(f"{name}/queries.jsonl", '{"_id": "q1", "text": "apples", "metadata": {}}\n{"_id": "q2", "text": "boats", "metadata": {}}\n')
        z.writestr(f"{name}/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq1\td1\t1\n")


def test_download_from_local_zip_and_use_the_files(tmp_path):
    make_beir_zip(tmp_path / "toy.zip")
    d = download_beir("toy", tmp_path / "raw", zip_path=tmp_path / "toy.zip")
    corpus = read_beir_corpus(d / "corpus.jsonl")
    assert corpus == {"d1": "Apples red fruit", "d2": "boats float"}
    queries = read_beir_queries(d / "queries.jsonl")
    qrels = read_qrels(d / "qrels" / "test.tsv")
    assert qrels == {"q1": {"d1": 1}}
    only_test = restrict_queries(queries, qrels)
    assert list(only_test) == ["q1"]
    run = run_bm25(corpus, only_test, k=5)
    assert run["q1"][0][0] == "d1"


def test_unexpected_layout_is_reported(tmp_path):
    with zipfile.ZipFile(tmp_path / "bad.zip", "w") as z:
        z.writestr("other/readme.txt", "x")
    with pytest.raises(FileNotFoundError, match="unexpected zip layout"):
        download_beir("toy", tmp_path / "raw", zip_path=tmp_path / "bad.zip")


def test_zip_path_traversal_is_rejected(tmp_path):
    with zipfile.ZipFile(tmp_path / "evil.zip", "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract(tmp_path / "evil.zip", tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_index_folders_are_separate_per_checkpoint(tmp_path):
    p = Paths.from_env(tmp_path)
    assert p.embeddings_file("scifact", "stage1") != p.embeddings_file("scifact", "pretrained")
    assert p.embeddings_file("scifact", "stage1") == tmp_path / "data/04_index/scifact/stage1/embeddings.npy"
    assert p.index_ids_file("scifact") == tmp_path / "data/04_index/scifact/ids.json"   # old behaviour kept
