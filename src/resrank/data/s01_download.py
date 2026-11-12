"""Stage s01: download raw datasets from the Hugging Face Hub.

OUT: data/01_raw/<name>/<file>

The file names below are what the dataset pages/READMEs indicate but have NOT been verified;
run with --list first to see what each repo really contains.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from resrank.paths import Paths

DATASETS = {
    "pe_rank": ("liuqi6777/pe_rank_data", ["train.jsonl"]),
    "e2rank": ("Alibaba-NLP/E2Rank_ranking_datasets", ["train.jsonl"]),
}


def list_files(repo_id: str) -> list[str]:
    from huggingface_hub import list_repo_files
    return sorted(list_repo_files(repo_id, repo_type="dataset"))


def download_file(repo_id: str, filename: str, out_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    out_dir.mkdir(parents=True, exist_ok=True)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset",
                                local_dir=str(out_dir)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("name", choices=sorted(DATASETS))
    ap.add_argument("--list", action="store_true", help="only list the repo's files")
    ap.add_argument("--files", nargs="*", help="override the files to download")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    repo_id, default_files = DATASETS[a.name]
    if a.list:
        print("\n".join(list_files(repo_id)))
        return
    out_dir = Paths.from_env(a.root).ensure().raw / a.name
    for f in a.files or default_files:
        print("downloaded", download_file(repo_id, f, out_dir))


if __name__ == "__main__":
    main()
