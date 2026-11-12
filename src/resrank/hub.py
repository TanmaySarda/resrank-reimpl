"""Back up and restore checkpoints on the Hugging Face Hub (Kaggle's /kaggle/working is wiped when the
session ends).

    python -m resrank.hub push --ckpt stage1 --repo your-name/resrank-ckpts
    python -m resrank.hub pull --ckpt stage1 --repo your-name/resrank-ckpts

Needs a Hugging Face token with write access in the HF_TOKEN environment variable. In Kaggle, add it under
Add-ons > Secrets, then in a notebook cell:
    import os; from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
Repos are created private. Each checkpoint is about 4.8 GB (both 0.6B models in fp32).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from resrank.paths import Paths


def push_folder(local_dir: str | Path, repo_id: str, path_in_repo: str, *,
                token: Optional[str] = None, private: bool = True,
                ignore_patterns: Optional[list[str]] = None, message: str = "resrank checkpoint") -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(local_dir), repo_id=repo_id, repo_type="model",
                      path_in_repo=path_in_repo, ignore_patterns=ignore_patterns, commit_message=message)


def pull_folder(repo_id: str, path_in_repo: str, local_root: str | Path, *,
                token: Optional[str] = None) -> Path:
    """Download `path_in_repo` from the repo into local_root/path_in_repo and return that folder."""
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(local_root),
                      allow_patterns=[f"{path_in_repo}/*"], token=token or os.environ.get("HF_TOKEN"))
    return Path(local_root) / path_in_repo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["push", "pull"])
    ap.add_argument("--ckpt", required=True, help="checkpoint name under checkpoints/ (e.g. stage1 or stage1/latest)")
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. your-name/resrank-ckpts")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    paths = Paths.from_env(a.root).ensure()
    if a.action == "push":
        src = paths.checkpoints / a.ckpt
        push_folder(src, a.repo, a.ckpt, ignore_patterns=["latest/*"] if "latest" not in a.ckpt else None)
        print(f"pushed {src} -> {a.repo}/{a.ckpt}")
    else:
        print("pulled to", pull_folder(a.repo, a.ckpt, paths.checkpoints))


if __name__ == "__main__":
    main()
