"""Push checkpoints out of /kaggle/working before the session disappears, as a Kaggle dataset.

Auth: kagglehub reads the token from the KAGGLE_API_TOKEN environment variable (or ~/.kaggle/access_token).
The token is never stored by this code. In a Kaggle notebook, add it as a Secret (Add-ons > Secrets) and run
this in a Python cell before the bash cells:

    import os
    from kaggle_secrets import UserSecretsClient
    os.environ["KAGGLE_API_TOKEN"] = UserSecretsClient().get_secret("KAGGLE_API_TOKEN")

Each upload creates a NEW VERSION of the dataset containing exactly what is staged, so include every
checkpoint you want to keep (e.g. stage1 and stage2). To resume in a new session, attach the dataset as an
input and copy it back:  cp -r /kaggle/input/<dataset>/* /kaggle/working/checkpoints/

CLI:  python -m resrank.train.publish --dataset <user>/<name> --names stage1 stage2_e2rank
"""
from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

from resrank.util import say

UploadFn = Callable[..., None]


def check_auth() -> None:
    """Raise a helpful error if no Kaggle credentials can be found."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    home = Path(os.path.expanduser("~")) / ".kaggle"
    if (home / "access_token").exists() or (home / "kaggle.json").exists():
        return
    raise RuntimeError(
        "No Kaggle credentials found. Set the KAGGLE_API_TOKEN environment variable (in a Kaggle notebook, "
        "load it from a Secret; see the docstring of resrank/train/publish.py). Never paste the token into code."
    )


def _superseded_snapshot(ckpt_root: Path, rel: Path) -> bool:
    """True if `rel` lies inside a `latest/` snapshot whose checkpoint folder also has final weights."""
    parts = rel.parts[:-1]
    for i in range(1, len(parts)):
        if parts[i] == "latest" and (ckpt_root.joinpath(*parts[:i]) / "encoder").is_dir():
            return True
    return False


def stage_checkpoints(ckpt_root: str | Path, stage_dir: str | Path,
                      names: Optional[Iterable[str]] = None, include_latest: bool = False) -> list[Path]:
    """Mirror the chosen checkpoint folders into `stage_dir` using hard links (no extra disk space).

    Falls back to copying if linking is not possible. `latest/` subfolders (the periodic mid-training
    snapshots) are skipped unless include_latest is set, but only when their checkpoint folder also holds final
    weights: for a run that was stopped before it finished, `latest/` is the only copy of the weights and is
    always kept. Returns the staged files.
    """
    ckpt_root, stage_dir = Path(ckpt_root), Path(stage_dir)
    wanted = list(names) if names is not None else sorted(p.name for p in ckpt_root.iterdir() if p.is_dir())
    missing = [n for n in wanted if not (ckpt_root / n).is_dir()]
    if missing:
        raise FileNotFoundError(f"checkpoint folder(s) not found under {ckpt_root}: {missing}")
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    staged: list[Path] = []
    for name in wanted:
        for src in sorted((ckpt_root / name).rglob("*")):
            if not src.is_file():
                continue
            rel = src.relative_to(ckpt_root)
            if any(part.endswith((".tmp", ".old")) for part in rel.parts):
                continue                      # half-written or superseded snapshot
            if not include_latest and _superseded_snapshot(ckpt_root, rel):
                continue
            dst = stage_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            staged.append(dst)
    if not staged:
        raise FileNotFoundError(f"nothing to upload under {ckpt_root} for {wanted}")
    return staged


def _default_uploader() -> UploadFn:
    """kagglehub's uploader with its very chatty per-file output captured, so it does not bury the training
    progress bar. If the upload fails, the tail of the captured output is included in the error."""
    import kagglehub

    def upload(handle: str, folder: str, version_notes: str = "") -> None:
        buf = io.StringIO()
        logging.disable(logging.INFO)        # kagglehub reports every file through logger.info (own handler)
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                kagglehub.dataset_upload(handle, folder, version_notes=version_notes)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{type(e).__name__}: {e}\n--- end of upload log ---\n{buf.getvalue()[-800:]}") from e
        finally:
            logging.disable(logging.NOTSET)

    return upload


def upload_checkpoints(ckpt_root: str | Path, handle: str, names: Optional[Iterable[str]] = None,
                       include_latest: bool = False, version_notes: str = "",
                       uploader: Optional[UploadFn] = None, stage_dir: Optional[str | Path] = None) -> int:
    """Upload checkpoints as a new version of the Kaggle dataset `handle` ("<user>/<dataset>").
    Returns the number of files uploaded."""
    if "/" not in handle:
        raise ValueError("handle must look like '<kaggle-username>/<dataset-name>'")
    if uploader is None:
        check_auth()
        uploader = _default_uploader()
    stage = Path(stage_dir) if stage_dir else Path(ckpt_root).parent / "_upload_stage"
    files = stage_checkpoints(ckpt_root, stage, names, include_latest)
    try:
        uploader(handle, str(stage), version_notes=version_notes)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return len(files)


def make_save_callback(ckpt_root: str | Path, handle: str, names: Iterable[str],
                       every_saves: int = 0, uploader: Optional[UploadFn] = None):
    """Build the `on_save(out_dir, final)` hook for the trainer.

    every_saves = N uploads after every Nth periodic save (these include the `latest` snapshot);
    0 uploads only when training finishes. Upload errors are printed, never raised, so a network problem
    cannot kill a long training run.
    """
    names = list(names)
    state = {"saves": 0, "warned": False}

    def on_save(out_dir: Path, final: bool) -> None:
        if not final:
            state["saves"] += 1
            if not every_saves or state["saves"] % every_saves:
                return
        present = [n for n in names if (Path(ckpt_root) / n).is_dir()]
        if not state["warned"] and len(present) < len(names):
            state["warned"] = True
            say(f"[publish] note: not uploading folders that are missing from {ckpt_root}: "
                f"{[n for n in names if n not in present]}. Restore them first if you want to keep them "
                f"in the dataset's latest version.")
        if not present:
            say("[publish] WARNING: nothing to upload; training continues")
            return
        t0 = time.time()
        try:
            n = upload_checkpoints(ckpt_root, handle, present, include_latest=not final,
                                   version_notes=f"{'final' if final else 'snapshot'}: {Path(out_dir).name}",
                                   uploader=uploader)
            say(f"[publish] uploaded {n} files to {handle} in {time.time() - t0:.0f}s")
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            say(f"[publish] WARNING: upload failed ({type(e).__name__}: {e}); training continues")

    return on_save


def main() -> None:
    from resrank.paths import Paths

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="<kaggle-username>/<dataset-name>")
    ap.add_argument("--names", nargs="*", help="checkpoint folders to include (default: all)")
    ap.add_argument("--include-latest", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    paths = Paths.from_env(a.root)
    n = upload_checkpoints(paths.checkpoints, a.dataset, a.names, a.include_latest, a.notes)
    print(f"uploaded {n} files to {a.dataset}")


if __name__ == "__main__":
    main()
