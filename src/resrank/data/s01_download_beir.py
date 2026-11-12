"""Stage s01 (evaluation data): download and unpack a BEIR dataset.

OUT: data/01_raw/beir/<name>/corpus.jsonl, queries.jsonl, qrels/test.tsv (and train/dev where they exist)

The URL is the standard BEIR distribution location and has NOT been verified from the build environment;
if the download fails, fetch the zip another way and pass it with --zip.
"""
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

from resrank.paths import Paths

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def safe_extract(zip_path: str | Path, out_dir: str | Path) -> list[Path]:
    """Extract a zip, refusing any member that would land outside `out_dir`."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = (out_dir / member.filename).resolve()
            if out_dir != target and out_dir not in target.parents:
                raise ValueError(f"unsafe path in zip: {member.filename}")
        z.extractall(out_dir)
        written = [out_dir / m.filename for m in z.infolist() if not m.is_dir()]
    return written


def download_beir(name: str, out_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if zip_path is None:
        zip_path = out_dir / f"{name}.zip"
        urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    safe_extract(zip_path, out_dir)
    dataset_dir = out_dir / name
    for needed in ("corpus.jsonl", "queries.jsonl"):
        if not (dataset_dir / needed).exists():
            raise FileNotFoundError(f"{dataset_dir / needed} missing after extraction; unexpected zip layout")
    return dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="e.g. scifact, nfcorpus")
    ap.add_argument("--zip", default=None, help="use an already-downloaded zip instead of downloading")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    out = Paths.from_env(a.root).ensure().raw / "beir"
    d = download_beir(a.name, out, a.zip)
    print("ready:", d)
    for p in sorted(d.rglob("*")):
        if p.is_file():
            print(" ", p.relative_to(d), p.stat().st_size)


if __name__ == "__main__":
    main()
