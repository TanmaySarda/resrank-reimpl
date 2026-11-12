"""Print what the first rows of the raw PE-Rank file look like and how the pieces relate.

    python checks/inspect_pe_rank.py path/to/train.jsonl --n 3
Answers: do the user-message passages equal some slice of extra_texts, does the assistant message match
`ranking`, is `ranking` a permutation, and what does the last passage's tail look like (leftover prompt text?).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resrank.data.pe_rank import inspect_pe_rank_row
from resrank.io_utils import read_jsonl

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("path")
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()
for i, row in enumerate(read_jsonl(args.path, limit=args.n)):
    print(f"--- row {i} ---")
    print(json.dumps(inspect_pe_rank_row(row), indent=2, ensure_ascii=False))
