"""Summarize the raw E2Rank file: sources (and how Chinese each is), list lengths, passage lengths, and how
often the labeler's top passage is `pos_index`,
and which sources look English (script filter pass rate and share of common English words).

    python checks/inspect_e2rank.py path/to/train.jsonl --limit 20000
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resrank.data.e2rank import e2rank_stats, source_report, suggest_english_sources

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("path")
ap.add_argument("--limit", type=int, default=None)
args = ap.parse_args()
print(json.dumps(e2rank_stats(args.path, args.limit), indent=2))
report = source_report(args.path, args.limit)
print("\nsource                           count  with_cjk  q_cjk  pass_script  english_words")
for source, r in report.items():
    print(f"{source:30s} {r['count']:7d} {r['frac_samples_with_cjk']:8.3f} {r['mean_query_cjk_ratio']:6.3f} "
          f"{r['frac_pass_script_filter']:11.3f} {r['english_word_rate']:13.3f}")
print("\nlooks English (check by eye):", " ".join(suggest_english_sources(report)))
