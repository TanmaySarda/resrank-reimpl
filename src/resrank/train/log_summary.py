"""Summarize a training log (train_log.jsonl): did the loss fall, were steps skipped, how fast was it.

    python -m resrank.train.log_summary /kaggle/working/checkpoints/stage1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def _session_ends(records: list[dict]) -> list[float]:
    """elapsed_s restarts from zero in every resumed session; return the last elapsed_s of each session."""
    ends: list[float] = []
    prev = None
    for r in records:
        e = r.get("elapsed_s")
        if e is None:
            continue
        if prev is not None and e < prev:
            ends.append(prev)
        prev = e
    if prev is not None:
        ends.append(prev)
    return ends


def _total_elapsed(records: list[dict]) -> float | None:
    """Training time summed over all sessions (after a resume, the last record only covers the last session)."""
    ends = _session_ends(records)
    return round(sum(ends), 1) if ends else None


def summarize(ckpt_dir: str | Path, window: int = 10) -> dict:
    ckpt_dir = Path(ckpt_dir)
    records = [json.loads(l) for l in (ckpt_dir / "train_log.jsonl").read_text().splitlines() if l.strip()]
    ok = [r for r in records if not r.get("oom_skipped")]
    out: dict = {"steps_logged": len(records), "oom_skipped_steps": len(records) - len(ok)}
    if not ok:
        return out
    w = min(window, max(1, len(ok) // 2))
    for key in ("total", "infonce", "ranknet"):
        out[f"{key}_first{w}_mean"] = round(mean(r[key] for r in ok[:w]), 4)
        out[f"{key}_last{w}_mean"] = round(mean(r[key] for r in ok[-w:]), 4)
    out["grad_norm_min_max"] = [round(min(r["grad_norm"] for r in ok), 2), round(max(r["grad_norm"] for r in ok), 2)]
    scales = [r["loss_scale"] for r in ok if r.get("loss_scale") is not None]
    if scales:
        out["loss_scale_min_max"] = [min(scales), max(scales)]
    out["sessions"] = len(_session_ends(records))
    out["elapsed_s_total"] = _total_elapsed(records)
    state_file = ckpt_dir / "train_state.json"
    if state_file.exists() and out["elapsed_s_total"]:
        st = json.loads(state_file.read_text())
        samples = (records[-1]["step"] - records[0]["step"] + 1) * st["micro_batch_size"] * st["grad_accum"]
        out["sec_per_sample_incl_warmup"] = round(out["elapsed_s_total"] / samples, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_dir")
    ap.add_argument("--window", type=int, default=10)
    a = ap.parse_args()
    print(json.dumps(summarize(a.ckpt_dir, a.window), indent=2))


if __name__ == "__main__":
    main()
