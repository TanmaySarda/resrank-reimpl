"""Turn a GPU-hour budget and a measured speed into a number of training samples.

    python -m resrank.train.plan --sec-per-sample 1.45 --hours-left 27
"""
from __future__ import annotations

import argparse
import json


def plan_samples(sec_per_sample: float, hours_left: float, *, reserve_hours: float = 4.7,
                 snapshot_minutes: float = 15.0, upload_seconds: float = 80.0, safety: float = 1.10,
                 per_step: int = 16, max_samples: int = 232_000, round_to: int = 1000) -> dict:
    """How many samples fit in the budget.

    reserve_hours covers everything that is not training: evaluation (about 1 h), setup and timing run
    (about 0.7 h) and a buffer for restarts (3 h; each restart re-installs, rebuilds data and reloads the model).
    Every snapshot interval also pays for one upload, and `safety` pads the measured speed by 10%.
    """
    if sec_per_sample <= 0 or hours_left <= reserve_hours:
        raise ValueError("need a positive speed and more hours than the reserve")
    effective = sec_per_sample * (1 + upload_seconds / (snapshot_minutes * 60)) * safety
    train_hours = hours_left - reserve_hours
    n = int(train_hours * 3600 / effective) // round_to * round_to
    n = max(per_step, min(n, max_samples))
    return {
        "n_samples": n,
        "optimizer_steps": n // per_step,
        "effective_sec_per_sample": round(effective, 3),
        "expected_training_hours": round(n * effective / 3600, 1),
        "expected_hours_without_safety_margin": round(n * effective / safety / 3600, 1),
        "budget_hours_left": hours_left,
        "reserved_hours": reserve_hours,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sec-per-sample", type=float, required=True)
    ap.add_argument("--hours-left", type=float, required=True)
    ap.add_argument("--reserve-hours", type=float, default=4.7)
    a = ap.parse_args()
    print(json.dumps(plan_samples(a.sec_per_sample, a.hours_left, reserve_hours=a.reserve_hours), indent=2))


if __name__ == "__main__":
    main()
