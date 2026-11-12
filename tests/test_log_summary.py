import json

from resrank.train.log_summary import summarize


def test_summary_reports_trend_skips_and_speed(tmp_path):
    recs = [{"step": i + 1, "total": 1.0 - 0.01 * i, "infonce": 2.0 - 0.01 * i, "ranknet": 1.0 - 0.01 * i,
             "grad_norm": 50.0 + i, "loss_scale": 1024, "elapsed_s": 20.0 * (i + 1)} for i in range(30)]
    recs.insert(5, {"step": 99, "oom_skipped": True, "elapsed_s": 1.0})
    (tmp_path / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (tmp_path / "train_state.json").write_text(json.dumps({"micro_batch_size": 1, "grad_accum": 16}))
    s = summarize(tmp_path, window=5)
    assert s["oom_skipped_steps"] == 1 and s["steps_logged"] == 31
    assert s["total_first5_mean"] > s["total_last5_mean"]
    assert s["loss_scale_min_max"] == [1024, 1024]
