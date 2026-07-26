from pathlib import Path
from unittest.mock import patch

from report_robust_strategy_closure import build_closure_report, render_markdown


def test_closure_report_marks_old_run_as_failed_baseline(tmp_path: Path) -> None:
    run_dir = tmp_path / "old-run"
    development = run_dir / "development"
    development.mkdir(parents=True)
    (development / "progress.json").write_text(
        '{"jobs":{"job":{"status":"completed","model_id":"linear_ols",'
        '"window_days":756,"quarter":"2023Q1","seed":null,'
        '"prediction_path":"prediction.parquet"}}}',
        encoding="utf-8",
    )
    aggregated = {
        "model_reports": [
            {
                "model_id": "linear_ols",
                "eligible": False,
                "gates": {"window_rank_correlation": True, "top50_overlap": False},
                "failed_gates": ["top50_overlap"],
                "positive_rank_ic_interval_ratio": 0.75,
                "rank_ic_mean": 0.01,
                "stability": {
                    "rank_correlation_median": 0.83,
                    "topk_overlap_mean": 0.56,
                },
                "seed_stability": {},
            }
        ]
    }
    config = {"portfolio": {"topk": 50}, "acceptance": {}}

    with patch(
        "report_robust_strategy_closure.aggregate_training_artifacts",
        return_value=aggregated,
    ):
        payload = build_closure_report(
            run_dir=run_dir, config=config, generated_at="2026-07-23T21:30:00+08:00"
        )

    assert payload["status"] == "development_failed"
    assert payload["qualification"] == "baseline_only"
    assert payload["fixed_parameter_job_count"] == 1
    assert payload["fixed_parameter_models"][0]["failed_gates"] == [
        "top50_overlap"
    ]
    assert "第零代失败基线" in render_markdown(payload)
