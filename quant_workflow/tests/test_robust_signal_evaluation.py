import pytest
import pandas as pd

from robust_signal_evaluation import (
    acceptance_verdict,
    feature_diagnostics,
    horizon_aligned_excess,
    market_state_attribution,
    topk_sensitivity,
)


def _prediction_frame() -> pd.DataFrame:
    rows = []
    for date in pd.to_datetime(["2025-01-02", "2025-01-03"]):
        for number in range(1, 101):
            rows.append(
                {
                    "datetime": date,
                    "instrument": f"S{number:03d}",
                    "score": float(number),
                    "label": float(number) / 1000.0,
                }
            )
    return pd.DataFrame(rows).set_index(["datetime", "instrument"])


def test_feature_diagnostics_reports_missing_constant_and_correlated_columns() -> None:
    frame = pd.DataFrame(
        {
            "missing": [1.0, None, None],
            "constant": [2.0, 2.0, 2.0],
            "a": [1.0, 2.0, 3.0],
            "b": [2.0, 4.0, 6.0],
        }
    )

    report = feature_diagnostics(frame, correlation_threshold=0.99)

    assert report["missing_rate_by_feature"]["missing"] == 2 / 3
    assert "constant" in report["constant_features"]
    assert any(pair["left"] == "a" and pair["right"] == "b" for pair in report["high_correlation_pairs"])


def test_topk_sensitivity_keeps_top50_as_primary() -> None:
    report = topk_sensitivity(_prediction_frame(), topks=[20, 30, 50, 100], primary=50)

    assert report["primary_topk"] == 50
    assert set(report["results"]) == {"20", "30", "50", "100"}
    assert report["results"]["50"]["selected_rows"] == 100


def test_acceptance_verdict_requires_every_registered_gate() -> None:
    values = {
        "positive_rank_ic_quarter_ratio": 0.75,
        "median_quarter_rank_ic": 0.03,
        "window_rank_correlation_median": 0.8,
        "top50_overlap_mean": 0.59,
        "positive_excess_quarter_ratio": 0.75,
        "annual_excess_return": 0.08,
        "monotonic_quarter_ratio": 0.75,
    }
    thresholds = {
        "positive_rank_ic_quarter_ratio_min": 0.70,
        "median_quarter_rank_ic_min": 0.02,
        "window_rank_correlation_median_min": 0.70,
        "top50_overlap_mean_min": 0.60,
        "positive_excess_quarter_ratio_min": 0.70,
        "annual_excess_return_min": 0.05,
        "monotonic_quarter_ratio_min": 0.70,
    }

    verdict = acceptance_verdict(values, thresholds)

    assert verdict["passed"] is False
    assert verdict["failed_gates"] == ["top50_overlap"]


def test_raw_return_excess_uses_stored_csi800_benchmark_not_universe_mean() -> None:
    frame = pd.DataFrame(
        [
            {
                "datetime": "2025-01-02",
                "instrument": "HIGH",
                "score": 2.0,
                "label": 0.10,
                "benchmark_return": 0.02,
            },
            {
                "datetime": "2025-01-02",
                "instrument": "LOW",
                "score": 1.0,
                "label": -0.10,
                "benchmark_return": 0.02,
            },
        ]
    ).set_index(["datetime", "instrument"])
    frame.index = frame.index.set_levels(
        pd.to_datetime(frame.index.levels[0]), level="datetime"
    )

    report = horizon_aligned_excess(
        frame,
        topk=1,
        horizon_days=1,
        total_cost_bps=0,
        labels_are_excess=False,
    )

    assert report["period_count"] == 1
    assert report["cumulative_excess_return"] == pytest.approx(0.08)


def test_raw_return_excess_fails_closed_without_stored_benchmark() -> None:
    with pytest.raises(ValueError, match="benchmark_return"):
        horizon_aligned_excess(
            _prediction_frame(),
            topk=50,
            horizon_days=5,
            total_cost_bps=20,
            labels_are_excess=False,
        )


def test_market_state_attribution_reports_all_four_states() -> None:
    rows = []
    states = ["risk_on", "neutral", "defensive", "risk_off"]
    for date, state in zip(pd.bdate_range("2025-01-02", periods=4), states):
        rows.extend(
            [
                {
                    "datetime": date,
                    "instrument": "HIGH",
                    "score": 2.0,
                    "label": 0.03,
                    "benchmark_return": 0.01,
                    "market_state": state,
                },
                {
                    "datetime": date,
                    "instrument": "LOW",
                    "score": 1.0,
                    "label": -0.02,
                    "benchmark_return": 0.01,
                    "market_state": state,
                },
            ]
        )
    frame = pd.DataFrame(rows).set_index(["datetime", "instrument"])

    report = market_state_attribution(
        frame,
        topk=1,
        horizon_days=1,
        total_cost_bps=0,
        labels_are_excess=False,
    )

    assert set(report) == {"risk_on", "neutral", "defensive", "risk_off"}
    assert all(report[state]["period_count"] == 1 for state in states)
    assert all(report[state]["annual_excess_return"] is not None for state in states)


def test_market_state_attribution_uses_one_global_horizon_schedule() -> None:
    rows = []
    dates = pd.bdate_range("2025-01-02", periods=8)
    states = [
        "risk_on",
        "neutral",
        "risk_on",
        "defensive",
        "risk_on",
        "risk_off",
        "risk_on",
        "neutral",
    ]
    for date, state in zip(dates, states, strict=True):
        rows.extend(
            [
                {
                    "datetime": date,
                    "instrument": "HIGH",
                    "score": 2.0,
                    "label": 0.03,
                    "benchmark_return": 0.01,
                    "market_state": state,
                },
                {
                    "datetime": date,
                    "instrument": "LOW",
                    "score": 1.0,
                    "label": -0.02,
                    "benchmark_return": 0.01,
                    "market_state": state,
                },
            ]
        )
    frame = pd.DataFrame(rows).set_index(["datetime", "instrument"])

    report = market_state_attribution(
        frame,
        topk=1,
        horizon_days=2,
        total_cost_bps=0,
        labels_are_excess=False,
    )

    assert report["risk_on"]["period_count"] == 4
    assert report["neutral"]["period_count"] == 0
    assert report["defensive"]["period_count"] == 0
    assert report["risk_off"]["period_count"] == 0
