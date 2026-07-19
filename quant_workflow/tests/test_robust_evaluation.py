from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robust_evaluation.py"
CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
ACCEPTANCE = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["acceptance"]


def load_module():
    spec = importlib.util.spec_from_file_location("robust_evaluation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prediction_frame(
    scores: list[float], labels: list[float] | None = None
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2025-01-02", "2025-01-03"]), ["A", "B", "C"]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {"score": scores, "label": labels or [0.03, 0.02, 0.01, -0.01, 0.0, 0.01]},
        index=index,
    )


def test_seed_aggregation_equal_weights_percentile_ranks_and_preserves_labels() -> None:
    module = load_module()
    first = prediction_frame([1, 2, 3, 3, 2, 1])
    second = prediction_frame([3, 2, 1, 1, 2, 3])

    result = module.aggregate_prediction_frames({"seed42": first, "seed2026": second})

    assert np.allclose(result["score"].values, [2 / 3] * 6)
    assert result["label"].equals(first["label"])


def test_seed_aggregation_rejects_label_drift() -> None:
    module = load_module()
    first = prediction_frame([1, 2, 3, 3, 2, 1])
    second = prediction_frame([1, 2, 3, 3, 2, 1], labels=[9, 8, 7, 6, 5, 4])

    try:
        module.aggregate_prediction_frames({"first": first, "second": second})
    except ValueError as exc:
        assert "label" in str(exc)
    else:
        raise AssertionError("label drift across seed artifacts was accepted")


def test_window_stability_reports_rank_correlation_and_top50_overlap() -> None:
    module = load_module()
    frame = prediction_frame([1, 2, 3, 3, 2, 1])

    result = module.window_stability(
        {504: frame, 756: frame.copy(), 1008: frame.copy()}, topk=2
    )

    assert result["pair_count"] == 3
    assert result["rank_correlation_median"] == 1.0
    assert result["topk_overlap_mean"] == 1.0


def test_seed_stability_reports_all_quarter_window_seed_pairs() -> None:
    module = load_module()
    frame = prediction_frame([1, 2, 3, 3, 2, 1])
    grouped = {
        (504, "2025Q1"): {"42": frame, "2026": frame.copy(), "3407": frame.copy()},
        (756, "2025Q1"): {"42": frame, "2026": frame.copy(), "3407": frame.copy()},
    }

    result = module.seed_stability(grouped, topk=2)

    assert result["applicable"] is True
    assert result["group_count"] == 2
    assert result["pair_count"] == 6
    assert result["rank_correlation_median"] == 1.0
    assert result["topk_overlap_mean"] == 1.0
    assert {item["window_days"] for item in result["pairs"]} == {504, 756}


def test_seed_stability_is_not_applicable_to_fixed_linear_models() -> None:
    module = load_module()
    frame = prediction_frame([1, 2, 3, 3, 2, 1])

    result = module.seed_stability({(504, "2025Q1"): {"fixed": frame}}, topk=2)

    assert result["applicable"] is False
    assert result["pair_count"] == 0
    assert result["rank_correlation_median"] is None
    assert result["topk_overlap_mean"] is None


def test_model_signal_report_requires_rankic_and_both_stability_gates() -> None:
    module = load_module()
    quarters = [{"quarter": f"Q{i}", "rank_ic": 0.02} for i in range(9)] + [
        {"quarter": f"Q{i}", "rank_ic": -0.01} for i in range(9, 12)
    ]

    report = module.model_signal_report(
        model_id="lightgbm",
        quarter_metrics=quarters,
        stability={"rank_correlation_median": 0.75, "topk_overlap_mean": 0.65},
        thresholds=ACCEPTANCE,
    )

    assert report["eligible"] is True
    failed = module.model_signal_report(
        model_id="unstable",
        quarter_metrics=quarters,
        stability={"rank_correlation_median": 0.69, "topk_overlap_mean": 0.65},
        thresholds=ACCEPTANCE,
    )
    assert failed["eligible"] is False
    assert "window_rank_correlation" in failed["failed_gates"]

    stricter = dict(ACCEPTANCE)
    stricter["positive_rank_ic_interval_ratio_min"] = 0.80
    strict = module.model_signal_report(
        model_id="strict",
        quarter_metrics=quarters,
        stability={"rank_correlation_median": 0.75, "topk_overlap_mean": 0.65},
        thresholds=stricter,
    )
    assert strict["eligible"] is False
    assert "positive_rank_ic_intervals" in strict["failed_gates"]

    seed_unstable = module.model_signal_report(
        model_id="seed-unstable",
        quarter_metrics=quarters,
        stability={"rank_correlation_median": 0.75, "topk_overlap_mean": 0.65},
        seed_stability={
            "applicable": True,
            "rank_correlation_median": 0.69,
            "topk_overlap_mean": 0.65,
        },
        thresholds=ACCEPTANCE,
    )
    assert seed_unstable["eligible"] is False
    assert "seed_rank_correlation" in seed_unstable["failed_gates"]


def test_model_selection_uses_only_eligible_development_score() -> None:
    module = load_module()
    reports = [
        {
            "model_id": "a",
            "eligible": True,
            "development_score": 1.2,
            "confirmation_score": -100,
        },
        {
            "model_id": "b",
            "eligible": True,
            "development_score": 1.0,
            "confirmation_score": 100,
        },
        {
            "model_id": "c",
            "eligible": True,
            "development_score": 0.8,
            "confirmation_score": 50,
        },
        {
            "model_id": "d",
            "eligible": False,
            "development_score": 99,
            "confirmation_score": 99,
        },
    ]

    selected = module.select_development_models(reports, minimum=2, maximum=3)

    assert [item["model_id"] for item in selected] == ["a", "b", "c"]


def test_confirmation_and_shadow_gates_fail_closed() -> None:
    module = load_module()

    confirmation = module.confirmation_verdict(
        annual_excess_return=0.01,
        information_ratio=0.2,
        max_drawdown=-0.10,
        thresholds=ACCEPTANCE,
    )
    assert confirmation["passed"] is True
    assert (
        module.confirmation_verdict(
            annual_excess_return=0.0,
            information_ratio=0.2,
            max_drawdown=-0.10,
            thresholds=ACCEPTANCE,
        )["passed"]
        is False
    )

    shadow = module.shadow_verdict(
        trading_days=60,
        net_excess_return=0.01,
        max_drawdown=-0.08,
        task_success_ratio=0.97,
        thresholds=ACCEPTANCE,
    )
    assert shadow["status"] == "robust_strategy_qualified"
    assert (
        module.shadow_verdict(
            trading_days=59,
            net_excess_return=0.02,
            max_drawdown=-0.05,
            task_success_ratio=1.0,
            thresholds=ACCEPTANCE,
        )["status"]
        == "shadow_observation"
    )

    stricter_confirmation = dict(ACCEPTANCE)
    stricter_confirmation["confirmation_excess_return_min_exclusive"] = 0.02
    assert (
        module.confirmation_verdict(
            annual_excess_return=0.01,
            information_ratio=0.2,
            max_drawdown=-0.10,
            thresholds=stricter_confirmation,
        )["passed"]
        is False
    )
    longer_shadow = dict(ACCEPTANCE)
    longer_shadow["shadow_days"] = 61
    strict_shadow = module.shadow_verdict(
        trading_days=60,
        net_excess_return=0.01,
        max_drawdown=-0.08,
        task_success_ratio=0.97,
        thresholds=longer_shadow,
    )
    assert strict_shadow["status"] == "shadow_observation"
    assert strict_shadow["remaining_days"] == 1


def test_aggregate_training_artifacts_builds_model_windows_and_quarter_reports(
    tmp_path: Path,
) -> None:
    module = load_module()
    records = []
    for window in [504, 756, 1008]:
        for seed, scores in [(42, [3, 2, 1, 1, 2, 3]), (2026, [3, 2, 1, 1, 2, 3])]:
            path = tmp_path / f"w{window}_s{seed}.parquet"
            prediction_frame(scores).reset_index().to_parquet(path, index=False)
            records.append(
                {
                    "model_id": "lightgbm",
                    "window_days": window,
                    "quarter": "2025Q1",
                    "seed": seed,
                    "prediction_path": str(path),
                }
            )

    result = module.aggregate_training_artifacts(records, topk=2, thresholds=ACCEPTANCE)

    assert set(result["model_predictions"]) == {"lightgbm"}
    assert result["model_reports"][0]["model_id"] == "lightgbm"
    assert result["model_reports"][0]["eligible"] is True
    assert result["model_reports"][0]["seed_stability"]["pair_count"] == 3
    assert result["model_reports"][0]["gates"]["seed_rank_correlation"] is True
    assert result["model_reports"][0]["gates"]["seed_top50_overlap"] is True
    assert result["model_reports"][0]["quarter_metrics"][0]["quarter"] == "2025Q1"
    assert result["model_reports"][0]["quarter_metrics"][0]["rank_ic"] == 1.0


def test_strategy_candidates_run_baseline_stress_delay_and_choose_only_passing_development(
    tmp_path: Path,
) -> None:
    module = load_module()
    config_path = (
        Path(__file__).resolve().parents[1] / "configs" / "robust_strategy.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dates = []
    prediction_rows = []
    for quarter in config["rolling_validation"]["test_quarters"]:
        period = pd.Period(quarter, freq="Q")
        quarter_dates = pd.bdate_range(period.start_time, periods=20)
        dates.extend(quarter_dates)
        signal_day = quarter_dates[0]
        for symbol, score, label in [
            ("A", 3.0, 0.03),
            ("B", 2.0, 0.02),
            ("C", 1.0, 0.01),
        ]:
            prediction_rows.append((signal_day, symbol, score, label))
    predictions = pd.DataFrame(
        prediction_rows, columns=["datetime", "instrument", "score", "label"]
    ).set_index(["datetime", "instrument"])
    benchmark = pd.DataFrame({"date": dates, "open": [100.0] * len(dates)})
    market = pd.DataFrame({"date": dates})
    regime_names = ["risk_on", "neutral", "defensive", "risk_off"]
    regimes = pd.Series(
        [regime_names[index % 4] for index in range(len(dates))],
        index=pd.DatetimeIndex(dates),
    )
    calls = []

    def simulator(**kwargs):
        calls.append(
            (
                kwargs["strategy_id"],
                kwargs.get("all_in_round_trip_bps"),
                kwargs.get("additional_execution_delay_days", 0),
            )
        )
        daily = pd.DataFrame(
            {
                "date": dates,
                "portfolio_return": [
                    0.001 if index % 2 == 0 else 0.0005 for index in range(len(dates))
                ],
                "benchmark_return": [0.0001] * len(dates),
                "transaction_cost": [0.0] * len(dates),
                "turnover": [0.1] * len(dates),
                "board_exposure_main": [1.0] * len(dates),
                "board_exposure_growth": [0.0] * len(dates),
                "board_exposure_star": [0.0] * len(dates),
                "board_start_exposure_main": [1.0] * len(dates),
                "board_start_exposure_growth": [0.0] * len(dates),
                "board_start_exposure_star": [0.0] * len(dates),
                "cash_start_exposure": [0.0] * len(dates),
            }
        )
        daily["board_return_contribution_main"] = daily["portfolio_return"]
        daily["board_return_contribution_growth"] = 0.0
        daily["board_return_contribution_star"] = 0.0
        daily["board_excess_contribution_main"] = (
            daily["portfolio_return"] - daily["benchmark_return"]
        )
        daily["board_excess_contribution_growth"] = 0.0
        daily["board_excess_contribution_star"] = 0.0
        daily["cash_excess_contribution"] = 0.0
        from robust_backtest import performance_metrics

        return {
            "strategy_id": kwargs["strategy_id"],
            "daily": daily,
            "metrics": performance_metrics(daily),
        }

    result = module.evaluate_strategy_candidates(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        regimes=regimes,
        config=config,
        selected_model_reports=[
            {
                "model_id": "a",
                "stability": {"rank_correlation_median": 0.8, "topk_overlap_mean": 0.7},
            },
            {
                "model_id": "b",
                "stability": {
                    "rank_correlation_median": 0.75,
                    "topk_overlap_mean": 0.65,
                },
            },
        ],
        simulator=simulator,
    )

    assert len(calls) == 20
    assert len(result["strategies"]) == 5
    assert result["status"] == "research_candidate_ready"
    assert result["selected_strategy"]["acceptance"]["passed"] is True
    assert result["selected_strategy"]["strategy_id"] == "full_risk_control"
    assert (
        sum(bool(item["eligible_for_final_selection"]) for item in result["strategies"])
        == 1
    )
    assert all(
        item["stress"]["metrics"]["annual_excess_return"] > 0
        for item in result["strategies"]
    )
    assert all("low_cost" in item for item in result["strategies"])
    assert all(
        set(item["board_attribution"]) == {"main", "growth", "star"}
        for item in result["strategies"]
    )
    assert all(
        item["attribution_reconciliation"]["reconciled"] is True
        for item in result["strategies"]
    )


def test_regime_attribution_uses_previous_close_regime_for_next_open_return() -> None:
    module = load_module()
    dates = pd.bdate_range("2025-01-02", periods=4)
    daily = pd.DataFrame(
        {
            "date": dates,
            "portfolio_return": [0.0, 0.01, 0.02, 0.03],
            "benchmark_return": [0.0] * 4,
            "transaction_cost": [0.0] * 4,
            "turnover": [0.0] * 4,
        }
    )
    regimes = pd.Series(["risk_on", "neutral", "defensive", "risk_off"], index=dates)

    result = module._regime_metrics(daily, regimes)

    assert result["risk_on"]["days"] == 1
    assert result["risk_on"]["annual_return"] > 0
    assert result["neutral"]["days"] == 1
    assert result["defensive"]["days"] == 1
    assert result["risk_off"]["days"] == 0


def test_board_attribution_reports_average_and_peak_exposure() -> None:
    module = load_module()
    daily = pd.DataFrame(
        {
            "board_exposure_main": [0.5, 0.4],
            "board_exposure_growth": [0.2, 0.3],
            "board_exposure_star": [0.1, 0.0],
            "board_start_exposure_main": [0.5, 0.4],
            "board_start_exposure_growth": [0.2, 0.3],
            "board_start_exposure_star": [0.1, 0.0],
            "cash_start_exposure": [0.2, 0.3],
            "cash_excess_contribution": [-0.0008, -0.0015],
            "portfolio_return": [0.012, 0.002],
            "benchmark_return": [0.004, 0.005],
            "board_return_contribution_main": [0.01, -0.002],
            "board_return_contribution_growth": [0.003, 0.004],
            "board_return_contribution_star": [-0.001, 0.0],
            "board_excess_contribution_main": [0.008, -0.004],
            "board_excess_contribution_growth": [0.0022, 0.0025],
            "board_excess_contribution_star": [-0.0014, 0.0],
        }
    )

    result = module._board_attribution(daily)

    assert result["main"]["average_exposure"] == 0.45
    assert result["main"]["max_exposure"] == 0.5
    assert result["main"]["days"] == 2
    assert np.isclose(result["main"]["return_contribution"], 0.007976)
    assert np.isclose(result["main"]["excess_contribution"], 0.003968)
    assert result["growth"]["max_exposure"] == 0.3
    assert result["star"]["average_exposure"] == 0.05

    reconciliation = module._attribution_reconciliation(daily, result)
    assert reconciliation["reconciled"] is True
    assert np.isclose(reconciliation["cash_excess_contribution"], -0.002306)
    assert np.isclose(reconciliation["cumulative_excess_return"], 0.005004)
    assert abs(reconciliation["excess_residual"]) < 1e-12
