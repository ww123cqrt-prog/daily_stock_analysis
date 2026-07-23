from pathlib import Path

import numpy as np
import pandas as pd

from robust_signal_dataset import (
    SignalResearchDataset,
    add_market_regime_features,
    apply_excess_label,
    label_expression,
    match_industry_intervals,
    normalize_cross_section,
    select_feature_family,
)


def test_label_expression_uses_t_plus_one_open_and_horizon_end_open() -> None:
    assert label_expression(5) == "Ref($open, -6) / Ref($open, -1) - 1"
    assert label_expression(10) == "Ref($open, -11) / Ref($open, -1) - 1"


def test_select_feature_family_uses_registered_prefixes() -> None:
    frame = pd.DataFrame(
        {"ROC5": [1.0], "MA10": [2.0], "STD20": [3.0], "BOARD_MAIN": [1.0]}
    )

    selected = select_feature_family(frame, prefixes=["ROC", "MA"])

    assert list(selected.columns) == ["ROC5", "MA10"]


def test_apply_excess_label_subtracts_same_date_benchmark() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-03"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    labels = pd.DataFrame({"LABEL0": [0.05, 0.02, 0.01]}, index=index)
    benchmark = pd.Series(
        [0.01, -0.01],
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )

    result = apply_excess_label(labels, benchmark)

    assert np.allclose(result["LABEL0"], [0.04, 0.01, 0.02])


def test_industry_interval_matching_uses_historical_date(tmp_path: Path) -> None:
    intervals = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000", "SZ000001"],
            "industry_code": ["OLD", "NEW", "BANK"],
            "start_date": ["2020-01-01", "2023-01-01", "2020-01-01"],
            "end_date": ["2022-12-31", "2025-12-31", "2025-12-31"],
        }
    )
    path = tmp_path / "industry.parquet"
    intervals.to_parquet(path)
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2022-06-01"), "SH600000"),
            (pd.Timestamp("2024-06-01"), "SH600000"),
            (pd.Timestamp("2024-06-01"), "SZ000001"),
        ],
        names=["datetime", "instrument"],
    )

    matched = match_industry_intervals(index, path)

    assert matched.tolist() == ["OLD", "NEW", "BANK"]


def test_industry_normalization_is_same_day_and_same_industry() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "A"),
            (pd.Timestamp("2025-01-02"), "B"),
            (pd.Timestamp("2025-01-02"), "C"),
            (pd.Timestamp("2025-01-02"), "D"),
        ],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame({"F": [1.0, 3.0, 10.0, 14.0]}, index=index)

    normalized = normalize_cross_section(
        frame, groups=pd.Index(["I1", "I1", "I2", "I2"])
    )

    assert np.allclose(normalized["F"], [-1.0, 1.0, -1.0, 1.0])


def test_market_regime_features_are_same_day_one_hot_values() -> None:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03"])
    index = pd.MultiIndex.from_product(
        [dates, ["SH600000", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame({"F": [1.0, 2.0, 3.0, 4.0]}, index=index)
    states = pd.Series(["risk_on", "defensive"], index=dates)

    result = add_market_regime_features(features, states)

    first = result.loc[dates[0]]
    second = result.loc[dates[1]]
    assert first["MARKET_REGIME_RISK_ON"].eq(1.0).all()
    assert first["MARKET_REGIME_DEFENSIVE"].eq(0.0).all()
    assert second["MARKET_REGIME_RISK_ON"].eq(0.0).all()
    assert second["MARKET_REGIME_DEFENSIVE"].eq(1.0).all()


def test_market_regime_features_fail_closed_when_a_date_is_missing() -> None:
    date = pd.Timestamp("2025-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "SH600000")], names=["datetime", "instrument"]
    )

    with np.testing.assert_raises_regex(ValueError, "missing"):
        add_market_regime_features(
            pd.DataFrame({"F": [1.0]}, index=index),
            pd.Series(dtype="string"),
        )


def test_prepare_raw_features_does_not_apply_model_input_normalization(
    monkeypatch,
) -> None:
    date = pd.Timestamp("2025-01-02")
    index = pd.MultiIndex.from_tuples(
        [(date, "SH600000"), (date, "SH600001")],
        names=["datetime", "instrument"],
    )
    raw = pd.DataFrame({"ROC5": [1.0, np.nan], "STD20": [3.0, 4.0]}, index=index)
    dataset = object.__new__(SignalResearchDataset)
    dataset.feature_prefixes = ("ROC",)

    monkeypatch.setattr(
        "robust_signal_dataset.DatasetH.prepare",
        lambda _self, _segments, **_kwargs: raw,
    )

    result = dataset.prepare_raw_features("test")

    assert list(result.columns) == ["ROC5"]
    assert result["ROC5"].isna().sum() == 1
