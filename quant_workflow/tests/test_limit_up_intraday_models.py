from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "limit_up_intraday_models.py"
    )
    assert path.exists(), "limit-up model module is missing"
    spec = importlib.util.spec_from_file_location("limit_up_intraday_models", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_frames() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        rng.normal(size=(80, 4)), columns=["f1", "f2", "f3", "f4"]
    )
    returns = pd.Series(
        0.03 * features["f1"] - 0.02 * features["f2"] + rng.normal(0, 0.01, 80)
    )
    return (
        features.iloc[:50].reset_index(drop=True),
        returns.iloc[:50].reset_index(drop=True),
        features.iloc[50:65].reset_index(drop=True),
        returns.iloc[50:65].reset_index(drop=True),
        features.iloc[65:].reset_index(drop=True),
    )


@pytest.mark.parametrize("family", ["baseline", "lightgbm_ensemble"])
def test_model_families_return_calibrated_probability_and_return_predictions(
    family: str,
) -> None:
    module = load_module()
    train_x, train_y, calibration_x, calibration_y, test_x = model_frames()
    seeds = (11, 23, 37) if family == "lightgbm_ensemble" else (11,)

    first = module.fit_predict_family(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        test_x,
        family=family,
        seeds=seeds,
    )
    second = module.fit_predict_family(
        train_x,
        train_y,
        calibration_x,
        calibration_y,
        test_x,
        family=family,
        seeds=seeds,
    )

    predictions = first["predictions"]
    assert predictions["calibrated_probability"].between(0, 1).all()
    assert np.isfinite(predictions["predicted_net_return"]).all()
    assert np.isfinite(predictions["seed_probability_std"]).all()
    assert np.isfinite(predictions["seed_return_std"]).all()
    pd.testing.assert_frame_equal(predictions, second["predictions"])
    assert first["seed_probabilities"].shape == (len(test_x), len(seeds))
    assert first["seed_returns"].shape == (len(test_x), len(seeds))


def test_lightgbm_preregistered_seed_list_has_ten_fixed_members() -> None:
    module = load_module()
    seeds = module.preregistered_lightgbm_seeds()

    assert len(seeds) == 10
    assert len(set(seeds)) == 10
    assert seeds == module.preregistered_lightgbm_seeds()


def test_quarterly_rolling_splits_never_train_on_test_or_2026_rows() -> None:
    module = load_module()
    dates = pd.date_range("2018-01-02", "2026-02-01", freq="20D")
    rows = pd.DataFrame({"date": dates, "value": range(len(dates))})

    splits = module.quarterly_rolling_splits(rows)

    assert splits[0]["quarter"] == "2023Q1"
    assert splits[-1]["quarter"] == "2025Q4"
    assert len(splits) == 12
    for split in splits:
        train_dates = pd.to_datetime(rows.loc[split["train_index"], "date"])
        test_dates = pd.to_datetime(rows.loc[split["test_index"], "date"])
        assert train_dates.max() < test_dates.min()
        assert test_dates.max() < pd.Timestamp("2026-01-01")


def test_block_bootstrap_and_leave_one_out_stability_are_deterministic() -> None:
    module = load_module()
    values = pd.Series([0.03, -0.01, 0.02, 0.01, -0.005] * 8)
    first = module.block_bootstrap_mean(
        values, block_size=5, samples=200, random_seed=19
    )
    second = module.block_bootstrap_mean(
        values, block_size=5, samples=200, random_seed=19
    )
    probabilities = np.array(
        [
            [0.9, 0.89, 0.91],
            [0.8, 0.79, 0.81],
            [0.7, 0.69, 0.71],
            [0.6, 0.59, 0.61],
        ]
    )
    stability = module.leave_one_out_topk_stability(
        probabilities, symbols=["A", "B", "C", "D"], topk=3
    )

    assert first == second
    assert first["lower_95"] <= first["mean"] <= first["upper_95"]
    assert stability["full_topk"] == ["A", "B", "C"]
    assert stability["changed_member_count"] == 0
