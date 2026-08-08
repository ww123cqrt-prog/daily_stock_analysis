"""Deterministic model helpers for limit-up offline research."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


LIGHTGBM_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131, 167, 199)


def preregistered_lightgbm_seeds() -> tuple[int, ...]:
    return LIGHTGBM_SEEDS


def _numeric_frame(frame: pd.DataFrame, *, columns: Sequence[str] | None = None) -> pd.DataFrame:
    result = frame.copy()
    if columns is not None:
        if set(result.columns) != set(columns):
            raise ValueError("model feature columns do not match")
        result = result.loc[:, list(columns)]
    for column in result:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result.empty or result.isna().any().any():
        raise ValueError("model features must be nonempty finite numbers")
    values = result.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("model features must be nonempty finite numbers")
    return result


def _numeric_target(values: pd.Series, *, expected_length: int) -> pd.Series:
    target = pd.to_numeric(values, errors="coerce").astype(float).reset_index(drop=True)
    if len(target) != expected_length or target.isna().any() or not np.isfinite(target).all():
        raise ValueError("model target is invalid")
    return target


def _fit_baseline(
    train_x: pd.DataFrame,
    train_returns: pd.Series,
    predict_x: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = (train_returns > 0).astype(int)
    if labels.nunique() < 2:
        probability = np.full(len(predict_x), float(labels.iloc[0]), dtype=float)
    else:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, random_state=int(seed)),
        )
        classifier.fit(train_x, labels)
        probability = classifier.predict_proba(predict_x)[:, 1]
    regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    regressor.fit(train_x, train_returns)
    predicted_return = regressor.predict(predict_x)
    return probability.astype(float), np.asarray(predicted_return, dtype=float)


def _fit_lightgbm(
    train_x: pd.DataFrame,
    train_returns: pd.Series,
    predict_x: pd.DataFrame,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    from lightgbm import LGBMClassifier, LGBMRegressor

    common = {
        "n_estimators": 100,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": int(seed),
        "n_jobs": 1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }
    labels = (train_returns > 0).astype(int)
    if labels.nunique() < 2:
        probability = np.full(len(predict_x), float(labels.iloc[0]), dtype=float)
    else:
        classifier = LGBMClassifier(objective="binary", **common)
        classifier.fit(train_x, labels)
        probability = classifier.predict_proba(predict_x)[:, 1]
    regressor = LGBMRegressor(objective="regression_l2", **common)
    regressor.fit(train_x, train_returns)
    predicted_return = regressor.predict(predict_x)
    return probability.astype(float), np.asarray(predicted_return, dtype=float)


def _raw_seed_predictions(
    train_x: pd.DataFrame,
    train_returns: pd.Series,
    predict_x: pd.DataFrame,
    *,
    family: str,
    seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    fit = _fit_baseline if family == "baseline" else _fit_lightgbm
    effective_seeds = tuple(seeds[:1]) if family == "baseline" else tuple(seeds)
    if not effective_seeds:
        raise ValueError("at least one model seed is required")
    for seed in effective_seeds:
        probability, predicted_return = fit(
            train_x, train_returns, predict_x, seed=int(seed)
        )
        probabilities.append(probability)
        returns.append(predicted_return)
    return np.column_stack(probabilities), np.column_stack(returns)


def _platt_calibrate(
    calibration_probability: np.ndarray,
    calibration_returns: pd.Series,
    test_probability: np.ndarray,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    labels = (calibration_returns > 0).astype(int)
    if labels.nunique() < 2:
        return np.full(len(test_probability), float(labels.iloc[0]), dtype=float)
    calibrator = LogisticRegression(C=1.0, max_iter=2000, random_state=0)
    calibrator.fit(calibration_probability.reshape(-1, 1), labels)
    return calibrator.predict_proba(test_probability.reshape(-1, 1))[:, 1]


def fit_predict_family(
    train_x: pd.DataFrame,
    train_returns: pd.Series,
    calibration_x: pd.DataFrame,
    calibration_returns: pd.Series,
    test_x: pd.DataFrame,
    *,
    family: str,
    seeds: Sequence[int] = LIGHTGBM_SEEDS,
) -> dict[str, object]:
    """Fit one preregistered family and calibrate on a disjoint time slice."""

    if family not in {"baseline", "lightgbm_ensemble"}:
        raise ValueError("family must be baseline or lightgbm_ensemble")
    train = _numeric_frame(train_x)
    calibration = _numeric_frame(calibration_x, columns=train.columns)
    test = _numeric_frame(test_x, columns=train.columns)
    train_target = _numeric_target(train_returns, expected_length=len(train))
    calibration_target = _numeric_target(
        calibration_returns, expected_length=len(calibration)
    )
    seed_values = tuple(int(seed) for seed in seeds)
    calibration_probabilities, _ = _raw_seed_predictions(
        train,
        train_target,
        calibration,
        family=family,
        seeds=seed_values,
    )
    test_probabilities, test_returns = _raw_seed_predictions(
        train,
        train_target,
        test,
        family=family,
        seeds=seed_values,
    )
    raw_calibration = calibration_probabilities.mean(axis=1)
    raw_test = test_probabilities.mean(axis=1)
    calibrated = _platt_calibrate(raw_calibration, calibration_target, raw_test)
    predictions = pd.DataFrame(
        {
            "raw_probability": raw_test,
            "calibrated_probability": np.clip(calibrated, 0.0, 1.0),
            "predicted_net_return": test_returns.mean(axis=1),
            "seed_probability_std": test_probabilities.std(axis=1, ddof=0),
            "seed_return_std": test_returns.std(axis=1, ddof=0),
        }
    )
    return {
        "family": family,
        "seeds": list(seed_values[:1] if family == "baseline" else seed_values),
        "predictions": predictions,
        "seed_probabilities": test_probabilities,
        "seed_returns": test_returns,
    }


def quarterly_rolling_splits(
    rows: pd.DataFrame,
    *,
    date_column: str = "date",
    start: str = "2023-01-01",
    end: str = "2025-12-31",
) -> list[dict[str, object]]:
    """Create expanding-window quarterly OOS splits with 2026 locked out."""

    if date_column not in rows.columns:
        raise ValueError("rows are missing the date column")
    dates = pd.to_datetime(rows[date_column], errors="coerce").dt.normalize()
    if dates.isna().any():
        raise ValueError("rows contain invalid dates")
    first = pd.Timestamp(start).to_period("Q")
    last = pd.Timestamp(end).to_period("Q")
    splits: list[dict[str, object]] = []
    for quarter in pd.period_range(first, last, freq="Q"):
        quarter_start = quarter.start_time.normalize()
        quarter_end = min(quarter.end_time.normalize(), pd.Timestamp(end).normalize())
        train_mask = dates < quarter_start
        test_mask = (dates >= quarter_start) & (dates <= quarter_end)
        if not train_mask.any() or not test_mask.any():
            raise ValueError(f"rolling split {quarter} has no train or test rows")
        splits.append(
            {
                "quarter": f"{quarter.year}Q{quarter.quarter}",
                "train_index": rows.index[train_mask].tolist(),
                "test_index": rows.index[test_mask].tolist(),
                "test_start": str(quarter_start.date()),
                "test_end": str(quarter_end.date()),
            }
        )
    return splits


def block_bootstrap_mean(
    values: Sequence[float] | pd.Series,
    *,
    block_size: int,
    samples: int,
    random_seed: int,
) -> dict[str, float | int]:
    """Moving-block bootstrap confidence interval for serial event returns."""

    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty or block_size <= 0 or samples <= 0:
        raise ValueError("bootstrap requires values, positive block_size, and samples")
    data = series.to_numpy(dtype=float)
    block = min(int(block_size), len(data))
    rng = np.random.default_rng(int(random_seed))
    starts = np.arange(0, len(data) - block + 1)
    sample_means: list[float] = []
    blocks_needed = int(np.ceil(len(data) / block))
    for _ in range(int(samples)):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        draw = np.concatenate([data[start : start + block] for start in chosen])[
            : len(data)
        ]
        sample_means.append(float(draw.mean()))
    return {
        "mean": float(data.mean()),
        "lower_95": float(np.quantile(sample_means, 0.025)),
        "upper_95": float(np.quantile(sample_means, 0.975)),
        "block_size": block,
        "samples": int(samples),
        "random_seed": int(random_seed),
    }


def leave_one_out_topk_stability(
    probabilities: np.ndarray,
    *,
    symbols: Sequence[str],
    topk: int,
) -> dict[str, object]:
    """Compare full-ensemble TopK with every leave-one-seed-out ensemble."""

    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(symbols):
        raise ValueError("probability matrix and symbols do not align")
    if matrix.shape[1] < 2 or topk <= 0:
        raise ValueError("leave-one-out stability requires at least two models and TopK")
    if not np.isfinite(matrix).all():
        raise ValueError("probability matrix contains invalid values")
    symbol_values = np.asarray([str(symbol) for symbol in symbols], dtype=object)

    def top_set(scores: np.ndarray) -> list[str]:
        order = np.argsort(-scores, kind="stable")[: min(int(topk), len(scores))]
        return symbol_values[order].tolist()

    full = top_set(matrix.mean(axis=1))
    full_set = set(full)
    rows: list[dict[str, object]] = []
    changed_total = 0
    for column in range(matrix.shape[1]):
        remaining = np.delete(matrix, column, axis=1).mean(axis=1)
        selected = top_set(remaining)
        changed = len(full_set - set(selected))
        changed_total += changed
        rows.append(
            {
                "left_out_model_index": column,
                "topk": selected,
                "changed_member_count": changed,
            }
        )
    return {
        "full_topk": full,
        "leave_one_out": rows,
        "changed_member_count": int(changed_total),
        "max_changed_members": int(
            max(row["changed_member_count"] for row in rows)
        ),
    }
