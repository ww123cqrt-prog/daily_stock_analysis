"""Deterministic baseline models for robust signal discovery."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _feature_and_label(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(frame.columns, pd.MultiIndex):
        raise ValueError("training frame must contain feature and label column groups")
    features = frame.xs("feature", axis=1, level=0, drop_level=True)
    labels = frame.xs("label", axis=1, level=0, drop_level=True)
    if labels.shape[1] != 1:
        raise ValueError("training frame must contain one label column")
    return features, labels.iloc[:, 0]


class EqualWeightRankModel:
    """Average same-day factor ranks after learning each factor direction on train."""

    def __init__(self, minimum_absolute_ic: float = 0.0) -> None:
        if minimum_absolute_ic < 0:
            raise ValueError("minimum_absolute_ic must be nonnegative")
        self.minimum_absolute_ic = float(minimum_absolute_ic)
        self.factor_signs: dict[str, float] = {}

    def fit(self, dataset: Any) -> "EqualWeightRankModel":
        frame = dataset.prepare(
            "train", col_set=["feature", "label"], data_key="learn"
        )
        features, label = _feature_and_label(frame)
        signs: dict[str, float] = {}
        for column in features.columns:
            joined = pd.concat(
                [pd.to_numeric(features[column], errors="coerce"), pd.to_numeric(label, errors="coerce")],
                axis=1,
            ).dropna()
            if joined.empty:
                continue
            daily_ic = joined.groupby(level="datetime", sort=True).apply(
                lambda group: group.iloc[:, 0].corr(group.iloc[:, 1], method="spearman")
            )
            clean = pd.to_numeric(daily_ic, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            mean_ic = float(clean.mean()) if not clean.empty else 0.0
            if abs(mean_ic) >= self.minimum_absolute_ic and mean_ic != 0.0:
                signs[str(column)] = 1.0 if mean_ic > 0 else -1.0
        if not signs:
            raise ValueError("training data did not produce any directional factors")
        self.factor_signs = signs
        return self

    def predict(self, dataset: Any, segment: str = "test") -> pd.Series:
        if not self.factor_signs:
            raise RuntimeError("EqualWeightRankModel must be fitted before prediction")
        features = dataset.prepare(segment, col_set="feature", data_key="infer")
        available = [column for column in self.factor_signs if column in features.columns]
        if not available:
            raise ValueError("prediction data has none of the fitted factors")
        ranked = features.loc[:, available].apply(pd.to_numeric, errors="coerce")
        ranked = ranked.groupby(level="datetime", sort=False).rank(method="average", pct=True)
        for column in available:
            if self.factor_signs[column] < 0:
                ranked[column] = 1.0 - ranked[column]
        return ranked.mean(axis=1, skipna=True).rename("score")
