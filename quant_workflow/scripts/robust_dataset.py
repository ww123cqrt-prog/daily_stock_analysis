"""Qlib DatasetH wrapper that normalizes Alpha158 features within each board."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH


BOARD_NAMES = ("main", "growth", "star", "other")
MARKET_STATE_FEATURES = (
    "MARKET_STATE_MEAN",
    "MARKET_STATE_BREADTH",
    "MARKET_STATE_DISPERSION",
)


def board_from_symbol(symbol: Any) -> str:
    value = str(symbol).strip().upper()
    if value.startswith(("SH60", "SZ00")):
        return "main"
    if value.startswith("SZ30"):
        return "growth"
    if value.startswith("SH68"):
        return "star"
    return "other"


def _index_values(frame: pd.DataFrame, name: str) -> pd.Index:
    if not isinstance(frame.index, pd.MultiIndex) or name not in frame.index.names:
        raise ValueError(f"feature frame index must contain {name}")
    return frame.index.get_level_values(name)


def normalize_feature_frame(
    features: pd.DataFrame,
    *,
    add_board_features: bool = True,
    add_market_state_features: bool = True,
) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    dates = pd.to_datetime(
        _index_values(features, "datetime"), errors="coerce"
    ).normalize()
    instruments = _index_values(features, "instrument").astype(str)
    boards = pd.Index(
        [board_from_symbol(symbol) for symbol in instruments], name="board"
    )
    numeric = features.apply(pd.to_numeric, errors="coerce")
    keys = [dates, boards]
    means = numeric.groupby(keys, sort=False).transform("mean")
    centered = numeric - means
    variances = centered.pow(2).groupby(keys, sort=False).transform("mean")
    scales = np.sqrt(variances).replace(0.0, np.nan)
    normalized = (centered / scales).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if add_market_state_features:
        row_mean = numeric.mean(axis=1).fillna(0.0)
        row_breadth = numeric.gt(0.0).mean(axis=1).fillna(0.0)
        normalized["MARKET_STATE_MEAN"] = row_mean.groupby(dates, sort=False).transform(
            "mean"
        )
        normalized["MARKET_STATE_BREADTH"] = row_breadth.groupby(
            dates, sort=False
        ).transform("mean")
        normalized["MARKET_STATE_DISPERSION"] = row_mean.groupby(
            dates, sort=False
        ).transform(lambda values: values.std(ddof=0))
    if add_board_features:
        for board in BOARD_NAMES:
            normalized[f"BOARD_{board.upper()}"] = (boards == board).astype(float)
    return normalized


def transform_prepared_frame(
    frame: pd.DataFrame,
    *,
    col_set: Any,
    normalize_within_board: bool = True,
    add_board_features: bool = True,
    add_market_state_features: bool = True,
) -> pd.DataFrame:
    if not normalize_within_board or frame.empty:
        return frame.copy()
    if isinstance(
        frame.columns, pd.MultiIndex
    ) and "feature" in frame.columns.get_level_values(0):
        feature_block = frame.xs("feature", axis=1, level=0, drop_level=True)
        normalized = normalize_feature_frame(
            feature_block,
            add_board_features=add_board_features,
            add_market_state_features=add_market_state_features,
        )
        normalized.columns = pd.MultiIndex.from_tuples(
            [("feature", str(column)) for column in normalized.columns]
        )
        other = frame.loc[:, frame.columns.get_level_values(0) != "feature"].copy()
        return pd.concat([normalized, other], axis=1)
    feature_only = (
        col_set == "feature" or col_set == ["feature"] or col_set == ("feature",)
    )
    if feature_only:
        return normalize_feature_frame(
            frame,
            add_board_features=add_board_features,
            add_market_state_features=add_market_state_features,
        )
    return frame.copy()


class BoardAwareDataset(DatasetH):
    """DatasetH with same-date, same-board feature normalization."""

    def __init__(
        self,
        *args: Any,
        normalize_within_board: bool = True,
        add_board_features: bool = True,
        add_market_state_features: bool = True,
        **kwargs: Any,
    ) -> None:
        self.normalize_within_board = bool(normalize_within_board)
        self.add_board_features = bool(add_board_features)
        self.add_market_state_features = bool(add_market_state_features)
        super().__init__(*args, **kwargs)

    def prepare(
        self,
        segments: Any,
        col_set: Any = "__all",
        data_key: str = "infer",
        **kwargs: Any,
    ):
        prepared = super().prepare(
            segments, col_set=col_set, data_key=data_key, **kwargs
        )

        def transform(frame: pd.DataFrame) -> pd.DataFrame:
            return transform_prepared_frame(
                frame,
                col_set=col_set,
                normalize_within_board=self.normalize_within_board,
                add_board_features=self.add_board_features,
                add_market_state_features=self.add_market_state_features,
            )

        if isinstance(prepared, list):
            return [transform(frame) for frame in prepared]
        return transform(prepared)
