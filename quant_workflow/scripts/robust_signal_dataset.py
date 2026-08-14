"""Leakage-safe dataset transformations for robust signal research."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH

from robust_dataset import BOARD_NAMES, board_from_symbol
from robust_strategy_core import RollingSplit


MARKET_REGIMES = ("risk_on", "neutral", "defensive", "risk_off")
MARKET_REGIME_FEATURES = tuple(
    f"MARKET_REGIME_{state.upper()}" for state in MARKET_REGIMES
)
LOCKED_CONFIRMATION_START = pd.Timestamp("2026-01-01")


def label_expression(horizon_days: int) -> str:
    horizon = int(horizon_days)
    if horizon not in {5, 10}:
        raise ValueError("signal research horizon_days must be 5 or 10")
    return f"Ref($open, -{horizon + 1}) / Ref($open, -1) - 1"


def select_feature_family(
    features: pd.DataFrame, *, prefixes: Sequence[str]
) -> pd.DataFrame:
    if not prefixes:
        return features.copy()
    normalized = tuple(str(prefix).upper() for prefix in prefixes)
    selected = [
        column
        for column in features.columns
        if str(column).upper().startswith(normalized)
    ]
    if not selected:
        raise ValueError("feature family selected no Alpha158 columns")
    return features.loc[:, selected].copy()


@lru_cache(maxsize=24)
def _read_external_feature_year(path: str) -> pd.DataFrame:
    return _normalize_external_features(pd.read_parquet(path))


def _normalize_external_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"datetime", "instrument"}
    if not required.issubset(frame.columns):
        raise ValueError("external features are missing datetime or instrument")
    result = frame.copy()
    result["datetime"] = pd.to_datetime(result["datetime"], errors="coerce").dt.normalize()
    result["instrument"] = result["instrument"].astype(str).str.upper()
    if result["datetime"].isna().any():
        raise ValueError("external features contain invalid dates")
    if (result["datetime"] >= LOCKED_CONFIRMATION_START).any():
        raise ValueError("Goal 1 external features attempted to expose 2026 data")
    if result.duplicated(["datetime", "instrument"]).any():
        raise ValueError("external features contain duplicate date-instrument rows")
    return result.set_index(["datetime", "instrument"]).sort_index()


def align_external_features(
    index: pd.MultiIndex,
    source_data: Path | pd.DataFrame,
    *,
    prefixes: Sequence[str],
) -> pd.DataFrame:
    """Load PIT daily features and align them exactly to a Qlib dataset index."""
    if not {"datetime", "instrument"}.issubset(index.names):
        raise ValueError("external feature alignment requires datetime and instrument levels")
    dates = pd.to_datetime(index.get_level_values("datetime"), errors="coerce").normalize()
    if dates.hasnans:
        raise ValueError("external feature alignment received invalid dates")
    if (dates >= LOCKED_CONFIRMATION_START).any():
        raise ValueError("Goal 1 external feature alignment attempted to access 2026")

    requested_years = sorted(set(int(value) for value in dates.year))
    if isinstance(source_data, pd.DataFrame):
        source = _normalize_external_features(source_data)
        available_years = set(
            int(value)
            for value in source.index.get_level_values("datetime").year
        )
        source = source[
            source.index.get_level_values("datetime").year.isin(requested_years)
        ]
    else:
        daily_dir = Path(source_data) / "daily"
        files = sorted(daily_dir.glob("*.parquet"))
        if not files:
            raise ValueError("external daily features are unavailable")
        years_by_path: dict[int, Path] = {}
        for path in files:
            try:
                year = int(path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"external feature filename is not a year: {path.name}"
                ) from exc
            if year >= LOCKED_CONFIRMATION_START.year:
                raise ValueError("Goal 1 external features attempted to expose 2026 data")
            years_by_path[year] = path
        available_years = set(years_by_path)
        source = pd.concat(
            [
                _read_external_feature_year(str(years_by_path[year].resolve()))
                for year in requested_years
                if year in years_by_path
            ],
            axis=0,
        )
    missing_years = [year for year in requested_years if year not in available_years]
    if missing_years:
        raise ValueError(
            "external daily features are missing years: "
            + ", ".join(str(value) for value in missing_years)
        )
    normalized_prefixes = tuple(str(value).upper() for value in prefixes)
    selected = (
        [
            column
            for column in source.columns
            if str(column).upper().startswith(normalized_prefixes)
        ]
        if normalized_prefixes
        else list(source.columns)
    )
    if not selected:
        raise ValueError("external feature family selected no columns")

    canonical_index = pd.MultiIndex.from_arrays(
        [dates, index.get_level_values("instrument").astype(str).str.upper()],
        names=["datetime", "instrument"],
    )
    aligned = source.loc[:, selected].reindex(canonical_index)
    aligned.index = index
    return aligned


def normalize_cross_section(
    features: pd.DataFrame, *, groups: Sequence[Any]
) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    if not isinstance(features.index, pd.MultiIndex) or "datetime" not in features.index.names:
        raise ValueError("feature frame requires a datetime MultiIndex level")
    dates = pd.to_datetime(
        features.index.get_level_values("datetime"), errors="coerce"
    ).normalize()
    group_index = pd.Index(groups, dtype="object").fillna("unknown")
    if len(group_index) != len(features):
        raise ValueError("normalization groups must align with feature rows")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    keys = [dates, group_index]
    means = numeric.groupby(keys, sort=False).transform("mean")
    centered = numeric - means
    variances = centered.pow(2).groupby(keys, sort=False).transform("mean")
    scales = np.sqrt(variances).replace(0.0, np.nan)
    return (centered / scales).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_market_regime_features(
    features: pd.DataFrame, market_states: pd.Series
) -> pd.DataFrame:
    """Add PIT market-regime one-hot features to an already prepared frame."""
    if not isinstance(features.index, pd.MultiIndex) or "datetime" not in features.index.names:
        raise ValueError("market-regime features require a datetime MultiIndex level")
    states = pd.Series(market_states, copy=True)
    states.index = pd.to_datetime(states.index, errors="coerce").normalize()
    if states.index.hasnans or states.index.has_duplicates:
        raise ValueError("market states contain invalid or duplicate dates")
    states = states.astype("string")
    unknown = sorted(set(states.dropna().astype(str)) - set(MARKET_REGIMES))
    if unknown:
        raise ValueError(f"unknown market states: {', '.join(unknown)}")
    dates = pd.to_datetime(
        features.index.get_level_values("datetime"), errors="coerce"
    ).normalize()
    aligned = pd.Series(dates, index=features.index).map(states)
    if aligned.isna().any():
        raise ValueError("market states are missing for one or more feature dates")
    result = features.copy()
    for state, column in zip(MARKET_REGIMES, MARKET_REGIME_FEATURES, strict=True):
        result[column] = aligned.eq(state).astype(float).to_numpy()
    return result


@lru_cache(maxsize=8)
def _read_industry_intervals(path: str) -> pd.DataFrame:
    return _normalize_industry_intervals(pd.read_parquet(path))


def _normalize_industry_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "industry_code", "start_date", "end_date"}
    if not required.issubset(frame.columns):
        raise ValueError("industry intervals are missing required columns")
    result = frame.loc[:, sorted(required)].copy()
    result["symbol"] = result["symbol"].astype(str).str.upper()
    result["start_date"] = pd.to_datetime(result["start_date"], errors="coerce").dt.normalize()
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce").dt.normalize()
    if result[["start_date", "end_date"]].isna().any().any():
        raise ValueError("industry intervals contain invalid dates")
    return result.sort_values(["symbol", "start_date"], kind="stable")


def match_industry_intervals(
    index: pd.MultiIndex, source: Path | pd.DataFrame
) -> pd.Index:
    if not {"datetime", "instrument"}.issubset(index.names):
        raise ValueError("industry matching requires datetime and instrument levels")
    dates = pd.to_datetime(index.get_level_values("datetime"), errors="coerce").normalize()
    symbols = index.get_level_values("instrument").astype(str).str.upper()
    intervals = (
        _normalize_industry_intervals(source)
        if isinstance(source, pd.DataFrame)
        else _read_industry_intervals(str(Path(source).resolve()))
    )
    output = np.full(len(index), "unknown", dtype=object)
    positions_by_symbol: dict[str, list[int]] = {}
    for position, symbol in enumerate(symbols):
        positions_by_symbol.setdefault(str(symbol), []).append(position)
    for symbol, positions in positions_by_symbol.items():
        candidates = intervals[intervals["symbol"] == symbol]
        if candidates.empty:
            continue
        selected_dates = dates.take(positions)
        for row in candidates.itertuples(index=False):
            matched = (selected_dates >= row.start_date) & (selected_dates <= row.end_date)
            for position in np.asarray(positions)[matched]:
                output[int(position)] = str(row.industry_code)
    return pd.Index(output, name="industry")


def apply_excess_label(
    labels: pd.DataFrame, benchmark_returns: pd.Series
) -> pd.DataFrame:
    if labels.empty:
        return labels.copy()
    if not isinstance(labels.index, pd.MultiIndex) or "datetime" not in labels.index.names:
        raise ValueError("label frame requires a datetime MultiIndex level")
    benchmark = pd.to_numeric(benchmark_returns, errors="coerce").copy()
    benchmark.index = pd.to_datetime(benchmark.index, errors="coerce").normalize()
    dates = pd.to_datetime(labels.index.get_level_values("datetime"), errors="coerce").normalize()
    aligned = pd.Series(dates, index=labels.index).map(benchmark)
    if aligned.isna().any():
        raise ValueError("benchmark return is missing for one or more label dates")
    numeric = labels.apply(pd.to_numeric, errors="coerce")
    return numeric.sub(aligned, axis=0)


def build_signal_dataset_config(
    *,
    split: RollingSplit,
    signal: Mapping[str, Any],
    feature_prefixes: Sequence[str],
    benchmark_returns: Mapping[str, float],
    market_states: Mapping[str, str],
    industry_intervals_path: Path | None = None,
    industry_intervals: pd.DataFrame | None = None,
    include_handler_features: bool = True,
    external_features_path: Path | None = None,
    external_features: pd.DataFrame | None = None,
    external_feature_prefixes: Sequence[str] = (),
    handler: Any | None = None,
) -> dict[str, Any]:
    def date_string(value: Any) -> str:
        return str(pd.Timestamp(value).date())

    handler_config = (
        handler
        if handler is not None
        else build_signal_handler_config(
            split=split,
            signal=signal,
            include_handler_features=include_handler_features,
        )
    )
    return {
        "class": "SignalResearchDataset",
        "module_path": "robust_signal_dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": [date_string(split.train_start), date_string(split.train_end)],
                "valid": [
                    date_string(split.validation_start),
                    date_string(split.validation_end),
                ],
                "test": [date_string(split.test_start), date_string(split.test_end)],
            },
            "feature_prefixes": list(feature_prefixes),
            "include_handler_features": bool(include_handler_features),
            "external_features_path": (
                str(Path(external_features_path).resolve())
                if external_features_path is not None
                else None
            ),
            "external_features": external_features,
            "external_feature_prefixes": list(external_feature_prefixes),
            "normalization_scope": str(signal["normalization_scope"]),
            "industry_intervals_path": (
                str(Path(industry_intervals_path).resolve())
                if industry_intervals_path is not None
                else None
            ),
            "industry_intervals": industry_intervals,
            "target": str(signal["target"]),
            "benchmark_returns": dict(benchmark_returns),
            "market_states": dict(market_states),
        },
    }


def build_signal_handler_config(
    *,
    split: RollingSplit,
    signal: Mapping[str, Any],
    include_handler_features: bool = True,
) -> dict[str, Any]:
    horizon = int(signal["horizon_days"])
    expression = label_expression(horizon)

    def date_string(value: Any) -> str:
        return str(pd.Timestamp(value).date())

    if include_handler_features:
        return {
            "class": "Alpha158",
            "module_path": "qlib.contrib.data.handler",
            "kwargs": {
                "start_time": date_string(split.train_start),
                "end_time": date_string(split.test_end),
                "fit_start_time": date_string(split.train_start),
                "fit_end_time": date_string(split.train_end),
                "instruments": "csi800",
                "label": [[expression], ["LABEL0"]],
                "infer_processors": [
                    {
                        "class": "RobustZScoreNorm",
                        "kwargs": {
                            "fields_group": "feature",
                            "clip_outlier": True,
                        },
                    },
                    {"class": "Fillna", "kwargs": {"fields_group": "feature"}},
                ],
                "learn_processors": [{"class": "DropnaLabel"}],
            },
        }

    return {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": date_string(split.train_start),
            "end_time": date_string(split.test_end),
            "instruments": "csi800",
            "data_loader": {
                "class": "QlibDataLoader",
                "kwargs": {
                    "config": {
                        "feature": [["$close"], ["ANCHOR_CLOSE"]],
                        "label": [[expression], ["LABEL0"]],
                    },
                    "freq": "day",
                },
            },
            "infer_processors": [],
            "learn_processors": [{"class": "DropnaLabel"}],
        },
    }


def _split_column_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(frame.columns, pd.MultiIndex):
        return frame.copy(), pd.DataFrame(index=frame.index)
    feature = (
        frame.xs("feature", axis=1, level=0, drop_level=True)
        if "feature" in frame.columns.get_level_values(0)
        else pd.DataFrame(index=frame.index)
    )
    other = frame.loc[:, frame.columns.get_level_values(0) != "feature"].copy()
    return feature, other


class SignalResearchDataset(DatasetH):
    """DatasetH that applies the frozen feature family and PIT group normalization."""

    def __init__(
        self,
        *args: Any,
        feature_prefixes: Sequence[str] = (),
        include_handler_features: bool = True,
        external_features_path: str | None = None,
        external_features: pd.DataFrame | None = None,
        external_feature_prefixes: Sequence[str] = (),
        normalization_scope: str = "board",
        industry_intervals_path: str | None = None,
        industry_intervals: pd.DataFrame | None = None,
        target: str = "raw_return",
        benchmark_returns: Mapping[str, float] | None = None,
        market_states: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.feature_prefixes = tuple(str(value) for value in feature_prefixes)
        self.include_handler_features = bool(include_handler_features)
        self.external_features_path = (
            Path(external_features_path) if external_features_path else None
        )
        self.external_features = external_features
        self.external_feature_prefixes = tuple(
            str(value) for value in external_feature_prefixes
        )
        self.normalization_scope = str(normalization_scope)
        self.industry_intervals_path = (
            Path(industry_intervals_path) if industry_intervals_path else None
        )
        self.industry_intervals = industry_intervals
        self.target = str(target)
        self.benchmark_returns = pd.Series(dict(benchmark_returns or {}), dtype=float)
        self.benchmark_returns.index = pd.to_datetime(
            self.benchmark_returns.index, errors="coerce"
        ).normalize()
        if self.benchmark_returns.index.hasnans:
            raise ValueError("benchmark_returns contains invalid dates")
        self.market_states = pd.Series(dict(market_states or {}), dtype="string")
        self.market_states.index = pd.to_datetime(
            self.market_states.index, errors="coerce"
        ).normalize()
        if self.market_states.index.hasnans or self.market_states.index.has_duplicates:
            raise ValueError("market_states contains invalid or duplicate dates")
        super().__init__(*args, **kwargs)

    def _select_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        if getattr(self, "include_handler_features", True):
            blocks.append(select_feature_family(frame, prefixes=self.feature_prefixes))
        external_path = getattr(self, "external_features_path", None)
        external_features = getattr(self, "external_features", None)
        if external_features is not None:
            blocks.append(
                align_external_features(
                    frame.index,
                    external_features,
                    prefixes=getattr(self, "external_feature_prefixes", ()),
                )
            )
        elif external_path is not None:
            blocks.append(
                align_external_features(
                    frame.index,
                    external_path,
                    prefixes=getattr(self, "external_feature_prefixes", ()),
                )
            )
        if not blocks:
            raise ValueError("signal dataset selected no handler or external features")
        selected = pd.concat(blocks, axis=1)
        if selected.columns.duplicated().any():
            raise ValueError("signal dataset contains duplicate feature columns")
        return selected

    def _transform_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        selected = self._select_features(frame)
        instruments = selected.index.get_level_values("instrument").astype(str)
        if self.normalization_scope == "industry":
            industry_intervals = getattr(self, "industry_intervals", None)
            if industry_intervals is not None:
                groups = match_industry_intervals(selected.index, industry_intervals)
            elif self.industry_intervals_path is not None:
                groups = match_industry_intervals(
                    selected.index, self.industry_intervals_path
                )
            else:
                raise ValueError("industry normalization requires interval metadata")
        elif self.normalization_scope == "board":
            groups = pd.Index([board_from_symbol(value) for value in instruments])
        else:
            raise ValueError("normalization_scope must be board or industry")
        normalized = normalize_cross_section(selected, groups=groups)
        boards = pd.Index([board_from_symbol(value) for value in instruments])
        for board in BOARD_NAMES:
            normalized[f"BOARD_{board.upper()}"] = (boards == board).astype(float)
        return add_market_regime_features(normalized, self.market_states)

    def prepare_raw_features(self, segments: Any) -> pd.DataFrame | list[pd.DataFrame]:
        """Return selected handler features before infer processors or normalization."""
        prepared = super().prepare(segments, col_set="feature", data_key="raw")

        def transform(frame: pd.DataFrame) -> pd.DataFrame:
            if isinstance(frame.columns, pd.MultiIndex):
                frame, _ = _split_column_groups(frame)
            return self._select_features(frame)

        if isinstance(prepared, list):
            return [transform(frame) for frame in prepared]
        return transform(prepared)

    def prepare(
        self,
        segments: Any,
        col_set: Any = "__all",
        data_key: str = "infer",
        **kwargs: Any,
    ):
        prepared = super().prepare(segments, col_set=col_set, data_key=data_key, **kwargs)

        def transform(frame: pd.DataFrame) -> pd.DataFrame:
            if frame.empty:
                return frame.copy()
            if isinstance(frame.columns, pd.MultiIndex):
                feature, other = _split_column_groups(frame)
                blocks: list[pd.DataFrame] = []
                if not feature.empty:
                    normalized = self._transform_features(feature)
                    normalized.columns = pd.MultiIndex.from_tuples(
                        [("feature", str(column)) for column in normalized.columns]
                    )
                    blocks.append(normalized)
                if not other.empty:
                    if self.target == "csi800_excess" and "label" in other.columns.get_level_values(0):
                        label = other.xs("label", axis=1, level=0, drop_level=True)
                        label = apply_excess_label(label, self.benchmark_returns)
                        label.columns = pd.MultiIndex.from_tuples(
                            [("label", str(column)) for column in label.columns]
                        )
                        other = pd.concat(
                            [other.loc[:, other.columns.get_level_values(0) != "label"], label],
                            axis=1,
                        )
                    blocks.append(other)
                return pd.concat(blocks, axis=1)
            feature_only = col_set == "feature" or col_set == ["feature"] or col_set == ("feature",)
            label_only = col_set == "label" or col_set == ["label"] or col_set == ("label",)
            if feature_only:
                return self._transform_features(frame)
            if label_only and self.target == "csi800_excess":
                return apply_excess_label(frame, self.benchmark_returns)
            return frame.copy()

        if isinstance(prepared, list):
            return [transform(frame) for frame in prepared]
        return transform(prepared)
