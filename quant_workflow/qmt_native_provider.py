"""Qlib storage for one verified frozen QMT experiment input."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.data import CalendarProvider, FeatureProvider, InstrumentProvider
from quant_qmt_data.dataset import load_dataset


class DirectProviderError(ValueError, RuntimeError):
    """Input errors preserve native ValueError and historical RuntimeError handling."""


@dataclass
class _FrozenInput:
    frame: pd.DataFrame
    calendar: pd.DatetimeIndex
    instruments: dict
    bars: dict


@lru_cache(maxsize=4)
def _load_input(input_dir: str) -> _FrozenInput:
    frame = load_dataset(input_dir)
    metadata = frame.attrs
    calendar = pd.DatetimeIndex(metadata["calendar"])
    if calendar.empty or not calendar.is_monotonic_increasing or calendar.has_duplicates:
        raise DirectProviderError("Frozen calendar must be nonempty, ordered and unique")
    if frame.duplicated(["symbol", "date"]).any():
        raise DirectProviderError("Frozen bars contain duplicate symbol/date rows")
    if not pd.DatetimeIndex(frame["date"]).isin(calendar).all():
        raise DirectProviderError("Frozen bars contain dates outside the experiment calendar")
    intervals = metadata.get("membership_intervals") if metadata["universe_kind"] == "historical" else None
    instruments = {}
    for symbol in metadata["universe"]:
        if intervals is None:
            spans = [(calendar[0], calendar[-1])]
        else:
            spans = []
        instruments[symbol] = spans
    if intervals is not None:
        for row in intervals:
            instruments[row["symbol"]].append((pd.Timestamp(row["start_date"]), pd.Timestamp(row["end_date"])))
    bars = {symbol: group.set_index("date").sort_index()
            for symbol, group in frame.groupby("symbol", sort=False)}
    return _FrozenInput(frame, calendar, instruments, bars)


class QMTNativeCalendarProvider(CalendarProvider):
    def __init__(self, *, input_dir):
        super().__init__()
        self.input = _load_input(str(Path(input_dir).resolve()))

    def load_calendar(self, freq, future):
        if freq not in ("day", "1d"):
            raise DirectProviderError("QMT frozen inputs support daily data only")
        return list(self.input.calendar)


class QMTNativeInstrumentProvider(InstrumentProvider):
    def __init__(self, *, input_dir):
        super().__init__()
        self.input = _load_input(str(Path(input_dir).resolve()))

    def list_instruments(self, instruments, start_time=None, end_time=None, freq="day", as_list=False):
        if freq not in ("day", "1d"):
            raise DirectProviderError("QMT frozen inputs support daily data only")
        if instruments.get("market", "all") != "all":
            raise DirectProviderError("Frozen input has only its selected pool; historical CSI800 is unavailable")
        result = self.input.instruments
        for filter_config in instruments.get("filter_pipe", []):
            from qlib.utils import init_instance_by_config
            filter_type = filter_config["filter_type"]
            kwargs = {key: value for key, value in filter_config.items() if key != "filter_type"}
            result = init_instance_by_config({"class": filter_type, "module_path": "qlib.data.filter",
                                              "kwargs": kwargs})(result, start_time, end_time, freq)
        start = pd.Timestamp(start_time) if start_time is not None else self.input.calendar[0]
        end = pd.Timestamp(end_time) if end_time is not None else self.input.calendar[-1]
        result = {symbol: [(max(left, start), min(right, end)) for left, right in spans
                           if max(left, start) <= min(right, end)]
                  for symbol, spans in result.items()}
        result = {symbol: spans for symbol, spans in result.items() if spans}
        return list(result) if as_list else result


class QMTNativeFeatureProvider(FeatureProvider):
    def __init__(self, *, input_dir):
        super().__init__()
        self.input = _load_input(str(Path(input_dir).resolve()))

    def feature(self, instrument, field, start_time, end_time, freq):
        if freq not in ("day", "1d"):
            raise DirectProviderError("QMT frozen inputs support daily data only")
        name = field.lstrip("$")
        if name not in ("open", "high", "low", "close", "volume", "amount", "preclose", "suspend_flag", "factor"):
            raise DirectProviderError(f"Unsupported QMT frozen field: {field}")
        start = max(0, int(start_time))
        end = min(len(self.input.calendar) - 1, int(end_time))
        dates = self.input.calendar[start:end + 1]
        bars = self.input.bars.get(instrument)
        if bars is None:
            values = np.full(len(dates), np.nan)
        elif name == "factor":
            # Prices are already in the saved research basis: never adjust twice.
            values = np.where(dates.isin(bars.index), 1.0, np.nan)
        else:
            values = bars[name].reindex(dates).to_numpy(dtype=float)
        return pd.Series(values, index=pd.RangeIndex(start, end + 1), dtype=float)


class QMTNativeOHLCV(Alpha158):
    """Qlib Alpha158-style OHLCV features, excluding the unavailable VWAP input."""
    @staticmethod
    def get_feature_config():
        return Alpha158DL.get_feature_config({
            "kbar": {}, "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW"]},
            "rolling": {},
        })


def _clear_context_cache() -> None:
    _load_input.cache_clear()


def qlib_direct_init_config(*, input_dir: str | Path, kernels: int | None = None) -> dict:
    directory = str(Path(input_dir).expanduser().resolve())
    provider = {"module_path": "quant_workflow.qmt_native_provider", "kwargs": {"input_dir": directory}}
    return {
        "provider_uri": directory, "region": "cn", "provider": "LocalProvider",
        "calendar_provider": {**provider, "class": "QMTNativeCalendarProvider"},
        "instrument_provider": {**provider, "class": "QMTNativeInstrumentProvider"},
        "feature_provider": {**provider, "class": "QMTNativeFeatureProvider"},
        "expression_provider": {"class": "LocalExpressionProvider", "module_path": "qlib.data.data", "kwargs": {"time2idx": True}},
        "dataset_provider": {"class": "LocalDatasetProvider", "module_path": "qlib.data.data", "kwargs": {"align_time": True}},
        "pit_provider": None, "calendar_cache": None, "expression_cache": None,
        "dataset_cache": None, "default_disk_cache": 0,
        "kernels": kernels if kernels is not None else 1, "joblib_backend": "threading",
    }


def init_qlib_from_quant_data(*, input_dir: str | Path, kernels: int | None = None, **overrides) -> None:
    import qlib
    _clear_context_cache()
    config = qlib_direct_init_config(input_dir=input_dir, kernels=kernels)
    config.update(overrides)
    qlib.init(default_conf="client", **config)
