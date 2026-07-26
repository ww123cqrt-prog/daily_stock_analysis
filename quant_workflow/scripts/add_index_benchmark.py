#!/usr/bin/env python3
"""Download CSI300 index data and add benchmark features without polluting stock pools."""

from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
import uuid
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import write_json_atomic  # noqa: E402
from _paths import qlib_provider_uri, quant_data_root, reports_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
DATA_DIR = qlib_provider_uri()
BENCHMARK_SYMBOL = "SH000300"
BENCHMARK_FEATURE_FIELDS = ("close", "volume", "factor", "open", "high", "low")


class BenchmarkSourceError(RuntimeError):
    """No source could provide a complete, real benchmark series."""


class BenchmarkPaths:
    def __init__(self, provider_dir: Path):
        self.provider_dir = provider_dir.expanduser()
        self.calendar_file = self.provider_dir / "calendars" / "day.txt"
        self.instruments_file = self.provider_dir / "instruments" / "all.txt"
        self.csi300_file = self.provider_dir / "instruments" / "csi300.txt"
        self.feature_dir = self.provider_dir / "features" / "sh000300"


def _load_baostock():
    import baostock as bs

    return bs


def write_feature_bin(field_path: Path, start_idx: int, values) -> None:
    field_path.parent.mkdir(parents=True, exist_ok=True)
    with field_path.open("wb") as handle:
        np.array([start_idx], dtype="<f4").tofile(handle)
        np.asarray(values, dtype="<f4").tofile(handle)


def read_feature_bin(field_path: Path) -> tuple[int, np.ndarray]:
    raw = np.fromfile(field_path, dtype="<f4")
    if len(raw) < 2:
        raise ValueError(f"invalid Qlib feature bin: {field_path}")
    return int(raw[0]), raw[1:]


def _remove_symbol_if_present(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    lines = [line for line in path.read_text().splitlines() if not line.startswith(f"{symbol}\t")]
    path.write_text("\n".join(lines) + "\n")


def _read_calendar(path: Path) -> list[str]:
    calendar = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not calendar:
        raise ValueError(f"empty Qlib calendar: {path}")
    return calendar


def _download_index_data(baostock_module, start_date: str, end_date: str) -> dict[str, float]:
    lg = baostock_module.login()
    if getattr(lg, "error_code", "0") not in ("0", 0):
        raise RuntimeError(f"baostock login failed: {getattr(lg, 'error_msg', '')}")
    try:
        rs = baostock_module.query_history_k_data_plus(
            "sh.000300",
            "date,close",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3",
        )
        data: dict[str, float] = {}
        while rs.next():
            row = rs.get_row_data()
            if row[1]:
                data[row[0]] = float(row[1])
        return data
    finally:
        baostock_module.logout()


def _download_eastmoney_index(start_date: str, end_date: str, *, use_system_proxy: bool = False) -> dict[str, float]:
    import requests

    session = requests.Session()
    session.trust_env = use_system_proxy
    try:
        response = session.get(
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            params={
                "secid": "1.000300",
                "fields1": "f1,f2,f3,f4,f5,f6",
                "fields2": "f51,f52,f53",
                "klt": "101",
                "fqt": "0",
                "beg": start_date.replace("-", ""),
                "end": end_date.replace("-", ""),
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    finally:
        session.close()
    klines = ((payload or {}).get("data") or {}).get("klines") or []
    data: dict[str, float] = {}
    for line in klines:
        parts = str(line).split(",")
        if len(parts) >= 3 and parts[2] not in {"", "-"}:
            data[parts[0]] = float(parts[2])
    return data


def _download_tencent_index(start_date: str, end_date: str, *, use_system_proxy: bool = False) -> dict[str, float]:
    import requests

    session = requests.Session()
    session.trust_env = use_system_proxy
    data: dict[str, float] = {}
    cursor_end = end_date
    try:
        for _ in range(64):
            response = session.get(
                "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                params={"param": f"sh000300,day,,{cursor_end},640,qfq"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            instrument = ((payload or {}).get("data") or {}).get("sh000300") or {}
            rows = instrument.get("day") or instrument.get("qfqday") or []
            if not rows:
                break

            page_dates: list[str] = []
            for row in rows:
                if len(row) < 3:
                    continue
                day = str(row[0])[:10]
                page_dates.append(day)
                if start_date <= day <= end_date and row[2] not in {"", "-"}:
                    data[day] = float(row[2])

            if not page_dates:
                break
            earliest = min(page_dates)
            if earliest <= start_date:
                break
            next_end = (datetime.strptime(earliest, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()
            if next_end >= cursor_end:
                break
            cursor_end = next_end
    finally:
        session.close()
    return data


def _download_akshare_index(start_date: str, end_date: str) -> dict[str, float]:
    import akshare as ak

    frame = ak.index_zh_a_hist(
        symbol="000300",
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if frame is None or frame.empty:
        return {}
    date_column = "日期" if "日期" in frame.columns else "date"
    close_column = "收盘" if "收盘" in frame.columns else "close"
    return {
        str(value)[:10]: float(close)
        for value, close in zip(
            np.asarray(frame[date_column], dtype="datetime64[D]"),
            frame[close_column],
        )
        if math.isfinite(float(close)) and float(close) > 0
    }


def _safe_error(exc: Exception) -> str:
    text = re.sub(r"https?://[^\s?]+\?[^\s]+", "<url-redacted>", str(exc))
    return text[:200]


def _validate_source_data(data: dict[str, float], calendar: list[str]) -> tuple[dict[str, float], int]:
    normalized: dict[str, float] = {}
    calendar_set = set(calendar)
    for raw_day, raw_close in data.items():
        day = str(raw_day)[:10]
        try:
            close = float(raw_close)
        except (TypeError, ValueError):
            continue
        if day in calendar_set and math.isfinite(close) and close > 0:
            normalized[day] = close
    if not normalized:
        raise ValueError("source returned no positive benchmark closes on the provider calendar")

    first_idx = min(calendar.index(day) for day in normalized)
    required = calendar[first_idx:]
    missing = [day for day in required if day not in normalized]
    if missing:
        raise ValueError(
            f"source benchmark coverage is incomplete after {calendar[first_idx]}: "
            f"missing {len(missing)} dates; first={missing[0]} last={missing[-1]}"
        )
    return normalized, first_idx


def _cache_data(provider: Path) -> dict[str, float]:
    calendar = _read_calendar(provider / "calendars" / "day.txt")
    start_idx, values = read_feature_bin(provider / "features" / "sh000300" / "close.day.bin")
    if start_idx < 0 or start_idx + len(values) > len(calendar):
        raise ValueError(f"cached benchmark bin is outside its calendar: {provider}")
    return {
        calendar[start_idx + offset]: float(value)
        for offset, value in enumerate(values)
        if math.isfinite(float(value)) and float(value) > 0
    }


def _default_cache_providers(provider_dir: Path) -> list[Path]:
    providers: list[Path] = []
    seen: set[Path] = set()
    for search_root in (QUANT_DATA / "qlib", QUANT_DATA / "runtime"):
        if not search_root.exists():
            continue
        for feature in search_root.rglob("features/sh000300/close.day.bin"):
            candidate = feature.parents[2]
            if candidate == provider_dir or candidate in seen:
                continue
            seen.add(candidate)
            providers.append(candidate)
    return sorted(providers)


def _write_benchmark_features(paths: BenchmarkPaths, calendar: list[str], data: dict[str, float]) -> None:
    close_arr = np.full(len(calendar), np.nan, dtype="<f4")
    for index, day in enumerate(calendar):
        if day in data:
            close_arr[index] = data[day]

    staging = paths.feature_dir.with_name(f".{paths.feature_dir.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        values_by_field = {
            "close": close_arr,
            "volume": np.zeros(len(calendar), dtype="<f4"),
            "factor": np.ones(len(calendar), dtype="<f4"),
            "open": close_arr,
            "high": close_arr,
            "low": close_arr,
        }
        for field in BENCHMARK_FEATURE_FIELDS:
            write_feature_bin(staging / f"{field}.day.bin", 0, values_by_field[field])
        paths.feature_dir.mkdir(parents=True, exist_ok=True)
        for field in BENCHMARK_FEATURE_FIELDS:
            (staging / f"{field}.day.bin").replace(paths.feature_dir / f"{field}.day.bin")
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def add_benchmark(
    provider_dir: Path = DATA_DIR,
    *,
    dry_run: bool = False,
    baostock_module=None,
    source_fetchers: list[tuple[str, Callable[[str, str], dict[str, float]]]] | None = None,
    cache_providers: list[Path] | None = None,
    status_dir: Path | None = None,
    use_system_proxy: bool = False,
    offline: bool = False,
) -> dict[str, object]:
    paths = BenchmarkPaths(provider_dir)
    try:
        calendar = _read_calendar(paths.calendar_file)
    except (FileNotFoundError, ValueError) as exc:
        if not dry_run:
            raise
        calendar = []
        print(f"DRY RUN: Qlib calendar unavailable: {exc}")
    if calendar:
        print(f"Calendar: {len(calendar)} trading days ({calendar[0]} ~ {calendar[-1]})")
    else:
        print("Calendar: 0 trading days (unavailable)")
    print(f"Provider: {paths.provider_dir}")
    print(f"Benchmark feature dir: {paths.feature_dir}")

    if dry_run:
        print("Dry run OK. No baostock request, bin write, or instruments rewrite will be performed.")
        return {
            "status": "dry-run",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "benchmark_symbol": BENCHMARK_SYMBOL,
        }

    attempts: list[dict[str, str]] = []
    if source_fetchers is None:
        source_fetchers = []
        if not offline:
            bs = _load_baostock() if baostock_module is None else baostock_module
            source_fetchers.extend(
                [
                    ("baostock", lambda start, end: _download_index_data(bs, start, end)),
                    (
                        "tencent",
                        lambda start, end: _download_tencent_index(
                            start,
                            end,
                            use_system_proxy=use_system_proxy,
                        ),
                    ),
                    (
                        "eastmoney",
                        lambda start, end: _download_eastmoney_index(
                            start,
                            end,
                            use_system_proxy=use_system_proxy,
                        ),
                    ),
                    ("akshare", _download_akshare_index),
                ]
            )
    cache_candidates = _default_cache_providers(paths.provider_dir) if cache_providers is None else cache_providers

    selected_source = ""
    selected_data: dict[str, float] | None = None
    first_observation_idx = 0
    for source_name, fetcher in source_fetchers:
        try:
            raw = fetcher(calendar[0], calendar[-1])
            selected_data, first_observation_idx = _validate_source_data(raw, calendar)
            selected_source = source_name
            attempts.append({"source": source_name, "status": "ok"})
            break
        except Exception as exc:
            attempts.append({"source": source_name, "status": "failed", "error": _safe_error(exc)})

    if selected_data is None:
        for cache_provider in cache_candidates:
            source_name = f"local_cache:{cache_provider}"
            try:
                selected_data, first_observation_idx = _validate_source_data(_cache_data(cache_provider), calendar)
                selected_source = source_name
                attempts.append({"source": source_name, "status": "ok"})
                break
            except Exception as exc:
                attempts.append({"source": source_name, "status": "failed", "error": _safe_error(exc)})

    status_root = Path(status_dir) if status_dir is not None else reports_root() / "runtime" / "status"
    if selected_data is None:
        payload = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "failed",
            "benchmark_symbol": BENCHMARK_SYMBOL,
            "provider_dir": str(paths.provider_dir),
            "attempts": attempts,
        }
        write_json_atomic(status_root / "benchmark_update_latest.json", payload)
        raise BenchmarkSourceError("all benchmark sources failed validation")

    print(f"Validated {len(selected_data)} SH000300 days from {selected_source}")
    _write_benchmark_features(paths, calendar, selected_data)
    print(f"Written bin files to {paths.feature_dir}")

    # Keep benchmark data in features, but never add it to training stock pools.
    _remove_symbol_if_present(paths.instruments_file, BENCHMARK_SYMBOL)
    _remove_symbol_if_present(paths.csi300_file, BENCHMARK_SYMBOL)
    print("Ensured SH000300 is not present in all.txt or csi300.txt")

    print("Done!")
    result = {
        "status": "completed",
        "provider_dir": str(paths.provider_dir),
        "calendar_days": len(calendar),
        "downloaded_days": len(selected_data),
        "first_observation": calendar[first_observation_idx],
        "benchmark_symbol": BENCHMARK_SYMBOL,
        "source": selected_source,
        "attempts": attempts,
    }
    write_json_atomic(
        status_root / "benchmark_update_latest.json",
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            **result,
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Add SH000300 benchmark features to a Qlib provider.")
    parser.add_argument("--provider-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate paths and calendar without network or writes.")
    parser.add_argument("--cache-provider", type=Path, action="append", default=None)
    parser.add_argument("--status-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true", help="Use validated local benchmark caches only.")
    parser.add_argument("--use-system-proxy", action="store_true")
    args = parser.parse_args(argv)
    try:
        add_benchmark(
            args.provider_dir,
            dry_run=args.dry_run,
            cache_providers=args.cache_provider,
            status_dir=args.status_dir,
            use_system_proxy=args.use_system_proxy,
            offline=args.offline,
        )
    except BenchmarkSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
