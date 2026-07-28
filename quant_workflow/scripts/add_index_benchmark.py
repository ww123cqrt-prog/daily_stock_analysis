#!/usr/bin/env python3
"""Download CSI300/CSI500/CSI800 data without polluting stock pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _daily_update_runtime import write_json_atomic  # noqa: E402
from _benchmark_sources import (  # noqa: E402
    TENCENT_BENCHMARKS,
    download_tencent_benchmarks as _download_tencent_benchmarks,
)
from _paths import (  # noqa: E402
    qlib_compatibility_provider_uri,
    quant_data_root,
    reports_root,
    workflow_root,
)
from _file_lock import lock_file, unlock_file  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
DATA_DIR = qlib_compatibility_provider_uri()
BENCHMARKS = {
    "SH000300": "sh.000300",
    "SH000905": "sh.000905",
    "SH000906": "sh.000906",
}
BENCHMARK_SYMBOL = "SH000300"
BENCHMARK_TRANSACTION_FILENAME = ".add_index_benchmark.transaction.json"
BENCHMARK_LOCK_FILENAME = ".add_index_benchmark.lock"
BENCHMARK_TRANSACTION_VERSION = 1
BENCHMARK_FIELDS = ("open", "high", "low", "close", "volume", "factor")
BENCHMARK_FEATURE_FIELDS = ("close", "volume", "factor", "open", "high", "low")


class BenchmarkSourceError(RuntimeError):
    """No source could provide a complete, real CSI300 close series."""


class BenchmarkPaths:
    def __init__(self, provider_dir: Path):
        self.provider_dir = provider_dir.expanduser()
        self.calendar_file = self.provider_dir / "calendars" / "day.txt"
        self.instruments_file = self.provider_dir / "instruments" / "all.txt"
        self.instruments_dir = self.provider_dir / "instruments"

    def feature_dir(self, symbol: str) -> Path:
        return self.provider_dir / "features" / symbol.lower()


def _load_baostock():
    import baostock as bs

    return bs


def _write_bin(field_path: Path, start_idx: int, values: np.ndarray) -> None:
    field_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = field_path.with_name(f".{field_path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.array([start_idx], dtype="<f4").tofile(handle)
        values.astype("<f4").tofile(handle)
    os.replace(temporary, field_path)


def _remove_symbol_if_present(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    lines = [
        line
        for line in path.read_text().splitlines()
        if not line.startswith(f"{symbol}\t")
    ]
    path.write_text("\n".join(lines) + "\n")


def _read_calendar(path: Path) -> list[str]:
    calendar = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not calendar:
        raise ValueError(f"empty Qlib calendar: {path}")
    return calendar


def _read_qlib_bin(path: Path) -> tuple[int, np.ndarray]:
    try:
        raw = np.fromfile(path, dtype="<f4")
    except OSError as exc:
        raise RuntimeError(f"existing benchmark bin is unreadable: {path.name}") from exc
    if len(raw) < 2 or not np.isfinite(raw[0]) or raw[0] != int(raw[0]):
        raise RuntimeError(f"existing benchmark bin is invalid: {path.name}")
    return int(raw[0]), raw[1:]


def _validate_existing_benchmarks(
    paths: BenchmarkPaths,
    calendar: list[str],
    required_start_date: str,
    *,
    allow_incomplete_transaction: bool = False,
) -> dict[str, int]:
    if (
        not allow_incomplete_transaction
        and (paths.provider_dir / BENCHMARK_TRANSACTION_FILENAME).exists()
    ):
        raise RuntimeError("benchmark transaction is incomplete")
    calendar_index = pd.Index(calendar)
    parsed_calendar = pd.to_datetime(calendar_index, errors="raise")
    if parsed_calendar.has_duplicates or not parsed_calendar.is_monotonic_increasing:
        raise RuntimeError("existing benchmark calendar is invalid")
    required_start = max(parsed_calendar[0], pd.Timestamp(required_start_date))
    required_positions = np.flatnonzero(parsed_calendar >= required_start)
    if not len(required_positions):
        raise RuntimeError("existing benchmark calendar has no required dates")
    first_required = int(required_positions[0])
    available_days: dict[str, int] = {}
    for symbol in BENCHMARKS:
        feature_dir = paths.feature_dir(symbol)
        for field in BENCHMARK_FIELDS:
            start_index, values = _read_qlib_bin(
                feature_dir / f"{field}.day.bin"
            )
            end_index = start_index + len(values)
            if start_index > first_required or end_index < len(calendar):
                raise RuntimeError(
                    f"existing benchmark {symbol} does not cover required dates"
                )
            required = values[first_required - start_index : len(calendar) - start_index]
            if not len(required) or not np.isfinite(required).all():
                raise RuntimeError(
                    f"existing benchmark {symbol} contains invalid {field} values"
                )
            if field in {"open", "high", "low", "close", "factor"}:
                valid = (required > 0).all()
            else:
                valid = (required >= 0).all()
            if not valid:
                raise RuntimeError(
                    f"existing benchmark {symbol} contains invalid {field} values"
                )
        available_days[symbol] = len(calendar) - first_required
    if paths.instruments_dir.is_symlink() or not paths.instruments_dir.is_dir():
        raise RuntimeError("existing benchmark instrument pools are unavailable")
    instrument_files = sorted(paths.instruments_dir.glob("*.txt"))
    if not instrument_files:
        raise RuntimeError("existing benchmark instrument pools are unavailable")
    for instrument_file in instrument_files:
        instruments = {
            line.split("\t", 1)[0].strip().upper()
            for line in instrument_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        polluted = sorted(set(BENCHMARKS).intersection(instruments))
        if polluted:
            raise RuntimeError(
                "existing benchmark symbols pollute stock instrument pools: "
                + ",".join(polluted)
            )
    return available_days


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError("benchmark feature tree must not contain symlinks")
        if path.is_dir():
            directories.append(path)
            continue
        if not path.is_file():
            raise RuntimeError("benchmark feature tree contains a non-regular file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        _fsync_directory(directory)


def _validate_feature_tree(feature_dir: Path) -> None:
    if feature_dir.is_symlink() or not feature_dir.is_dir():
        raise RuntimeError("benchmark feature directory is invalid")
    expected = {f"{field}.day.bin" for field in BENCHMARK_FIELDS}
    actual: set[str] = set()
    for path in feature_dir.iterdir():
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("benchmark feature directory contains invalid entries")
        actual.add(path.name)
    if actual != expected:
        raise RuntimeError("benchmark feature directory has an unexpected schema")


def _regular_file_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("benchmark provider contains an invalid file")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return digest.hexdigest()


def _feature_tree_fingerprint(feature_dir: Path) -> str:
    _validate_feature_tree(feature_dir)
    digest = hashlib.sha256()
    for name in sorted(f"{field}.day.bin" for field in BENCHMARK_FIELDS):
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(_regular_file_sha256(feature_dir / name)))
    return digest.hexdigest()


def _source_provider_fingerprint(paths: BenchmarkPaths) -> str:
    digest = hashlib.sha256()
    files = [
        paths.calendar_file,
        *sorted(paths.instruments_dir.glob("*.txt"), key=lambda value: value.name),
    ]
    for path in files:
        relative = path.relative_to(paths.provider_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_regular_file_sha256(path)))
    for symbol in BENCHMARKS:
        encoded = symbol.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(
            bytes.fromhex(_feature_tree_fingerprint(paths.feature_dir(symbol)))
        )
    return digest.hexdigest()


def _write_transaction_journal(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _transaction_entries(
    paths: BenchmarkPaths, transaction_id: str
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for symbol in BENCHMARKS:
        target = paths.feature_dir(symbol)
        entries.append(
            {
                "symbol": symbol,
                "target": str(target.resolve()),
                "stage": str(
                    target.with_name(f".{target.name}.{transaction_id}.stage").resolve()
                ),
                "backup": str(
                    target.with_name(f".{target.name}.{transaction_id}.backup").resolve()
                ),
                "had_original": target.exists(),
            }
        )
    return entries


def _validated_journal_entries(
    paths: BenchmarkPaths, payload: object
) -> tuple[str, list[dict[str, object]]]:
    if (
        not isinstance(payload, dict)
        or payload.get("version") != BENCHMARK_TRANSACTION_VERSION
        or payload.get("state") not in {"prepared", "committed"}
        or not isinstance(payload.get("transaction_id"), str)
        or re.fullmatch(r"[0-9a-f]{32}", str(payload["transaction_id"])) is None
        or not isinstance(payload.get("entries"), list)
    ):
        raise RuntimeError("benchmark transaction journal is invalid")
    transaction_id = str(payload["transaction_id"])
    expected = {
        str(entry["symbol"]): entry
        for entry in _transaction_entries(paths, transaction_id)
    }
    raw_entries = payload["entries"]
    if len(raw_entries) != len(expected):
        raise RuntimeError("benchmark transaction journal is invalid")
    entries: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict) or type(raw.get("had_original")) is not bool:
            raise RuntimeError("benchmark transaction journal is invalid")
        symbol = raw.get("symbol")
        expected_entry = expected.get(str(symbol))
        if expected_entry is None or str(symbol) in seen:
            raise RuntimeError("benchmark transaction journal is invalid")
        if any(
            raw.get(key) != expected_entry[key]
            for key in ("target", "stage", "backup")
        ):
            raise RuntimeError("benchmark transaction journal is invalid")
        entries.append(dict(raw))
        seen.add(str(symbol))
    return str(payload["state"]), entries


def _cleanup_transaction(entries: list[dict[str, object]], journal: Path) -> None:
    for entry in entries:
        _remove_path(Path(str(entry["stage"])))
        _remove_path(Path(str(entry["backup"])))
    journal.unlink(missing_ok=True)
    _fsync_directory(journal.parent)


def _recover_transaction(paths: BenchmarkPaths) -> None:
    journal = paths.provider_dir / BENCHMARK_TRANSACTION_FILENAME
    try:
        mode = journal.lstat().st_mode
    except FileNotFoundError:
        return
    if journal.is_symlink() or not stat.S_ISREG(mode):
        raise RuntimeError("benchmark transaction journal is invalid")
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("benchmark transaction journal is invalid") from exc
    state, entries = _validated_journal_entries(paths, payload)
    if state == "prepared":
        for entry in reversed(entries):
            target = Path(str(entry["target"]))
            backup = Path(str(entry["backup"]))
            if bool(entry["had_original"]):
                if backup.is_dir() and not backup.is_symlink():
                    _remove_path(target)
                    os.replace(backup, target)
                elif not (target.is_dir() and not target.is_symlink()):
                    raise RuntimeError("benchmark transaction recovery failed")
            else:
                _remove_path(target)
    else:
        for entry in entries:
            target = Path(str(entry["target"]))
            if target.is_symlink() or not target.is_dir():
                raise RuntimeError("benchmark transaction recovery failed")
    _cleanup_transaction(entries, journal)


def _acquire_benchmark_lock(paths: BenchmarkPaths):
    paths.provider_dir.parent.mkdir(parents=True, exist_ok=True)
    handle = (
        paths.provider_dir.parent
        / f".{paths.provider_dir.name}{BENCHMARK_LOCK_FILENAME}"
    ).open("a+b")
    try:
        lock_file(handle, exclusive=True)
    except Exception:
        handle.close()
        raise
    return handle


def _release_benchmark_lock(handle) -> None:
    try:
        unlock_file(handle)
    finally:
        handle.close()


def _import_benchmark_features(
    source_paths: BenchmarkPaths,
    target_paths: BenchmarkPaths,
    calendar: list[str],
    required_start_date: str,
) -> dict[str, int]:
    if source_paths.provider_dir.resolve() == target_paths.provider_dir.resolve():
        raise RuntimeError("benchmark import provider must differ from target provider")
    try:
        source_calendar_bytes = source_paths.calendar_file.read_bytes()
        target_calendar_bytes = target_paths.calendar_file.read_bytes()
    except OSError as exc:
        raise RuntimeError("benchmark import calendar is unavailable") from exc
    if source_calendar_bytes != target_calendar_bytes:
        raise RuntimeError("benchmark import calendar differs from target calendar")
    source_calendar = _read_calendar(source_paths.calendar_file)
    if source_calendar != calendar:
        raise RuntimeError("benchmark import calendar differs from target calendar")
    source_fingerprint = _source_provider_fingerprint(source_paths)
    available_days = _validate_existing_benchmarks(
        source_paths, source_calendar, required_start_date
    )
    for symbol in BENCHMARKS:
        _validate_feature_tree(source_paths.feature_dir(symbol))

    lock = _acquire_benchmark_lock(target_paths)
    try:
        _recover_transaction(target_paths)
        transaction_id = uuid.uuid4().hex
        entries = _transaction_entries(target_paths, transaction_id)
        journal = target_paths.provider_dir / BENCHMARK_TRANSACTION_FILENAME
        try:
            for entry in entries:
                target = Path(str(entry["target"]))
                if target.is_symlink() or (
                    target.exists() and not target.is_dir()
                ):
                    raise RuntimeError("existing benchmark feature directory is invalid")
            for entry in entries:
                symbol = str(entry["symbol"])
                stage = Path(str(entry["stage"]))
                _remove_path(stage)
                shutil.copytree(source_paths.feature_dir(symbol), stage)
                _validate_feature_tree(stage)
                if _feature_tree_fingerprint(stage) != _feature_tree_fingerprint(
                    source_paths.feature_dir(symbol)
                ):
                    raise RuntimeError("benchmark feature copy verification failed")
                _fsync_tree(stage)
            if _source_provider_fingerprint(source_paths) != source_fingerprint:
                raise RuntimeError("benchmark import source changed during validation")
            _write_transaction_journal(
                {
                    "version": BENCHMARK_TRANSACTION_VERSION,
                    "transaction_id": transaction_id,
                    "state": "prepared",
                    "entries": entries,
                },
                journal,
            )
            for entry in entries:
                target = Path(str(entry["target"]))
                backup = Path(str(entry["backup"]))
                stage = Path(str(entry["stage"]))
                if bool(entry["had_original"]):
                    if target.is_symlink() or not target.is_dir():
                        raise RuntimeError("existing benchmark feature directory is invalid")
                    os.replace(target, backup)
                os.replace(stage, target)
                _fsync_directory(target.parent)
            _validate_existing_benchmarks(
                target_paths,
                calendar,
                required_start_date,
                allow_incomplete_transaction=True,
            )
            _write_transaction_journal(
                {
                    "version": BENCHMARK_TRANSACTION_VERSION,
                    "transaction_id": transaction_id,
                    "state": "committed",
                    "entries": entries,
                },
                journal,
            )
            _cleanup_transaction(entries, journal)
        except Exception:
            _recover_transaction(target_paths)
            raise
    finally:
        _release_benchmark_lock(lock)
    return available_days


def _download_index_data(
    baostock_module, code: str, start_date: str, end_date: str
) -> pd.DataFrame:
    rs = baostock_module.query_history_k_data_plus(
        code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )

    def raise_for_error() -> None:
        error_code = str(getattr(rs, "error_code", "0"))
        if error_code != "0":
            raise RuntimeError(
                "baostock benchmark query failed: "
                f"{error_code} {getattr(rs, 'error_msg', '')}"
            )

    raise_for_error()
    rows: list[list[str]] = []
    while True:
        has_row = rs.next()
        raise_for_error()
        if not has_row:
            break
        rows.append(rs.get_row_data())
        raise_for_error()
    frame = pd.DataFrame(
        rows, columns=["date", "open", "high", "low", "close", "volume"]
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    for field in ("open", "high", "low", "close", "volume"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    return (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .set_index("date")
        .sort_index()
    )


def _download_benchmarks(
    baostock_module, start_date: str, end_date: str
) -> dict[str, pd.DataFrame]:
    lg = baostock_module.login()
    if getattr(lg, "error_code", "0") not in ("0", 0):
        raise RuntimeError(f"baostock login failed: {getattr(lg, 'error_msg', '')}")
    try:
        return {
            symbol: _download_index_data(baostock_module, code, start_date, end_date)
            for symbol, code in BENCHMARKS.items()
        }
    finally:
        baostock_module.logout()


def write_feature_bin(field_path: Path, start_idx: int, values) -> None:
    _write_bin(field_path, start_idx, np.asarray(values, dtype="<f4"))


def read_feature_bin(field_path: Path) -> tuple[int, np.ndarray]:
    return _read_qlib_bin(field_path)


def _download_eastmoney_index(
    start_date: str, end_date: str, *, use_system_proxy: bool = False
) -> dict[str, float]:
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
    return {
        parts[0]: float(parts[2])
        for line in klines
        if len(parts := str(line).split(",")) >= 3 and parts[2] not in {"", "-"}
    }


def _download_tencent_index(
    start_date: str, end_date: str, *, use_system_proxy: bool = False
) -> dict[str, float]:
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
            next_end = (
                datetime.strptime(earliest, "%Y-%m-%d").date()
                - timedelta(days=1)
            ).isoformat()
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
    result: dict[str, float] = {}
    for value, raw_close in zip(frame[date_column], frame[close_column]):
        close = float(raw_close)
        if math.isfinite(close) and close > 0:
            result[str(pd.Timestamp(value).date())] = close
    return result


def _download_single_baostock_index(
    baostock_module, start_date: str, end_date: str
) -> dict[str, float]:
    login = baostock_module.login()
    if getattr(login, "error_code", "0") not in ("0", 0):
        raise RuntimeError(f"baostock login failed: {getattr(login, 'error_msg', '')}")
    try:
        frame = _download_index_data(
            baostock_module, "sh.000300", start_date, end_date
        )
    finally:
        baostock_module.logout()
    return {
        str(day)[:10]: float(value)
        for day, value in frame["close"].dropna().items()
    }


def _safe_error(exc: Exception) -> str:
    text = re.sub(r"https?://[^\s?]+\?[^\s]+", "<url-redacted>", str(exc))
    return text[:200]


def _validate_source_data(
    data: dict[str, float], calendar: list[str]
) -> tuple[dict[str, float], int]:
    calendar_set = set(calendar)
    normalized: dict[str, float] = {}
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
    missing = [day for day in calendar[first_idx:] if day not in normalized]
    if missing:
        raise ValueError(
            f"source benchmark coverage is incomplete after {calendar[first_idx]}: "
            f"missing {len(missing)} dates; first={missing[0]} last={missing[-1]}"
        )
    return normalized, first_idx


def _cache_data(provider: Path) -> dict[str, float]:
    calendar = _read_calendar(provider / "calendars" / "day.txt")
    start_idx, values = read_feature_bin(
        provider / "features" / "sh000300" / "close.day.bin"
    )
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


def _write_single_benchmark_features(
    paths: BenchmarkPaths, calendar: list[str], data: dict[str, float]
) -> None:
    close_values = np.array([data.get(day, np.nan) for day in calendar], dtype="<f4")
    target = paths.feature_dir(BENCHMARK_SYMBOL)
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    staging.mkdir(parents=True)
    try:
        values_by_field = {
            "close": close_values,
            "volume": np.zeros(len(calendar), dtype="<f4"),
            "factor": np.ones(len(calendar), dtype="<f4"),
            "open": close_values,
            "high": close_values,
            "low": close_values,
        }
        for field in BENCHMARK_FEATURE_FIELDS:
            write_feature_bin(staging / f"{field}.day.bin", 0, values_by_field[field])
        backup = target.with_name(f".{target.name}.{uuid.uuid4().hex}.old")
        if target.exists():
            os.replace(target, backup)
        try:
            os.replace(staging, target)
        except Exception:
            if backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _add_single_benchmark_compat(
    provider_dir: Path,
    *,
    baostock_module,
    source_fetchers: list[tuple[str, Callable[[str, str], dict[str, float]]]] | None,
    cache_providers: list[Path] | None,
    status_dir: Path | None,
    use_system_proxy: bool,
    offline: bool,
) -> dict[str, object]:
    paths = BenchmarkPaths(provider_dir)
    calendar = _read_calendar(paths.calendar_file)
    attempts: list[dict[str, str]] = []
    if source_fetchers is None:
        source_fetchers = []
        if not offline:
            bs = _load_baostock() if baostock_module is None else baostock_module
            source_fetchers = [
                ("baostock", lambda start, end: _download_single_baostock_index(bs, start, end)),
                ("tencent", lambda start, end: _download_tencent_index(start, end, use_system_proxy=use_system_proxy)),
                ("eastmoney", lambda start, end: _download_eastmoney_index(start, end, use_system_proxy=use_system_proxy)),
                ("akshare", _download_akshare_index),
            ]
    caches = _default_cache_providers(provider_dir) if cache_providers is None else cache_providers
    selected_source = ""
    selected_data: dict[str, float] | None = None
    first_idx = 0
    candidates = list(source_fetchers) + [
        (f"local_cache:{provider}", lambda _start, _end, provider=provider: _cache_data(provider))
        for provider in caches
    ]
    for source_name, fetcher in candidates:
        try:
            selected_data, first_idx = _validate_source_data(
                fetcher(calendar[0], calendar[-1]), calendar
            )
            selected_source = source_name
            attempts.append({"source": source_name, "status": "ok"})
            break
        except Exception as exc:
            attempts.append(
                {"source": source_name, "status": "failed", "error": _safe_error(exc)}
            )
    status_root = Path(status_dir) if status_dir else reports_root() / "runtime" / "status"
    if selected_data is None:
        write_json_atomic(
            status_root / "benchmark_update_latest.json",
            {
                "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": "failed",
                "benchmark_symbol": BENCHMARK_SYMBOL,
                "provider_dir": str(provider_dir),
                "attempts": attempts,
            },
        )
        raise BenchmarkSourceError("all benchmark sources failed validation")
    _write_single_benchmark_features(paths, calendar, selected_data)
    for instrument in paths.instruments_dir.glob("*.txt"):
        _remove_symbol_if_present(instrument, BENCHMARK_SYMBOL)
    result: dict[str, object] = {
        "status": "completed",
        "provider_dir": str(provider_dir),
        "calendar_days": len(calendar),
        "downloaded_days": len(selected_data),
        "first_observation": calendar[first_idx],
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


def add_benchmark(
    provider_dir: Path = DATA_DIR,
    *,
    dry_run: bool = False,
    reuse_existing: bool = False,
    import_provider_dir: Path | None = None,
    baostock_module=None,
    required_start_date: str = "2014-01-02",
    source_fetchers: list[
        tuple[str, Callable[[str, str], dict[str, float]]]
    ] | None = None,
    cache_providers: list[Path] | None = None,
    status_dir: Path | None = None,
    use_system_proxy: bool = False,
    offline: bool = False,
    benchmark_source: str = "baostock",
) -> dict[str, object]:
    if benchmark_source not in {"baostock", "tencent"}:
        raise ValueError("benchmark_source must be baostock or tencent")
    legacy_single_requested = (
        source_fetchers is not None
        or cache_providers is not None
        or offline
        or (use_system_proxy and benchmark_source == "baostock")
    )
    if legacy_single_requested:
        if dry_run or reuse_existing or import_provider_dir is not None:
            raise ValueError(
                "legacy single-benchmark fallback cannot be combined with robust benchmark modes"
            )
        return _add_single_benchmark_compat(
            provider_dir,
            baostock_module=baostock_module,
            source_fetchers=source_fetchers,
            cache_providers=cache_providers,
            status_dir=status_dir,
            use_system_proxy=use_system_proxy,
            offline=offline,
        )
    if sum((bool(dry_run), bool(reuse_existing), import_provider_dir is not None)) > 1:
        raise ValueError(
            "dry_run, reuse_existing, and import_provider_dir are mutually exclusive"
        )
    paths = BenchmarkPaths(provider_dir)
    try:
        calendar = _read_calendar(paths.calendar_file)
    except (FileNotFoundError, ValueError) as exc:
        if not dry_run:
            raise
        calendar = []
        print(f"DRY RUN: Qlib calendar unavailable: {exc}")
    if calendar:
        print(
            f"Calendar: {len(calendar)} trading days ({calendar[0]} ~ {calendar[-1]})"
        )
    else:
        print("Calendar: 0 trading days (unavailable)")
    print(f"Provider: {paths.provider_dir}")
    print(
        f"Benchmark feature dirs: {', '.join(str(paths.feature_dir(symbol)) for symbol in BENCHMARKS)}"
    )

    if dry_run:
        print(
            "Dry run OK. No baostock request, bin write, or instruments rewrite will be performed."
        )
        return {
            "status": "dry-run",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "benchmark_symbols": list(BENCHMARKS),
        }

    if reuse_existing:
        available_days = _validate_existing_benchmarks(
            paths, calendar, required_start_date
        )
        print("Existing benchmark features passed strict offline validation")
        return {
            "status": "reused-existing",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "downloaded_days": available_days,
            "benchmark_symbols": list(BENCHMARKS),
        }

    if import_provider_dir is not None:
        available_days = _import_benchmark_features(
            BenchmarkPaths(import_provider_dir),
            paths,
            calendar,
            required_start_date,
        )
        print("Imported benchmark features from a verified offline provider")
        return {
            "status": "imported-existing",
            "provider_dir": str(paths.provider_dir),
            "calendar_days": len(calendar),
            "downloaded_days": available_days,
            "benchmark_symbols": list(BENCHMARKS),
        }

    if benchmark_source == "tencent":
        downloaded = _download_tencent_benchmarks(
            calendar[0], calendar[-1], use_system_proxy=use_system_proxy
        )
    else:
        bs = _load_baostock() if baostock_module is None else baostock_module
        downloaded = _download_benchmarks(bs, calendar[0], calendar[-1])
    downloaded_days: dict[str, int] = {}
    calendar_index = pd.Index(calendar)
    required_start = max(pd.Timestamp(calendar[0]), pd.Timestamp(required_start_date))
    required_calendar = calendar_index[
        pd.to_datetime(calendar_index, errors="coerce") >= required_start
    ]
    if required_calendar.empty:
        raise RuntimeError("Qlib calendar has no dates in the required benchmark period")
    aligned_by_symbol: dict[str, pd.DataFrame] = {}
    for symbol, frame in downloaded.items():
        if frame.empty:
            raise RuntimeError(f"no benchmark data returned for {symbol}")
        aligned = frame.reindex(calendar_index)
        price_fields = ["open", "high", "low", "close"]
        required = aligned.loc[required_calendar]
        missing_dates = required.index[
            required[price_fields + ["volume"]].isna().any(axis=1)
        ].tolist()
        if missing_dates:
            raise RuntimeError(
                f"benchmark {symbol} is missing required trading days: "
                f"{missing_dates[:10]}"
            )
        if (required[price_fields] <= 0).any(axis=None) or (
            required["volume"] < 0
        ).any(axis=None):
            raise RuntimeError(f"benchmark OHLCV is invalid for {symbol}")
        aligned_by_symbol[symbol] = aligned
        downloaded_days[symbol] = int(len(frame))

    features_root = paths.provider_dir / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".benchmark-stage-", dir=features_root))
    backup_root = Path(tempfile.mkdtemp(prefix=".benchmark-backup-", dir=features_root))
    installed: list[tuple[Path, Path | None]] = []
    try:
        for symbol, aligned in aligned_by_symbol.items():
            feature_dir = staging_root / symbol.lower()
            for field in ["open", "high", "low", "close", "volume"]:
                _write_bin(
                    feature_dir / f"{field}.day.bin",
                    0,
                    aligned[field].to_numpy(dtype="<f4"),
                )
            _write_bin(
                feature_dir / "factor.day.bin",
                0,
                np.ones(len(calendar), dtype="<f4"),
            )
        for symbol in aligned_by_symbol:
            target = paths.feature_dir(symbol)
            backup = backup_root / symbol.lower()
            previous = backup if target.exists() else None
            if previous is not None:
                os.replace(target, previous)
            try:
                os.replace(staging_root / symbol.lower(), target)
            except Exception:
                if previous is not None and previous.exists():
                    os.replace(previous, target)
                raise
            installed.append((target, previous))
    except Exception:
        for target, previous in reversed(installed):
            shutil.rmtree(target, ignore_errors=True)
            if previous is not None and previous.exists():
                os.replace(previous, target)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
        shutil.rmtree(backup_root, ignore_errors=True)

    for symbol, frame in downloaded.items():
        print(
            f"Written {symbol}: {len(frame)} downloaded days -> "
            f"{paths.feature_dir(symbol)}"
        )

    # Benchmarks are features only and must never enter any stock training pool.
    for instrument_file in paths.instruments_dir.glob("*.txt"):
        for symbol in BENCHMARKS:
            _remove_symbol_if_present(instrument_file, symbol)
    print("Ensured benchmark symbols are absent from stock instrument pools")

    print("Done!")
    return {
        "status": "completed",
        "source": benchmark_source,
        "provider_dir": str(paths.provider_dir),
        "calendar_days": len(calendar),
        "downloaded_days": downloaded_days,
        "benchmark_symbols": list(BENCHMARKS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add CSI300/CSI500/CSI800 benchmark features to a Qlib provider."
    )
    parser.add_argument("--provider-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--required-start-date", default="2014-01-02")
    parser.add_argument("--cache-provider", type=Path, action="append", default=None)
    parser.add_argument("--status-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument(
        "--source", choices=("baostock", "tencent"), default="baostock"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and calendar without network or writes.",
    )
    mode.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Strictly validate existing benchmark bins without network access.",
    )
    mode.add_argument(
        "--import-provider-dir",
        type=Path,
        default=None,
        help=(
            "Validate benchmark bins in a separate provider with an identical "
            "calendar and transactionally install them without network access."
        ),
    )
    args = parser.parse_args(argv)
    try:
        add_benchmark(
            args.provider_dir,
            dry_run=args.dry_run,
            reuse_existing=args.reuse_existing,
            import_provider_dir=args.import_provider_dir,
            required_start_date=args.required_start_date,
            cache_providers=args.cache_provider,
            status_dir=args.status_dir,
            use_system_proxy=args.use_system_proxy,
            offline=args.offline,
            benchmark_source=args.source,
        )
    except BenchmarkSourceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
