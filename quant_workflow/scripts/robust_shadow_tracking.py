#!/usr/bin/env python3
"""Persist and evaluate the 60-trading-day paper observation period."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robust_evaluation import shadow_verdict  # noqa: E402
from run_robust_strategy import DEFAULT_CONFIG, load_config  # noqa: E402

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - msvcrt is only available on Windows.
    msvcrt = None


_FILE_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = str(Path(path).resolve())
    with _FILE_LOCKS_GUARD:
        return _FILE_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _file_guard(path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with _thread_lock(lock_path):
        with lock_path.open("a+b") as stream:
            if fcntl is not None:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            elif msvcrt is not None:  # pragma: no cover - Windows fallback.
                if lock_path.stat().st_size == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
                elif msvcrt is not None:  # pragma: no cover - Windows fallback.
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_signal_predictions(
    predictions: pd.DataFrame, *, as_of_date: str
) -> pd.DataFrame:
    required = {"datetime", "instrument", "score"}
    if not required.issubset(predictions.columns):
        raise ValueError(
            "shadow predictions must contain datetime, instrument, and score"
        )
    result = predictions.loc[:, ["datetime", "instrument", "score"]].copy()
    result["datetime"] = pd.to_datetime(
        result["datetime"], errors="coerce"
    ).dt.normalize()
    result["instrument"] = result["instrument"].astype(str).str.strip().str.upper()
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    as_of = pd.Timestamp(as_of_date).normalize()
    if (
        result.empty
        or result["datetime"].isna().any()
        or not result["datetime"].eq(as_of).all()
    ):
        raise ValueError("shadow predictions must contain only as_of_date")
    if (
        result["instrument"].eq("").any()
        or result.duplicated(["datetime", "instrument"]).any()
    ):
        raise ValueError(
            "shadow prediction date/instrument keys must be unique and nonempty"
        )
    if not result["score"].map(math.isfinite).all():
        raise ValueError("shadow prediction scores must be finite")
    return result.sort_values(["datetime", "instrument"], kind="stable").reset_index(
        drop=True
    )


def append_signal_predictions(
    path: Path, predictions: pd.DataFrame, *, as_of_date: str
) -> dict[str, int]:
    path = Path(path)
    incoming = _normalize_signal_predictions(predictions, as_of_date=as_of_date)
    with _file_guard(path):
        if path.is_file():
            existing = pd.read_parquet(path)
            if not {"datetime", "instrument", "score"}.issubset(existing.columns):
                raise ValueError("stored shadow predictions have an invalid schema")
            existing = existing.loc[:, ["datetime", "instrument", "score"]].copy()
            existing["datetime"] = pd.to_datetime(
                existing["datetime"], errors="coerce"
            ).dt.normalize()
            existing["instrument"] = (
                existing["instrument"].astype(str).str.strip().str.upper()
            )
            existing["score"] = pd.to_numeric(existing["score"], errors="coerce")
            if (
                existing[["datetime", "instrument", "score"]].isna().any(axis=None)
                or existing["instrument"].eq("").any()
                or existing.duplicated(["datetime", "instrument"]).any()
                or not existing["score"].map(math.isfinite).all()
            ):
                raise ValueError("stored shadow predictions are invalid")
        else:
            existing = pd.DataFrame(columns=["datetime", "instrument", "score"])

        keyed = (
            existing.set_index(["datetime", "instrument"])["score"]
            if not existing.empty
            else pd.Series(dtype=float)
        )
        incoming_keyed = incoming.set_index(["datetime", "instrument"])["score"]
        overlap = incoming_keyed.index.intersection(keyed.index)
        for key in overlap:
            if float(incoming_keyed.loc[key]) != float(keyed.loc[key]):
                raise ValueError(
                    "shadow prediction key already exists with a different score"
                )
        additions = incoming.loc[
            ~incoming.set_index(["datetime", "instrument"]).index.isin(keyed.index)
        ]
        if additions.empty:
            return {"appended_rows": 0, "total_rows": int(len(existing))}
        combined = (
            additions.copy()
            if existing.empty
            else pd.concat([existing, additions], ignore_index=True)
        )
        combined = combined.sort_values(
            ["datetime", "instrument"], kind="stable"
        ).reset_index(drop=True)
        _atomic_write_parquet(combined, path)
        return {"appended_rows": int(len(additions)), "total_rows": int(len(combined))}


def new_shadow_state(
    *,
    confirmation_run_id: str,
    frozen_config_hash: str,
    start_after: str,
    acceptance_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    start = pd.Timestamp(start_after).normalize()
    if not confirmation_run_id or len(str(frozen_config_hash)) != 64:
        raise ValueError("confirmation_run_id and frozen_config_hash are required")
    thresholds = copy.deepcopy(dict(acceptance_thresholds))
    initial_verdict = shadow_verdict(
        trading_days=0,
        net_excess_return=0.0,
        max_drawdown=0.0,
        task_success_ratio=0.0,
        thresholds=thresholds,
    )
    return {
        "status": "shadow_observation",
        "confirmation_run_id": str(confirmation_run_id),
        "frozen_config_hash": str(frozen_config_hash),
        "start_after": str(start.date()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "acceptance_thresholds": thresholds,
        "observations": [],
        "metrics": {
            "trading_days": 0,
            "net_excess_return": 0.0,
            "max_drawdown": 0.0,
            "task_success_ratio": 0.0,
        },
        "verdict": initial_verdict,
    }


def _metrics(observations: list[Mapping[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(observations)
    portfolio = (
        1.0 + pd.to_numeric(frame["portfolio_return"], errors="raise")
    ).cumprod()
    benchmark = (
        1.0 + pd.to_numeric(frame["benchmark_return"], errors="raise")
    ).cumprod()
    drawdown = portfolio / portfolio.cummax() - 1.0
    return {
        "trading_days": int(len(frame)),
        "portfolio_return": float(portfolio.iloc[-1] - 1.0),
        "benchmark_return": float(benchmark.iloc[-1] - 1.0),
        "net_excess_return": float(portfolio.iloc[-1] - benchmark.iloc[-1]),
        "max_drawdown": float(drawdown.min()),
        "task_success_ratio": float(frame["task_success"].astype(bool).mean()),
        "first_date": str(frame["date"].iloc[0]),
        "last_date": str(frame["date"].iloc[-1]),
    }


def append_observation(
    state: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(state))
    if result.get("status") in {"robust_strategy_qualified", "development_failed"}:
        raise ValueError("shadow observation is already finalized")
    day = pd.Timestamp(observation.get("date")).normalize()
    start_after = pd.Timestamp(result["start_after"]).normalize()
    if day <= start_after:
        raise ValueError(
            "shadow observation date must be after the confirmation period"
        )
    try:
        portfolio_return = float(observation.get("portfolio_return"))
        benchmark_return = float(observation.get("benchmark_return"))
    except (TypeError, ValueError) as exc:
        raise ValueError("shadow returns must be finite numbers") from exc
    if not all(
        math.isfinite(value) and value > -1.0
        for value in (portfolio_return, benchmark_return)
    ):
        raise ValueError("shadow returns must be finite and greater than -1")
    normalized = {
        "date": str(day.date()),
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "task_success": bool(observation.get("task_success")),
    }
    observations = result.setdefault("observations", [])
    existing = next(
        (item for item in observations if item.get("date") == normalized["date"]), None
    )
    if existing is not None:
        if existing == normalized:
            return result
        raise ValueError("shadow observation date already exists with different values")
    if observations and day <= pd.Timestamp(observations[-1]["date"]):
        raise ValueError("shadow observations must be appended chronologically")
    observations.append(normalized)
    metrics = _metrics(observations)
    verdict = shadow_verdict(
        trading_days=metrics["trading_days"],
        net_excess_return=metrics["net_excess_return"],
        max_drawdown=metrics["max_drawdown"],
        task_success_ratio=metrics["task_success_ratio"],
        thresholds=result["acceptance_thresholds"],
    )
    result["metrics"] = metrics
    result["verdict"] = verdict
    result["status"] = verdict["status"]
    result["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    return result


def update_shadow_state_file(
    path: Path,
    *,
    confirmation_run_id: str,
    frozen_config_hash: str,
    start_after: str,
    observation: Mapping[str, Any],
    acceptance_thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    with _file_guard(path):
        state = _read_json(path)
        if not state:
            state = new_shadow_state(
                confirmation_run_id=confirmation_run_id,
                frozen_config_hash=frozen_config_hash,
                start_after=start_after,
                acceptance_thresholds=acceptance_thresholds,
            )
        elif state.get("acceptance_thresholds") != dict(acceptance_thresholds):
            raise ValueError("shadow acceptance thresholds changed")
        updated = append_observation(state, observation)
        _atomic_write(
            path,
            json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one read-only paper observation to the frozen robust strategy."
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    confirmation = _read_json(args.confirmation)
    frozen = _read_json(args.frozen)
    if (
        confirmation.get("status") != "shadow_observation"
        or frozen.get("confirmation_status") != "shadow_observation"
    ):
        raise SystemExit("confirmation has not passed; shadow tracking cannot start")
    observation = _read_json(args.observation)
    updated = update_shadow_state_file(
        args.state,
        confirmation_run_id=str(confirmation["run_id"]),
        frozen_config_hash=str(frozen["config_hash"]),
        start_after=str(confirmation["latest_market_day"]),
        observation=observation,
        acceptance_thresholds=config["acceptance"],
    )
    print(json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True))
    return (
        0
        if updated["status"] in {"shadow_observation", "robust_strategy_qualified"}
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
