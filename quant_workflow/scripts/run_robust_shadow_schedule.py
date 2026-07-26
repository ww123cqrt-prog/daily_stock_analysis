#!/usr/bin/env python3
"""Run the active frozen strategy after the daily data update completes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, time as clock_time
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from run_robust_shadow_day import run_shadow_day  # noqa: E402
from run_robust_strategy import DEFAULT_CONFIG  # noqa: E402

SHANGHAI = ZoneInfo("Asia/Shanghai")
READY_DAILY_STATUSES = {"ok", "ok_with_warnings", "skipped"}
DEFAULT_WAIT_TIMEOUT_SECONDS = 4 * 60 * 60
DAILY_UPDATE_START_TIME = clock_time(18, 40)
A_SHARE_CALENDAR = "XSHG"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _mps_probe() -> dict[str, Any]:
    import torch

    built = bool(torch.backends.mps.is_built())
    available = bool(torch.backends.mps.is_available())
    if not built or not available:
        raise RuntimeError("MPS is required but unavailable for shadow inference")
    tensor = torch.ones(4, device="mps")
    tensor_sum = float((tensor * tensor).sum().cpu())
    if tensor_sum != 4.0:
        raise RuntimeError("MPS shadow tensor probe returned an invalid result")
    return {
        "python_executable": str(Path(sys.executable).absolute()),
        "python_resolved": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "mps_built": built,
        "mps_available": available,
        "device": str(tensor.device),
        "tensor_sum": tensor_sum,
        "checked_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
    }


def _is_active(frozen: Mapping[str, Any], confirmation: Mapping[str, Any]) -> bool:
    return (
        confirmation.get("status") == "shadow_observation"
        and frozen.get("confirmation_status") == "shadow_observation"
        and frozen.get("shadow_tracking_started") is True
    )


def _summary_is_for_date(summary: Mapping[str, Any], as_of_date: str) -> bool:
    try:
        generated = datetime.fromisoformat(str(summary["generated_at"]))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=SHANGHAI)
        local_generated = generated.astimezone(SHANGHAI)
        return (
            local_generated.date().isoformat() == as_of_date
            and local_generated.time() >= DAILY_UPDATE_START_TIME
        )
    except (KeyError, TypeError, ValueError):
        return False


def _wait_for_daily_update(
    path: Path,
    *,
    as_of_date: str,
    timeout_seconds: float,
    poll_seconds: float = 30.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        summary = _read_json(path)
        if _summary_is_for_date(summary, as_of_date):
            return summary
        if time.monotonic() >= deadline:
            return {}
        sleeper(min(float(poll_seconds), max(0.0, deadline - time.monotonic())))


def _daily_data_gate_reason(
    summary: Mapping[str, Any], *, as_of_date: str
) -> str | None:
    if summary.get("official_published") is not True:
        return "daily_data_not_published"
    try:
        source_date = datetime.fromisoformat(
            str(summary.get("source_date_after") or "")
        ).date().isoformat()
    except ValueError:
        return "daily_data_not_fresh"
    if source_date != as_of_date:
        return "daily_data_not_fresh"
    return None


def _is_a_share_trading_session(as_of_date: str) -> bool | None:
    """Return None when the bundled exchange calendar cannot cover the date."""
    try:
        import exchange_calendars
        from exchange_calendars.errors import DateOutOfBounds
    except ImportError:
        return None
    try:
        datetime.fromisoformat(as_of_date).date()
        calendar = exchange_calendars.get_calendar(A_SHARE_CALENDAR)
        return bool(calendar.is_session(as_of_date))
    except (DateOutOfBounds, TypeError, ValueError):
        return None


def run_scheduled_shadow(
    *,
    config_path: Path,
    quant_data_dir: Path,
    reports_root: Path,
    as_of_date: str,
    wait_timeout_seconds: float = DEFAULT_WAIT_TIMEOUT_SECONDS,
    shadow_runner: Callable[..., dict[str, Any]] = run_shadow_day,
) -> dict[str, Any]:
    reports = Path(reports_root).resolve()
    robust = reports / "robust_strategy"
    runner_dir = robust / "shadow_runner"
    base = {
        "as_of": str(as_of_date),
        "generated_at": datetime.now(SHANGHAI).isoformat(timespec="seconds"),
        "python_executable": str(Path(sys.executable).absolute()),
    }
    try:
        as_of = datetime.fromisoformat(str(as_of_date)).date().isoformat()
    except (TypeError, ValueError):
        result = {
            **base,
            "status": "failed",
            "outcome": "failed",
            "reason_code": "invalid_as_of_date",
        }
        _atomic_write_json(runner_dir / "schedule_latest.json", result)
        return result
    base["as_of"] = as_of
    frozen_path = robust / "frozen_strategy_latest.json"
    confirmation_path = robust / "confirmation_latest.json"
    provider_path = robust / "provider_latest.json"
    frozen = _read_json(frozen_path)
    confirmation = _read_json(confirmation_path)
    shadow_state = _read_json(robust / "shadow_state_latest.json")

    if shadow_state.get("status") in {
        "robust_strategy_qualified",
        "development_failed",
    }:
        result = {
            **base,
            "status": "inactive",
            "outcome": "skipped",
            "reason_code": "shadow_tracking_finalized",
        }
    elif not _is_active(frozen, confirmation):
        result = {
            **base,
            "status": "inactive",
            "outcome": "skipped",
            "reason_code": "shadow_not_active",
        }
    elif (calendar_session := _is_a_share_trading_session(as_of)) is None:
        result = {
            **base,
            "status": "failed",
            "outcome": "failed",
            "reason_code": "trading_calendar_unavailable",
            "calendar": A_SHARE_CALENDAR,
        }
    elif calendar_session is False:
        result = {
            **base,
            "status": "shadow_observation",
            "outcome": "skipped",
            "reason_code": "non_trading_day",
            "calendar": A_SHARE_CALENDAR,
        }
    else:
        summary = _wait_for_daily_update(
            reports / "daily_update_summary_latest.json",
            as_of_date=as_of,
            timeout_seconds=wait_timeout_seconds,
        )
        daily_status = str(summary.get("status") or "")
        data_gate_reason = _daily_data_gate_reason(summary, as_of_date=as_of)
        if not summary:
            result = {
                **base,
                "status": "failed",
                "outcome": "failed",
                "reason_code": "daily_update_timeout",
            }
        elif summary.get("dry_run") is True:
            result = {
                **base,
                "status": "failed",
                "outcome": "failed",
                "reason_code": "daily_update_not_executed",
                "daily_update_status": daily_status,
            }
        elif daily_status not in READY_DAILY_STATUSES:
            result = {
                **base,
                "status": "failed",
                "outcome": "failed",
                "reason_code": "daily_update_failed",
                "daily_update_status": daily_status,
            }
        elif data_gate_reason is not None:
            result = {
                **base,
                "status": "failed",
                "outcome": "failed",
                "reason_code": data_gate_reason,
                "daily_update_status": daily_status,
                "official_published": summary.get("official_published") is True,
                "source_date_after": summary.get("source_date_after"),
            }
        else:
            mps_preflight = None
            if "mps_dnn" in {
                str(value) for value in frozen.get("selected_models") or []
            }:
                try:
                    mps_preflight = _mps_probe()
                except Exception as exc:
                    result = {
                        **base,
                        "status": "failed",
                        "outcome": "failed",
                        "reason_code": "mps_preflight_failed",
                        "daily_update_status": daily_status,
                        "error_type": type(exc).__name__,
                    }
                else:
                    result = {}
            else:
                result = {}
            if not result:
                try:
                    shadow = shadow_runner(
                        config_path=Path(config_path),
                        quant_data_dir=Path(quant_data_dir),
                        report_dir=robust,
                        frozen_path=frozen_path,
                        confirmation_path=confirmation_path,
                        provider_state_path=provider_path,
                        as_of_date=as_of,
                    )
                except Exception as exc:
                    result = {
                        **base,
                        "status": "failed",
                        "outcome": "failed",
                        "reason_code": "shadow_runner_failed",
                        "daily_update_status": daily_status,
                        "mps_preflight": mps_preflight,
                        "error_type": type(exc).__name__,
                    }
                else:
                    result = {
                        **base,
                        **dict(shadow),
                        "daily_update_status": daily_status,
                        "mps_preflight": mps_preflight,
                    }

    _atomic_write_json(runner_dir / f"schedule_{as_of.replace('-', '')}.json", result)
    _atomic_write_json(runner_dir / "schedule_latest.json", result)
    return result


def exit_code_for_schedule(result: Mapping[str, Any]) -> int:
    if result.get("status") == "inactive" and result.get("outcome") == "skipped":
        return 0
    if (
        result.get("outcome") == "skipped"
        and result.get("reason_code") == "non_trading_day"
    ):
        return 0
    if result.get("status") in {
        "shadow_observation",
        "robust_strategy_qualified",
    } and not result.get("reason_code"):
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the scheduled robust shadow observation."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument("--reports-root", type=Path, default=reports_root())
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    as_of = args.as_of_date or datetime.now(SHANGHAI).date().isoformat()
    result = run_scheduled_shadow(
        config_path=args.config,
        quant_data_dir=args.quant_data_dir,
        reports_root=args.reports_root,
        as_of_date=as_of,
        wait_timeout_seconds=args.wait_timeout_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code_for_schedule(result)


if __name__ == "__main__":
    raise SystemExit(main())
