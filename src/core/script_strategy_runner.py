# -*- coding: utf-8 -*-
"""Child-process entrypoint for bounded DSA strategy script execution."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import resource
from typing import Any

from src.core.script_strategy import execute_strategy_script
from src.core.strategy_backtest import PriceBar


def _apply_limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _serialize_result(payload: dict[str, Any]) -> dict[str, Any]:
    bars = [
        PriceBar(
            symbol=str(item["symbol"]),
            date=date.fromisoformat(str(item["date"])),
            open=float(item["open"]),
            close=float(item["close"]),
        )
        for item in payload.get("bars", [])
    ]
    start_date = date.fromisoformat(payload["start_date"]) if payload.get("start_date") else None
    end_date = date.fromisoformat(payload["end_date"]) if payload.get("end_date") else None
    result = execute_strategy_script(
        str(payload.get("script") or ""),
        bars=bars,
        symbols=tuple(str(symbol) for symbol in payload.get("symbols", [])),
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "status": "completed",
        "allocations": [
            {
                "source_id": item.source_id,
                "signal_date": item.signal_date.isoformat(),
                "target_weights": dict(item.target_weights),
                "reason": item.reason,
            }
            for item in result.allocations
        ],
        "logs": list(result.logs),
        "check": result.check,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _apply_limits()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = _serialize_result(payload)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
