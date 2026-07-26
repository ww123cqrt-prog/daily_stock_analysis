# -*- coding: utf-8 -*-
"""Read-only access to the canonical ETF momentum strategy report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter

from src.services.quant_workflow_paths import dsa_root, reports_root


ETF_REPORT_PATH = (
    reports_root(dsa_root())
    / "etf_momentum_rotation"
    / "strategy_report_latest.json"
)
ETF_REPORT_MAX_BYTES = 12 * 1024 * 1024

router = APIRouter()


def _error_payload(reason_code: str, message: str) -> Dict[str, str]:
    return {
        "status": "invalid",
        "reason_code": reason_code,
        "message": message,
    }


def read_etf_momentum_strategy_report(
    path: Path = ETF_REPORT_PATH,
) -> Dict[str, Any]:
    """Read one bounded JSON object without exposing its filesystem path."""

    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {
            "status": "missing",
            "reason_code": "report_not_generated",
            "message": "ETF momentum strategy report has not been generated yet.",
        }
    except OSError:
        return _error_payload(
            "report_unreadable",
            "ETF momentum strategy report could not be read.",
        )
    if size > ETF_REPORT_MAX_BYTES:
        return _error_payload(
            "report_too_large",
            "ETF momentum strategy report exceeds the safe response size.",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _error_payload(
            "report_invalid_json",
            "ETF momentum strategy report contains invalid JSON.",
        )
    except OSError:
        return _error_payload(
            "report_unreadable",
            "ETF momentum strategy report could not be read.",
        )
    if not isinstance(payload, dict):
        return _error_payload(
            "report_invalid_shape",
            "ETF momentum strategy report must be a JSON object.",
        )
    return payload


@router.get("/strategies/etf-momentum")
async def get_etf_momentum_strategy_report() -> Dict[str, Any]:
    return read_etf_momentum_strategy_report(ETF_REPORT_PATH)
