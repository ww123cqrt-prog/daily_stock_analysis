# -*- coding: utf-8 -*-
"""Read-only snapshot metadata from the local lightweight data package."""
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter
from api.v1.errors import api_error
from src.services.quant_workflow_paths import quant_data_root

router = APIRouter()


@router.get("/status")
def quant_data_status() -> dict[str, Any]:
    """Inspect committed identities; this is not a full-history data audit."""
    try:
        from quant_qmt_data import DataContractError, inspect_datasets
    except ImportError as exc:
        raise api_error(503, "quant_data_unavailable", "本机 quant_qmt_data 尚未安装。") from exc
    try:
        status = inspect_datasets(quant_data_root=quant_data_root())
    except (DataContractError, OSError, ValueError, TypeError) as exc:
        raise api_error(503, "quant_data_unavailable", "本机数据清单暂不可用。") from exc
    return {**status, "checked_at": datetime.now(timezone.utc).isoformat()}
