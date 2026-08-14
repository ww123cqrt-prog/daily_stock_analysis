# -*- coding: utf-8 -*-
"""Read-only status proxy for the governed local quant_data service."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter

from api.v1.errors import api_error
from quant_workflow.scripts.quant_data_service_client import (
    DEFAULT_QUANT_DATA_SERVICE_URL,
    QuantDataClientError,
    QuantDataServiceClient,
)


router = APIRouter()


@router.get("/status")
def quant_data_status() -> dict[str, Any]:
    """Return canonical readiness and immutable identities without filesystem paths."""
    try:
        with QuantDataServiceClient(
            base_url=os.getenv(
                "QUANT_DATA_SERVICE_URL", DEFAULT_QUANT_DATA_SERVICE_URL
            )
        ) as client:
            readiness = client.ready()
    except QuantDataClientError as exc:
        raise api_error(
            503,
            "quant_data_unavailable",
            "quant_data 数据后端暂不可用。",
        ) from exc

    return {
        "status": readiness.status,
        "validated_datasets": readiness.validated_datasets,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "datasets": [dict(dataset) for dataset in readiness.datasets],
    }
