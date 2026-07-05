# -*- coding: utf-8 -*-
"""Quant workflow job endpoints."""

from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, StreamingResponse

from api.v1.errors import api_error
from api.v1.schemas.quant_jobs import (
    QuantJobListResponse,
    QuantJobLogResponse,
    QuantJobResponse,
    QuantJobSubmitRequest,
    QuantStatusResponse,
)
from api.v1.endpoints.research_status import build_research_status_payload
from src.services.quant_job_service import QuantJob, QuantJobService, REPORTS_ROOT


router = APIRouter()
JOB_STREAM_POLL_INTERVAL_SECONDS = 2.0
JOB_STREAM_MAX_JOB_EVENTS = 300
TEXT_ARTIFACT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".jsonl", ".log"}
DEFAULT_MLFLOW_URL = "http://127.0.0.1:6006/"


def _mlflow_url() -> str:
    return os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL).strip() or DEFAULT_MLFLOW_URL


@lru_cache(maxsize=1)
def get_quant_job_service() -> QuantJobService:
    return QuantJobService()


def _job_response(job: QuantJob) -> QuantJobResponse:
    return QuantJobResponse(**job.to_dict())


def _is_allowed_artifact_path(path: Path, service: QuantJobService) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError:
        return False
    allowed_roots = {REPORTS_ROOT, service.jobs_root.parent}
    for root in allowed_roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
            return True
        except ValueError:
            continue
    return False


@router.get("/status", response_model=QuantStatusResponse)
async def get_quant_status() -> QuantStatusResponse:
    """Return quant workflow status used by the web workflow page."""
    service = get_quant_job_service()
    latest_jobs = {
        job_type: _job_response(job)
        for job_type, job in service.get_latest_jobs_by_type().items()
    }
    research_status = build_research_status_payload()
    return QuantStatusResponse(
        available_job_types=service.available_job_types(),
        active_count=service.active_count(),
        latest_jobs=latest_jobs,
        recent_jobs=[_job_response(job) for job in service.list_jobs(limit=10)],
        generated_at=research_status.get("generated_at"),
        trainable=bool(research_status.get("trainable")),
        training_readiness=research_status.get("training_readiness") or {},
        data_sources=research_status.get("data_sources") or [],
        qlib_provider=research_status.get("qlib_provider"),
        latest_training=research_status.get("latest_training"),
        cash_control_backtest=research_status.get("cash_control_backtest"),
        qlib_predictions=research_status.get("qlib_predictions"),
        experiment_notebook=research_status.get("experiment_notebook"),
        mlflow={
            "url": _mlflow_url(),
            "status": "linked",
            "role": "experiment_raw_records",
        },
    )


@router.post("/jobs", response_model=QuantJobResponse)
async def submit_quant_job(request: QuantJobSubmitRequest) -> QuantJobResponse:
    """Submit one whitelisted quant workflow job."""
    service = get_quant_job_service()
    try:
        job = service.submit_job(request.job_type, request.params)
    except ValueError as exc:
        raise api_error(400, "invalid_params", str(exc)) from exc
    return _job_response(job)


@router.get("/jobs", response_model=QuantJobListResponse)
async def list_quant_jobs(
    limit: int = Query(50, ge=1, le=200),
    job_type: Optional[str] = Query(None),
) -> QuantJobListResponse:
    service = get_quant_job_service()
    jobs = service.list_jobs(limit=limit, job_type=job_type)
    return QuantJobListResponse(total=len(jobs), items=[_job_response(job) for job in jobs])


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _list_jobs_for_stream(service: QuantJobService) -> list[QuantJob]:
    """Fetch stream snapshots without blocking the event loop on SQLite I/O."""
    return await asyncio.to_thread(service.list_jobs, limit=20)


async def _quant_job_event_generator(
    service: QuantJobService,
    *,
    max_job_events: int = JOB_STREAM_MAX_JOB_EVENTS,
    poll_interval_seconds: float = JOB_STREAM_POLL_INTERVAL_SECONDS,
):
    yield _sse_event("connected", {"type": "connected"})
    for _ in range(max(1, int(max_job_events))):
        jobs = [job.to_dict() for job in await _list_jobs_for_stream(service)]
        yield _sse_event("jobs", {"type": "jobs", "jobs": jobs})
        if poll_interval_seconds > 0:
            await asyncio.sleep(poll_interval_seconds)
    yield _sse_event("stream_end", {"type": "stream_end", "reason": "max_events"})


@router.get("/jobs/stream")
async def stream_quant_jobs() -> StreamingResponse:
    """Stream lightweight quant job snapshots for the workflow page."""
    service = get_quant_job_service()
    return StreamingResponse(
        _quant_job_event_generator(service),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.get("/jobs/{job_id}", response_model=QuantJobResponse)
async def get_quant_job(job_id: str) -> QuantJobResponse:
    service = get_quant_job_service()
    try:
        job = service.get_job(job_id)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc
    return _job_response(job)


@router.get("/jobs/{job_id}/log", response_model=QuantJobLogResponse)
async def get_quant_job_log(
    job_id: str,
    max_bytes: int = Query(20000, ge=1, le=200000),
) -> QuantJobLogResponse:
    service = get_quant_job_service()
    try:
        payload = service.get_log(job_id, max_bytes=max_bytes)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc
    return QuantJobLogResponse(**payload)


@router.get("/jobs/{job_id}/artifacts/{artifact_index}")
async def get_quant_job_artifact(job_id: str, artifact_index: int) -> FileResponse:
    service = get_quant_job_service()
    try:
        job = service.get_job(job_id)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc

    if artifact_index < 0 or artifact_index >= len(job.artifact_paths):
        raise api_error(404, "not_found", f"Quant job artifact {artifact_index} was not found.")

    artifact_path = Path(job.artifact_paths[artifact_index])
    if not artifact_path.is_file():
        raise api_error(404, "not_found", f"Quant job artifact {artifact_index} is no longer available.")
    if not _is_allowed_artifact_path(artifact_path, service):
        raise api_error(403, "forbidden", f"Quant job artifact {artifact_index} is outside the allowed reports directory.")

    is_text_artifact = artifact_path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES
    try:
        return FileResponse(
            artifact_path,
            filename=artifact_path.name,
            media_type="text/plain" if is_text_artifact else None,
            content_disposition_type="inline" if is_text_artifact else "attachment",
        )
    except (OSError, RuntimeError) as exc:
        raise api_error(404, "not_found", f"Quant job artifact {artifact_index} is no longer available.") from exc


@router.post("/jobs/{job_id}/cancel", response_model=QuantJobResponse)
async def cancel_quant_job(job_id: str) -> QuantJobResponse:
    service = get_quant_job_service()
    try:
        job = service.cancel_job(job_id)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc
    return _job_response(job)
