# -*- coding: utf-8 -*-
"""Quant workflow job endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse, urlunparse
from urllib.request import Request, urlopen

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, StreamingResponse

from api.v1.errors import api_error
from api.v1.schemas.quant_jobs import (
    QuantJobListResponse,
    QuantJobLogResponse,
    QuantJobResponse,
    QuantJobSubmitRequest,
    QuantJobTypesResponse,
    QuantStatusResponse,
)
from api.v1.endpoints.research_status import (
    approved_ashare_qfq_snapshot_identity,
    approved_robust_signal_data_publication,
    build_research_status_payload,
)
from quant_workflow.scripts.quant_data_contract import DataContractError
from quant_workflow.scripts._mlflow_run_provenance import (
    RunProvenanceError,
    normalize_provider_lineage,
)
from quant_workflow.scripts.robust_signal_data_contract import (
    RobustSignalDataBlocked,
    readiness_artifact_path_for_sha,
)
from src.safe_errors import sanitize_public_error
from src.services.quant_job_service import QuantJob, QuantJobService, REPORTS_ROOT
from src.safe_urls import public_http_url


router = APIRouter()
logger = logging.getLogger(__name__)
JOB_STREAM_POLL_INTERVAL_SECONDS = 2.0
JOB_STREAM_MAX_JOB_EVENTS = 300
TEXT_ARTIFACT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".jsonl", ".log"}
DEFAULT_MLFLOW_URL = "http://127.0.0.1:6006/"
DEFAULT_CATDESK_URL = "http://127.0.0.1:8765/"
MLFLOW_HEALTH_CACHE_TTL_SECONDS = 15.0
MLFLOW_HEALTH_CACHE_MAX_ENTRIES = 8
LOCAL_MLFLOW_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOCAL_MLFLOW_HEALTH_PORTS = {None, 6006, 6606}
LOCAL_CATDESK_HOSTS = {"127.0.0.1", "localhost", "::1"}
FORMAL_TRAINING_JOB_TYPES = {"train_baseline_lgb", "run_candidate_pipeline"}
PROVIDER_PINNED_TRAINING_JOB_TYPES = FORMAL_TRAINING_JOB_TYPES | {
    "train_baseline_lgb_equivalence"
}
READINESS_GATED_JOB_TYPES = PROVIDER_PINNED_TRAINING_JOB_TYPES
ROBUST_SIGNAL_READINESS_GATED_JOB_TYPES = {
    "run_robust_signal_research",
    "resume_robust_signal_research",
}
MARKET_DATA_PINNED_JOB_TYPES = {
    "report_market_regime",
    "report_cash_control_backtest",
}
TRAINING_PROVIDER_DATASET_IDS = {
    "canonical": "ashare_daily_qfq",
    "cleaned": "ashare_daily_qfq",
}
LOCAL_CATDESK_HEALTH_PORTS = {8765}
_MLFLOW_STATUS_CACHE_LOCK = threading.Lock()
_MLFLOW_STATUS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_QUANT_JOB_STREAM_LOCK = threading.Lock()
_QUANT_JOB_STREAM_ACTIVE = 0
QUANT_API_CONTRACT_VERSION = "quant-workbench-v2"
BACKEND_STARTED_AT = datetime.now(timezone.utc).isoformat()
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class _ArtifactPathResolution(NamedTuple):
    path: Optional[Path]
    is_forbidden: bool = False


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        return default


def _float_env(name: str, default: float, *, minimum: float = 0.1) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


MAX_QUANT_JOB_STREAMS = _int_env("MAX_QUANT_JOB_STREAMS", 8)
MLFLOW_HEALTH_TIMEOUT_SECONDS = _float_env("MLFLOW_HEALTH_TIMEOUT_SECONDS", 2.0)
CATDESK_HEALTH_TIMEOUT_SECONDS = _float_env("CATDESK_HEALTH_TIMEOUT_SECONDS", 0.8)


def _mlflow_url() -> str:
    return os.environ.get("MLFLOW_URL", DEFAULT_MLFLOW_URL).strip() or DEFAULT_MLFLOW_URL


def _catdesk_url() -> str:
    return os.environ.get("CATDESK_URL", DEFAULT_CATDESK_URL).strip() or DEFAULT_CATDESK_URL


def _public_mlflow_url(url: str) -> str:
    return public_http_url(url, DEFAULT_MLFLOW_URL)


def _public_catdesk_url(url: str) -> str:
    return public_http_url(url, DEFAULT_CATDESK_URL)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_backend_build_id() -> str:
    configured = os.environ.get("DSA_BACKEND_BUILD_ID", "").strip()
    if configured:
        return configured[:128]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "backend-local"
    return completed.stdout.strip() or "backend-local"


BACKEND_BUILD_ID = _resolve_backend_build_id()


def _api_contract_payload() -> Dict[str, str]:
    return {
        "version": QUANT_API_CONTRACT_VERSION,
        "backend_build_id": BACKEND_BUILD_ID,
        "backend_started_at": BACKEND_STARTED_AT,
    }


def _sanitize_public_error(error: Optional[str]) -> Optional[str]:
    return sanitize_public_error(error)


def _fresh_cached_mlflow_payload(url: str, now: float) -> Optional[Dict[str, Any]]:
    cached = _MLFLOW_STATUS_CACHE.get(url)
    if not cached or now - cached[0] > MLFLOW_HEALTH_CACHE_TTL_SECONDS:
        return None
    payload = cached[1]
    if payload.get("status") == "available":
        return None
    return dict(payload)


def _store_mlflow_status_cache(url: str, payload: Dict[str, Any], store_time: float) -> None:
    if payload.get("status") == "available":
        _MLFLOW_STATUS_CACHE.pop(url, None)
        return
    if len(_MLFLOW_STATUS_CACHE) >= MLFLOW_HEALTH_CACHE_MAX_ENTRIES and url not in _MLFLOW_STATUS_CACHE:
        oldest_url = min(_MLFLOW_STATUS_CACHE, key=lambda key: _MLFLOW_STATUS_CACHE[key][0])
        _MLFLOW_STATUS_CACHE.pop(oldest_url, None)
    _MLFLOW_STATUS_CACHE[url] = (store_time, dict(payload))


def _mlflow_status_payload() -> Dict[str, Any]:
    url = _mlflow_url()
    now = time.monotonic()
    with _MLFLOW_STATUS_CACHE_LOCK:
        cached_payload = _fresh_cached_mlflow_payload(url, now)
        if cached_payload is not None:
            return cached_payload

    payload = _probe_mlflow_status_payload(url)
    store_time = time.monotonic()
    with _MLFLOW_STATUS_CACHE_LOCK:
        cached_payload = _fresh_cached_mlflow_payload(url, store_time)
        if cached_payload is not None:
            return cached_payload
        _store_mlflow_status_cache(url, payload, store_time)
    return payload


def _clear_mlflow_status_cache() -> None:
    with _MLFLOW_STATUS_CACHE_LOCK:
        _MLFLOW_STATUS_CACHE.clear()


def _probe_mlflow_status_payload(url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "url": _public_mlflow_url(url),
        "status": "unavailable",
        "role": "experiment_raw_records",
        "checked_at": _now_iso(),
        "health_checked": False,
    }
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL must be an http(s) URL."
        return payload
    try:
        parsed_port = parsed.port
    except ValueError:
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL port is invalid."
        return payload
    if parsed.username or parsed.password:
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL must not include username or password."
        return payload
    decoded_path = unquote(parsed.path or "")
    if any(part == ".." for part in decoded_path.replace("\\", "/").split("/")):
        payload["status"] = "invalid"
        payload["error"] = "MLFLOW_URL must not contain path traversal."
        return payload
    # This local-only probe avoids remote SSRF by refusing non-loopback hostnames
    # and by limiting local health checks to the MLflow ports DSA owns.
    if parsed.hostname not in LOCAL_MLFLOW_HOSTS:
        payload["status"] = "linked"
        payload["error"] = "Health check intentionally skipped for non-local MLflow URL; link is displayed without availability verification."
        return payload
    if parsed_port not in LOCAL_MLFLOW_HEALTH_PORTS:
        payload["status"] = "invalid"
        payload["error"] = "Local MLflow health checks are limited to ports 6006 and 6606."
        return payload

    probe_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path or "/", "", "", ""))
    request = Request(probe_url, headers={"User-Agent": "DSA-MLflow-health-check/1.0"})
    payload["health_checked"] = True
    try:
        with urlopen(request, timeout=MLFLOW_HEALTH_TIMEOUT_SECONDS) as response:
            status_code = response.getcode()
    except HTTPError as exc:
        status_code = exc.code
    except (OSError, TimeoutError, URLError, ValueError) as exc:
        payload["error"] = _sanitize_public_error(str(exc))
        return payload

    payload["status"] = "available" if 200 <= int(status_code) < 400 else "unavailable"
    if payload["status"] == "unavailable":
        payload["error"] = f"MLflow returned HTTP {status_code}."
    return payload


def _catdesk_status_payload() -> Dict[str, Any]:
    return _probe_catdesk_status_payload(_catdesk_url())


def _probe_catdesk_status_payload(url: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "url": _public_catdesk_url(url),
        "status": "unavailable",
        "role": "external_catalyst_dashboard",
        "checked_at": _now_iso(),
        "health_checked": False,
    }
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        payload["status"] = "invalid"
        payload["error"] = "CATDESK_URL must be an http(s) URL."
        return payload
    try:
        parsed_port = parsed.port
    except ValueError:
        payload["status"] = "invalid"
        payload["error"] = "CATDESK_URL port is invalid."
        return payload
    if parsed.username or parsed.password:
        payload["status"] = "invalid"
        payload["error"] = "CATDESK_URL must not include username or password."
        return payload
    decoded_path = unquote(parsed.path or "")
    if any(part == ".." for part in decoded_path.replace("\\", "/").split("/")):
        payload["status"] = "invalid"
        payload["error"] = "CATDESK_URL must not contain path traversal."
        return payload
    if parsed.hostname not in LOCAL_CATDESK_HOSTS:
        payload["status"] = "linked"
        payload["error"] = "Health check intentionally skipped for non-local CatDesk URL; link is displayed without availability verification."
        return payload
    if parsed_port not in LOCAL_CATDESK_HEALTH_PORTS:
        payload["status"] = "invalid"
        payload["error"] = "Local CatDesk health checks are limited to port 8765."
        return payload

    payload["health_checked"] = True
    try:
        with socket.create_connection((parsed.hostname, parsed_port), timeout=CATDESK_HEALTH_TIMEOUT_SECONDS):
            payload["status"] = "available"
    except (OSError, TimeoutError, ValueError) as exc:
        payload["error"] = _sanitize_public_error(str(exc))
    return payload


def _dsa_status_payload() -> Dict[str, Any]:
    return {
        "url": "/",
        "status": "available",
        "role": "local_web_api",
        "checked_at": _now_iso(),
        "health_checked": True,
    }


def _external_services_payload(mlflow_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        "dsa": _dsa_status_payload(),
        "mlflow": mlflow_payload,
        "catdesk": _catdesk_status_payload(),
    }


@lru_cache(maxsize=1)
def get_quant_job_service() -> QuantJobService:
    return QuantJobService()


def _clear_quant_job_service_cache() -> None:
    get_quant_job_service.cache_clear()


def _sanitize_job_error(error: Optional[str]) -> Optional[str]:
    return _sanitize_public_error(error)


def _job_public_payload(job: QuantJob) -> Dict[str, Any]:
    payload = job.to_dict()
    payload.pop("command", None)
    payload.pop("cwd", None)
    payload.pop("log_path", None)
    payload["error"] = _sanitize_job_error(job.error)
    payload["artifact_paths"] = [Path(path).name for path in job.artifact_paths]
    return payload


def _job_response(job: QuantJob) -> QuantJobResponse:
    payload = _job_public_payload(job)
    return QuantJobResponse(**payload)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _requires_training_readiness_gate(
    job_type: str, params: Dict[str, Any]
) -> bool:
    if job_type not in READINESS_GATED_JOB_TYPES:
        return False
    if job_type == "run_candidate_pipeline":
        return not _is_true(params.get("dry_run"))
    return True


def _approved_robust_signal_submission_data(
    job_type: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    if job_type != "resume_robust_signal_research":
        return approved_robust_signal_data_publication()

    run_id = params.get("run_id")
    if (
        not isinstance(run_id, str)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,256}", run_id) is None
    ):
        raise ValueError("A valid run_id is required to resume robust signal research.")
    context_path = REPORTS_ROOT / "robust_signal" / "runs" / run_id / "context.json"
    try:
        if (
            context_path.is_symlink()
            or not context_path.is_file()
            or context_path.stat().st_size > 1024 * 1024
        ):
            raise ValueError
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            "The original robust signal run has no valid frozen data context."
        ) from exc
    artifact_sha256 = (
        context.get("readiness_artifact_sha256") if isinstance(context, dict) else None
    )
    if (
        not isinstance(artifact_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) is None
    ):
        raise ValueError(
            "The original robust signal run has no frozen readiness artifact hash."
        )
    pointer_path = (
        REPORTS_ROOT / "robust_signal" / "robust_signal_data_readiness_latest.json"
    )
    artifact_path = readiness_artifact_path_for_sha(pointer_path, artifact_sha256)
    return approved_robust_signal_data_publication(
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256,
    )


def _training_provider_dataset_id(params: Dict[str, Any]) -> str:
    provider = str(params.get("provider") or "canonical").strip().lower()
    dataset_id = TRAINING_PROVIDER_DATASET_IDS.get(provider)
    if dataset_id is None:
        raise ValueError("Provider must be one of: canonical, cleaned.")
    return dataset_id


def _training_provider_name(params: Dict[str, Any]) -> str:
    provider = str(params.get("provider") or "canonical").strip().lower()
    if provider not in TRAINING_PROVIDER_DATASET_IDS:
        raise ValueError("Provider must be one of: canonical, cleaned.")
    return provider


def _approved_provider_identity_from_status(
    research_status: Dict[str, Any],
) -> Dict[str, str]:
    provider = research_status.get("qlib_provider")
    if not isinstance(provider, dict):
        raise ValueError("Training status does not expose an approved Qlib Provider.")
    try:
        return normalize_provider_lineage(provider)
    except RunProvenanceError as exc:
        raise ValueError("Training status has an invalid Qlib Provider lineage.") from exc


def _resolved_allowed_artifact_path(path: Path, service: QuantJobService) -> _ArtifactPathResolution:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        return _ArtifactPathResolution(path=None)
    allowed_roots = {REPORTS_ROOT, service.jobs_root.parent}
    for root in allowed_roots:
        try:
            resolved.relative_to(root.expanduser().resolve())
            return _ArtifactPathResolution(path=resolved if resolved.is_file() else None)
        except ValueError:
            continue
    return _ArtifactPathResolution(path=None, is_forbidden=True)


@router.get("/status", response_model=QuantStatusResponse)
async def get_quant_status(
    include_research: bool = Query(
        True,
        description="Include the full research/data readiness snapshot.",
    ),
) -> QuantStatusResponse:
    """Return quant workflow status used by the web workflow page."""
    service = get_quant_job_service()
    if include_research:
        research_status, mlflow_payload = await asyncio.gather(
            asyncio.to_thread(build_research_status_payload),
            asyncio.to_thread(_mlflow_status_payload),
        )
    else:
        research_status = {}
        mlflow_payload = await asyncio.to_thread(_mlflow_status_payload)
    # Research and service probes can be slow. Read mutable job state last so
    # the response does not report a task as active after it finished while
    # the immutable enrichment data was loading.
    status_snapshot = await asyncio.to_thread(service.get_status_snapshot)
    latest_job_snapshot, active_count, recent_jobs = status_snapshot
    latest_jobs = {
        job_type: _job_response(job)
        for job_type, job in latest_job_snapshot.items()
    }
    return QuantStatusResponse(
        api_contract=_api_contract_payload(),
        available_job_types=service.available_job_types(),
        active_count=active_count,
        latest_jobs=latest_jobs,
        recent_jobs=[_job_response(job) for job in recent_jobs],
        generated_at=research_status.get("generated_at"),
        trainable=bool(research_status.get("trainable")),
        training_readiness=research_status.get("training_readiness") or {},
        data_sources=research_status.get("data_sources") or [],
        qlib_provider=research_status.get("qlib_provider"),
        provider_quality=research_status.get("provider_quality") or {},
        current_provider_identities=research_status.get(
            "current_provider_identities"
        )
        or {},
        latest_training=research_status.get("latest_training"),
        cash_control_backtest=research_status.get("cash_control_backtest"),
        qlib_predictions=research_status.get("qlib_predictions"),
        qlib_research=research_status.get("qlib_research"),
        robust_strategy=research_status.get("robust_strategy"),
        robust_signal=research_status.get("robust_signal"),
        experiment_notebook=research_status.get("experiment_notebook"),
        mlflow=mlflow_payload,
        external_services=_external_services_payload(mlflow_payload),
    )


@router.get("/job-types", response_model=QuantJobTypesResponse)
async def get_quant_job_types() -> QuantJobTypesResponse:
    """Return the static task whitelist without loading research artifacts."""

    service = get_quant_job_service()
    available = await asyncio.to_thread(service.available_job_types)
    return QuantJobTypesResponse(
        available_job_types=available,
        api_contract=_api_contract_payload(),
    )


@router.post("/jobs", response_model=QuantJobResponse)
async def submit_quant_job(request: QuantJobSubmitRequest) -> QuantJobResponse:
    """Submit one whitelisted quant workflow job."""
    service = get_quant_job_service()
    approved_provider_identity = None
    approved_market_data_identity = None
    approved_robust_signal_data = None
    try:
        if request.job_type in MARKET_DATA_PINNED_JOB_TYPES:
            try:
                approved_market_data_identity = await asyncio.to_thread(
                    approved_ashare_qfq_snapshot_identity
                )
            except DataContractError as exc:
                raise api_error(
                    409,
                    "market_data_blocked",
                    "A股日线数据尚未通过完整验收，或验收后的快照已经变化。请先在数据页重新运行完整数据质量报告。",
                ) from exc
        if request.job_type in ROBUST_SIGNAL_READINESS_GATED_JOB_TYPES:
            try:
                approved_robust_signal_data = await asyncio.to_thread(
                    _approved_robust_signal_submission_data,
                    request.job_type,
                    request.params,
                )
            except RobustSignalDataBlocked as exc:
                raise api_error(
                    409,
                    "robust_signal_data_blocked",
                    exc.summary_zh,
                ) from exc
            except ValueError as exc:
                raise api_error(
                    409,
                    "robust_signal_resume_blocked",
                    str(exc),
                ) from exc
        elif _requires_training_readiness_gate(request.job_type, request.params):
            provider_dataset_id = _training_provider_dataset_id(request.params)
            research_status = await asyncio.to_thread(
                build_research_status_payload,
                provider_dataset_id=provider_dataset_id,
                provider=_training_provider_name(request.params),
            )
            if research_status.get("trainable") is not True:
                readiness = research_status.get("training_readiness")
                summary = (
                    readiness.get("summary")
                    if isinstance(readiness, dict)
                    else None
                )
                raise api_error(
                    409,
                    "training_blocked",
                    str(
                        summary
                        or "Current data quality evidence does not allow formal training."
                    ),
                )
            if request.job_type in PROVIDER_PINNED_TRAINING_JOB_TYPES:
                try:
                    approved_provider_identity = _approved_provider_identity_from_status(
                        research_status,
                    )
                except ValueError as exc:
                    raise api_error(
                        409,
                        "training_blocked",
                        "Current data quality evidence cannot pin an approved Qlib snapshot.",
                    ) from exc
        submit_kwargs = {}
        if approved_provider_identity is not None:
            submit_kwargs["approved_provider_identity"] = approved_provider_identity
        if approved_market_data_identity is not None:
            submit_kwargs["approved_market_data_identity"] = (
                approved_market_data_identity
            )
        if approved_robust_signal_data is not None:
            submit_kwargs["approved_robust_signal_data"] = approved_robust_signal_data
        job = await asyncio.to_thread(
            service.submit_job,
            request.job_type,
            request.params,
            **submit_kwargs,
        )
    except ValueError as exc:
        raise api_error(400, "invalid_params", str(exc)) from exc
    return _job_response(job)


@router.get("/jobs", response_model=QuantJobListResponse)
async def list_quant_jobs(
    limit: int = Query(50, ge=1, le=200),
    job_type: Optional[str] = Query(None),
) -> QuantJobListResponse:
    service = get_quant_job_service()
    jobs = await asyncio.to_thread(
        service.list_jobs, limit=limit, job_type=job_type
    )
    return QuantJobListResponse(total=len(jobs), items=[_job_response(job) for job in jobs])


def _sse_event(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _list_jobs_for_stream(service: QuantJobService) -> list[QuantJob]:
    """Fetch stream snapshots without blocking the event loop on SQLite I/O."""
    return await asyncio.to_thread(
        service.list_jobs_for_monitoring, history_limit=20
    )


async def _quant_job_event_generator(
    service: QuantJobService,
    *,
    max_job_events: int = JOB_STREAM_MAX_JOB_EVENTS,
    poll_interval_seconds: float = JOB_STREAM_POLL_INTERVAL_SECONDS,
):
    yield _sse_event("connected", {"type": "connected"})
    try:
        for _ in range(max(1, int(max_job_events))):
            jobs = [_job_public_payload(job) for job in await _list_jobs_for_stream(service)]
            yield _sse_event("jobs", {"type": "jobs", "jobs": jobs})
            if poll_interval_seconds > 0:
                await asyncio.sleep(poll_interval_seconds)
    except Exception:  # noqa: BLE001 - keep the browser stream self-describing.
        logger.exception("Unable to stream quant job updates.")
        yield _sse_event(
            "stream_end",
            {"type": "stream_end", "reason": "error", "error": "Unable to stream quant job updates."},
        )
        return
    yield _sse_event("stream_end", {"type": "stream_end", "reason": "max_events"})


def _try_acquire_stream_slot() -> bool:
    global _QUANT_JOB_STREAM_ACTIVE
    with _QUANT_JOB_STREAM_LOCK:
        if _QUANT_JOB_STREAM_ACTIVE >= MAX_QUANT_JOB_STREAMS:
            return False
        _QUANT_JOB_STREAM_ACTIVE += 1
        return True


def _release_stream_slot() -> None:
    global _QUANT_JOB_STREAM_ACTIVE
    with _QUANT_JOB_STREAM_LOCK:
        _QUANT_JOB_STREAM_ACTIVE = max(0, _QUANT_JOB_STREAM_ACTIVE - 1)


async def _limited_quant_job_event_generator(service: QuantJobService):
    try:
        async for event in _quant_job_event_generator(service):
            yield event
    finally:
        _release_stream_slot()


@router.get("/jobs/stream")
async def stream_quant_jobs() -> StreamingResponse:
    """Stream lightweight quant job snapshots for the workflow page."""
    service = get_quant_job_service()
    if not _try_acquire_stream_slot():
        raise api_error(429, "too_many_streams", "Too many quant job stream connections.")
    try:
        return StreamingResponse(
            _limited_quant_job_event_generator(service),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Content-Type-Options": "nosniff",
            },
        )
    except Exception:
        _release_stream_slot()
        raise


@router.get("/jobs/{job_id}", response_model=QuantJobResponse)
async def get_quant_job(job_id: str) -> QuantJobResponse:
    service = get_quant_job_service()
    try:
        job = await asyncio.to_thread(service.get_job, job_id)
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
        payload = await asyncio.to_thread(
            service.get_log, job_id, max_bytes=max_bytes
        )
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc
    payload.pop("log_path", None)
    return QuantJobLogResponse(**payload)


@router.get("/jobs/{job_id}/artifacts/{artifact_index}")
async def get_quant_job_artifact(job_id: str, artifact_index: int) -> FileResponse:
    service = get_quant_job_service()
    try:
        job = await asyncio.to_thread(service.get_job, job_id)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc

    if artifact_index < 0 or artifact_index >= len(job.artifact_paths):
        raise api_error(404, "not_found", f"Quant job artifact {artifact_index} was not found.")

    artifact_path = Path(job.artifact_paths[artifact_index])
    artifact_resolution = await asyncio.to_thread(
        _resolved_allowed_artifact_path, artifact_path, service
    )
    if artifact_resolution.is_forbidden:
        raise api_error(403, "forbidden", f"Quant job artifact {artifact_index} is outside the allowed reports directory.")
    if artifact_resolution.path is None:
        raise api_error(404, "not_found", f"Quant job artifact {artifact_index} is no longer available.")

    is_text_artifact = artifact_path.suffix.lower() in TEXT_ARTIFACT_SUFFIXES
    try:
        return FileResponse(
            artifact_resolution.path,
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
        job = await asyncio.to_thread(service.cancel_job, job_id)
    except KeyError as exc:
        raise api_error(404, "not_found", f"Quant job {job_id} was not found.") from exc
    return _job_response(job)
