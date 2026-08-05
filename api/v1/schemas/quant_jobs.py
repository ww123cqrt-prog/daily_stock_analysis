# -*- coding: utf-8 -*-
"""Quant workflow job API schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuantJobSubmitRequest(BaseModel):
    job_type: str = Field(..., description="白名单量化任务类型")
    params: Dict[str, Any] = Field(default_factory=dict, description="白名单任务参数")


class QuantJobResponse(BaseModel):
    job_id: str
    job_type: str
    label: str
    status: str
    params: Dict[str, Any] = Field(default_factory=dict)
    pid: Optional[int] = None
    return_code: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
    error: Optional[str] = None
    artifact_paths: List[str] = Field(default_factory=list)
    progress: Dict[str, Any] = Field(default_factory=dict)
    runner_alive: Optional[bool] = None
    can_cancel: bool = False
    log_updated_at: Optional[str] = None


class QuantJobListResponse(BaseModel):
    total: int
    items: List[QuantJobResponse] = Field(default_factory=list)


class QuantJobLogResponse(BaseModel):
    job_id: str
    content: str
    truncated: bool


class QuantJobTypesResponse(BaseModel):
    available_job_types: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    api_contract: Dict[str, str] = Field(default_factory=dict)


class QuantStatusResponse(BaseModel):
    api_contract: Dict[str, str] = Field(default_factory=dict)
    available_job_types: Dict[str, Dict[str, Any]]
    active_count: int
    latest_jobs: Dict[str, QuantJobResponse]
    recent_jobs: List[QuantJobResponse] = Field(default_factory=list)
    generated_at: Optional[str] = None
    trainable: bool = False
    training_readiness: Dict[str, Any] = Field(default_factory=dict)
    data_sources: List[Dict[str, Any]] = Field(default_factory=list)
    qlib_provider: Optional[Dict[str, Any]] = None
    provider_quality: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    latest_training: Optional[Dict[str, Any]] = None
    cash_control_backtest: Optional[Dict[str, Any]] = None
    qlib_predictions: Optional[Dict[str, Any]] = None
    qlib_research: Optional[Dict[str, Any]] = None
    robust_strategy: Optional[Dict[str, Any]] = None
    robust_signal: Optional[Dict[str, Any]] = None
    experiment_notebook: Optional[Dict[str, Any]] = None
    mlflow: Dict[str, Any] = Field(default_factory=dict)
    external_services: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
