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
    command: List[str] = Field(default_factory=list)
    cwd: str
    pid: Optional[int] = None
    return_code: Optional[int] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    updated_at: str
    error: Optional[str] = None
    log_path: str
    artifact_paths: List[str] = Field(default_factory=list)


class QuantJobListResponse(BaseModel):
    total: int
    items: List[QuantJobResponse] = Field(default_factory=list)


class QuantJobLogResponse(BaseModel):
    job_id: str
    content: str
    truncated: bool
    log_path: str


class QuantStatusResponse(BaseModel):
    available_job_types: Dict[str, Dict[str, Any]]
    active_count: int
    latest_jobs: Dict[str, QuantJobResponse]
    recent_jobs: List[QuantJobResponse] = Field(default_factory=list)
