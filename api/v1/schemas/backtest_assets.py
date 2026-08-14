# -*- coding: utf-8 -*-
"""
Pydantic schemas for strategy asset persistence and backtest run history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# === Strategy assets ===


class StrategyAssetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    script: str = Field(..., min_length=1)


class StrategyAssetUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    script: Optional[str] = Field(None, min_length=1)


class StrategyAssetSummary(BaseModel):
    id: str
    name: str
    description: str
    script_sha256: str
    api_version: str
    created_at: str
    updated_at: str


class StrategyAssetDetail(StrategyAssetSummary):
    script: str


# === Backtest runs ===


class BacktestRunSummary(BaseModel):
    id: str
    strategy_id: Optional[str] = None
    strategy_name: str
    script_sha256: str
    symbols: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    status: str
    error_message: Optional[str] = None
    created_at: str


class BacktestRunDetail(BacktestRunSummary):
    report: Dict[str, Any] = Field(default_factory=dict)
    script: str = Field(
        default="",
        description="冻结的脚本源码，用于历史运行的精确重放",
    )
