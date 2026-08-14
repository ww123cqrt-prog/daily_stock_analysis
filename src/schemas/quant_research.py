# -*- coding: utf-8 -*-
"""
Quant Research domain schemas.

Defines Pydantic models for API request/response validation and the Signal contract.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# === Signal Contract ===

class Signal(BaseModel):
    """
    Minimal signal contract for stock selection.

    A signal represents a model's prediction or rule-based score for a stock
    on a specific date. It includes lineage metadata for traceability.
    """
    date: str = Field(..., description="Signal date in YYYY-MM-DD format")
    symbol: str = Field(..., description="Stock symbol (e.g., SH600519)")
    score: float = Field(..., description="Model score or rule output")

    # Optional enrichment
    rank: Optional[int] = Field(None, description="Rank among all signals for this date (1=highest)")
    expected_return: Optional[float] = Field(None, description="Expected return (e.g., 0.05 for 5%)")
    confidence: Optional[float] = Field(None, description="Model confidence [0, 1]")

    # Lineage metadata
    producer_id: Optional[str] = Field(None, description="Experiment or run ID that produced this signal")
    producer_type: Optional[str] = Field(None, description="Type of producer: 'algorithm', 'rule', 'ensemble'")
    algorithm_id: Optional[str] = Field(None, description="Algorithm ID from registry")
    protocol_hash: Optional[str] = Field(None, description="Hash of the frozen protocol")

    class Config:
        json_schema_extra = {
            "example": {
                "date": "2025-06-30",
                "symbol": "SH600519",
                "score": 0.85,
                "rank": 1,
                "expected_return": 0.03,
                "confidence": 0.72,
                "producer_id": "abc-123",
                "producer_type": "algorithm",
                "algorithm_id": "lightgbm",
            }
        }


class SignalBatch(BaseModel):
    """A batch of signals with metadata."""
    signals: List[Signal]
    producer_id: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    count: int = 0

    def model_post_init(self, __context: Any) -> None:
        self.count = len(self.signals)


# === Research CRUD Schemas ===

class ResearchCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    research_type: str = Field("stock_selection", pattern="^(stock_selection|strategy)$")
    config: Dict[str, Any] = Field(default_factory=dict)


class ResearchUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    status: Optional[str] = Field(None, pattern="^(draft|active|completed|archived)$")
    research_type: Optional[str] = Field(None, pattern="^(stock_selection|strategy)$")
    config: Optional[Dict[str, Any]] = None


class ResearchResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    research_type: str
    config: Dict[str, Any]
    created_at: str
    updated_at: str
    experiments: Optional[List[Dict[str, Any]]] = None


# === Experiment CRUD Schemas ===

class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field("", max_length=2000)
    experiment_type: str = Field("stock_selection", pattern="^(stock_selection|strategy)$")
    method_kind: str = Field("algorithm", pattern="^(rule|algorithm)$")
    algorithm_id: Optional[str] = None
    protocol: Dict[str, Any] = Field(default_factory=dict)


class ExperimentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    experiment_type: Optional[str] = Field(None, pattern="^(stock_selection|strategy)$")
    method_kind: Optional[str] = Field(None, pattern="^(rule|algorithm)$")
    algorithm_id: Optional[str] = None
    protocol: Optional[Dict[str, Any]] = None
    status: Optional[str] = Field(None, pattern="^(draft|submitted|running|completed|failed|cancelled)$")


class ExperimentResponse(BaseModel):
    id: str
    research_id: str
    name: str
    description: str
    experiment_type: str
    method_kind: str
    algorithm_id: Optional[str]
    protocol: Dict[str, Any]
    protocol_hash: Optional[str] = None
    status: str
    created_at: str
    updated_at: str
    runs: Optional[List[Dict[str, Any]]] = None


# === Run Schemas ===

class RunCreate(BaseModel):
    run_label: str = Field("", max_length=200)
    is_legacy: bool = False
    mlflow_run_id: Optional[str] = None


class RunResponse(BaseModel):
    id: str
    experiment_id: str
    run_label: str
    status: str
    started_at: Optional[str]
    completed_at: Optional[str]
    metrics: Dict[str, Any]
    artifacts: List[Any]
    error_message: Optional[str]
    is_legacy: bool
    mlflow_run_id: Optional[str]
    created_at: str
    updated_at: str


# === Algorithm Registry Schemas ===

class AlgorithmCapabilityResponse(BaseModel):
    supports_classification: bool
    supports_regression: bool
    supports_feature_importance: bool
    supports_online_learning: bool
    requires_gpu: bool
    max_features: Optional[int]
    supported_loss_functions: List[str]


class AlgorithmConfigSchemaResponse(BaseModel):
    parameters: Dict[str, Dict[str, Any]]
    required: List[str]


class AlgorithmResponse(BaseModel):
    id: str
    name: str
    name_zh: str
    category: str
    description: str
    description_zh: str
    capability: AlgorithmCapabilityResponse
    config_schema: AlgorithmConfigSchemaResponse
    default_params: Dict[str, Any]


# === Overview Schema ===

class QuantResearchOverview(BaseModel):
    total_research: int
    total_experiments: int
    total_runs: int
    recent_research: List[ResearchResponse]
