# -*- coding: utf-8 -*-
"""
Quant Research API endpoints.

Provides CRUD operations for Research, Experiment, and Run entities,
plus Algorithm Registry and Signal contract endpoints.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from src.models.algorithm_registry import (
    algorithm_to_dict,
    get_algorithm,
    list_algorithms,
)
from src.repositories.quant_research_repo import QuantResearchRepository, validate_experiment_definition
from src.schemas.quant_research import (
    AlgorithmResponse,
    ExperimentCreate,
    ExperimentResponse,
    ExperimentUpdate,
    QuantResearchOverview,
    ResearchCreate,
    ResearchResponse,
    ResearchUpdate,
    RunCreate,
    RunResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Singleton repository instance
_repo: Optional[QuantResearchRepository] = None


def _get_repo() -> QuantResearchRepository:
    global _repo
    if _repo is None:
        _repo = QuantResearchRepository()
    return _repo


def _algorithm_exists(algorithm_id: str) -> bool:
    return get_algorithm(algorithm_id) is not None


# === Overview ===

@router.get("/overview", response_model=QuantResearchOverview)
async def get_overview():
    """Get quant research workspace overview."""
    repo = _get_repo()
    data = repo.get_overview()
    return QuantResearchOverview(**data)


# === Research CRUD ===

@router.get("/research", response_model=List[ResearchResponse])
async def list_research(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all research entries."""
    repo = _get_repo()
    return [ResearchResponse(**r) for r in repo.list_research(limit=limit, offset=offset)]


@router.post("/research", response_model=ResearchResponse, status_code=201)
async def create_research(body: ResearchCreate):
    """Create a new research entry."""
    repo = _get_repo()
    research = repo.create_research(
        name=body.name,
        description=body.description,
        research_type=body.research_type,
        config=body.config,
    )
    return ResearchResponse(**research)


@router.get("/research/{research_id}", response_model=ResearchResponse)
async def get_research(research_id: str):
    """Get a research entry with its experiments and runs. Returns 409 if any frozen experiment is tampered or semantically invalid."""
    repo = _get_repo()
    try:
        research = repo.get_research_with_experiments(research_id, _algorithm_exists)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if research is None:
        raise HTTPException(status_code=404, detail="Research not found")
    return ResearchResponse(**research)


@router.patch("/research/{research_id}", response_model=ResearchResponse)
async def update_research(research_id: str, body: ResearchUpdate):
    """Update a research entry."""
    repo = _get_repo()
    updates = body.model_dump(exclude_none=True)
    research = repo.update_research(research_id, **updates)
    if research is None:
        raise HTTPException(status_code=404, detail="Research not found")
    return ResearchResponse(**research)


@router.delete("/research/{research_id}", status_code=204)
async def delete_research(research_id: str):
    """Delete a research entry. Only allowed when all experiments are draft and have no runs."""
    repo = _get_repo()
    try:
        repo.delete_research(research_id)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=409, detail=msg)


# === Experiment CRUD ===

@router.get("/research/{research_id}/experiments", response_model=List[ExperimentResponse])
async def list_experiments(research_id: str):
    """List experiments under a research. Returns 409 if any frozen experiment is tampered or semantically invalid."""
    repo = _get_repo()
    research = repo.get_research(research_id)
    if research is None:
        raise HTTPException(status_code=404, detail="Research not found")
    try:
        experiments = repo.list_experiments_checked(research_id, _algorithm_exists)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return [ExperimentResponse(**e) for e in experiments]


@router.post("/research/{research_id}/experiments", response_model=ExperimentResponse, status_code=201)
async def create_experiment(research_id: str, body: ExperimentCreate):
    """Create an experiment under a research."""
    repo = _get_repo()
    research = repo.get_research(research_id)
    if research is None:
        raise HTTPException(status_code=404, detail="Research not found")
    try:
        validate_experiment_definition(body.method_kind, body.algorithm_id, _algorithm_exists)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    experiment = repo.create_experiment(
        research_id=research_id,
        name=body.name,
        description=body.description,
        experiment_type=body.experiment_type,
        method_kind=body.method_kind,
        algorithm_id=body.algorithm_id,
        protocol=body.protocol,
    )
    return ExperimentResponse(**experiment)


@router.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get an experiment with its runs. Returns 409 if frozen experiment is tampered or semantically invalid."""
    repo = _get_repo()
    try:
        experiment = repo.get_experiment_checked(experiment_id, _algorithm_exists)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    experiment["runs"] = repo.list_runs(experiment_id)
    return ExperimentResponse(**experiment)


@router.patch("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def update_experiment(experiment_id: str, body: ExperimentUpdate):
    """Update an experiment. Immutable fields, status rollback, and invalid definitions are rejected."""
    repo = _get_repo()
    updates = body.model_dump(exclude_unset=True)
    try:
        experiment = repo.update_experiment(experiment_id, algorithm_exists=_algorithm_exists, **updates)
    except ValueError as e:
        raise HTTPException(status_code=409 if "frozen" in str(e).lower() or "integrity" in str(e).lower() or "draft" in str(e).lower() else 400, detail=str(e))
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(**experiment)


@router.delete("/experiments/{experiment_id}", status_code=204)
async def delete_experiment(experiment_id: str):
    """Delete an experiment. Only allowed for draft experiments with no runs."""
    repo = _get_repo()
    try:
        repo.delete_experiment(experiment_id)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg.lower():
            raise HTTPException(status_code=404, detail=msg)
        raise HTTPException(status_code=409, detail=msg)


# === Run CRUD ===

@router.get("/experiments/{experiment_id}/runs", response_model=List[RunResponse])
async def list_runs(experiment_id: str):
    """List runs under an experiment."""
    repo = _get_repo()
    experiment = repo.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return [RunResponse(**r) for r in repo.list_runs(experiment_id)]


@router.post("/experiments/{experiment_id}/runs", response_model=RunResponse, status_code=201)
async def create_run(experiment_id: str, body: RunCreate):
    """Create a run under an experiment. Verifies experiment integrity and definition validity first."""
    repo = _get_repo()
    experiment = repo.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    try:
        run = repo.create_run(
            experiment_id=experiment_id,
            run_label=body.run_label,
            is_legacy=body.is_legacy,
            mlflow_run_id=body.mlflow_run_id,
            algorithm_exists=_algorithm_exists,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RunResponse(**run)


@router.get("/runs/{run_id}", response_model=RunResponse)
async def get_run(run_id: str):
    """Get a run."""
    repo = _get_repo()
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunResponse(**run)


# === Algorithm Registry ===

@router.get("/algorithms", response_model=List[AlgorithmResponse])
async def list_algorithms_endpoint(
    category: Optional[str] = Query(None, description="Filter by category: rule, linear, tree_ensemble"),
):
    """List available algorithms in the registry."""
    algorithms = list_algorithms(category=category)
    return [AlgorithmResponse(**algorithm_to_dict(a)) for a in algorithms]


@router.get("/algorithms/{algorithm_id}", response_model=AlgorithmResponse)
async def get_algorithm_endpoint(algorithm_id: str):
    """Get details of a specific algorithm."""
    algo = get_algorithm(algorithm_id)
    if algo is None:
        raise HTTPException(status_code=404, detail=f"Algorithm not found: {algorithm_id}")
    return AlgorithmResponse(**algorithm_to_dict(algo))
