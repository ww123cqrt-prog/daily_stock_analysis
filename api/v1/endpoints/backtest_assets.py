# -*- coding: utf-8 -*-
"""
Backtest asset endpoints.

Named scripted strategies (save / list / load / update / delete) and the
durable backtest run history. Runs are written by the script backtest
endpoint in ``backtest.py``; this module only reads them back for the
workbench UI, so a persistence failure can never break a backtest.
"""

from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, HTTPException, Query

from src.core.script_strategy import SCRIPT_API_VERSION, validate_strategy_script
from src.repositories.strategy_assets_repo import get_strategy_assets_repo
from api.v1.schemas.backtest_assets import (
    BacktestRunDetail,
    BacktestRunSummary,
    StrategyAssetCreate,
    StrategyAssetDetail,
    StrategyAssetSummary,
    StrategyAssetUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _script_sha256(script: str) -> str:
    return hashlib.sha256(script.encode("utf-8")).hexdigest()


@router.get("/strategies", response_model=list[StrategyAssetSummary])
async def list_strategies() -> list[StrategyAssetSummary]:
    """List saved named strategies (newest first, script bodies excluded)."""
    repo = get_strategy_assets_repo()
    return [StrategyAssetSummary(**item) for item in repo.list_strategies()]


@router.post("/strategies", response_model=StrategyAssetDetail, status_code=201)
async def create_strategy(body: StrategyAssetCreate) -> StrategyAssetDetail:
    """Save a named strategy. The script must pass the static sandbox check."""
    try:
        validate_strategy_script(body.script)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"脚本未通过静态检查：{exc}") from exc
    repo = get_strategy_assets_repo()
    saved = repo.create_strategy(
        name=body.name,
        script=body.script,
        script_sha256=_script_sha256(body.script),
        description=body.description,
        api_version=SCRIPT_API_VERSION,
    )
    return StrategyAssetDetail(**saved)


@router.get("/strategies/{strategy_id}", response_model=StrategyAssetDetail)
async def get_strategy(strategy_id: str) -> StrategyAssetDetail:
    repo = get_strategy_assets_repo()
    saved = repo.get_strategy(strategy_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="策略不存在或已被删除。")
    return StrategyAssetDetail(**saved)


@router.patch("/strategies/{strategy_id}", response_model=StrategyAssetDetail)
async def update_strategy(strategy_id: str, body: StrategyAssetUpdate) -> StrategyAssetDetail:
    repo = get_strategy_assets_repo()
    if repo.get_strategy(strategy_id) is None:
        raise HTTPException(status_code=404, detail="策略不存在或已被删除。")
    updates: dict = body.model_dump(exclude_unset=True)
    if "script" in updates and updates["script"] is not None:
        try:
            validate_strategy_script(updates["script"])
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"脚本未通过静态检查：{exc}") from exc
        updates["script_sha256"] = _script_sha256(updates["script"])
        updates["api_version"] = SCRIPT_API_VERSION
    saved = repo.update_strategy(strategy_id, **updates)
    return StrategyAssetDetail(**saved)


@router.delete("/strategies/{strategy_id}", status_code=204)
async def delete_strategy(strategy_id: str) -> None:
    repo = get_strategy_assets_repo()
    if not repo.delete_strategy(strategy_id):
        raise HTTPException(status_code=404, detail="策略不存在或已被删除。")


@router.get("/runs", response_model=list[BacktestRunSummary])
async def list_backtest_runs(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    strategy_id: str | None = Query(None),
    script_sha256: str | None = Query(None),
) -> list[BacktestRunSummary]:
    """List backtest runs (newest first). Summaries exclude equity curves and trades."""
    repo = get_strategy_assets_repo()
    runs = repo.list_runs(
        limit=limit,
        offset=offset,
        strategy_id=strategy_id,
        script_sha256=script_sha256,
    )
    return [BacktestRunSummary(**item) for item in runs]


@router.get("/runs/{run_id}", response_model=BacktestRunDetail)
async def get_backtest_run(run_id: str) -> BacktestRunDetail:
    """Full run detail including the complete backtest report."""
    repo = get_strategy_assets_repo()
    run = repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="回测运行记录不存在。")
    return BacktestRunDetail(**run)


@router.delete("/runs/{run_id}", status_code=204)
async def delete_backtest_run(run_id: str) -> None:
    """Delete one historical run record."""
    repo = get_strategy_assets_repo()
    if not repo.delete_run(run_id):
        raise HTTPException(status_code=404, detail="回测运行记录不存在。")
