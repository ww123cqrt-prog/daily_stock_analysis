# -*- coding: utf-8 -*-
"""Backtest endpoints."""

from __future__ import annotations

import json
import logging
from datetime import date
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.deps import get_database_manager
from api.v1.schemas.backtest import (
    BacktestRunRequest,
    BacktestRunResponse,
    BacktestResultItem,
    BacktestResultsResponse,
    FormStrategyBacktestRunRequest,
    PerformanceMetrics,
    ScriptStrategyBacktestRunRequest,
    StabilityTestRequest,
    StabilityTestResponse,
    StrategyBacktestResponse,
    StrategyBacktestRunRequest,
    TrainedStrategyCatalogResponse,
    TrainedStrategyBacktestResponse,
    TrainedStrategyBacktestRunRequest,
)
from api.v1.schemas.common import ErrorResponse
from src.repositories.strategy_assets_repo import get_strategy_assets_repo
from src.services.backtest_service import BacktestService
from src.services.trained_strategy_backtest_service import TrainedStrategyBacktestService
from src.storage import DatabaseManager

logger = logging.getLogger(__name__)

router = APIRouter()

BacktestAnalysisPhaseQuery = Literal["premarket", "intraday", "postmarket", "unknown"]


def _validate_analysis_date_range(
    analysis_date_from: Optional[date],
    analysis_date_to: Optional[date],
) -> None:
    if analysis_date_from and analysis_date_to and analysis_date_from > analysis_date_to:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_params",
                "message": "analysis_date_from cannot be after analysis_date_to",
            },
        )


@router.get(
    "/strategy/trained/catalog",
    response_model=TrainedStrategyCatalogResponse,
    responses={
        200: {"description": "可加载的历史选股模型 Run 与冻结交易策略目录"},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="读取历史选股模型与冻结交易策略目录",
    description="只返回可公开的 Run、配置哈希和数据快照身份，不返回本机绝对路径。",
)
def get_trained_strategy_catalog() -> TrainedStrategyCatalogResponse:
    try:
        return TrainedStrategyCatalogResponse(
            **TrainedStrategyBacktestService().catalog()
        )
    except Exception as exc:
        logger.error("读取历史选股模型回测目录失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "历史选股模型目录读取失败"},
        )


@router.post(
    "/run",
    response_model=BacktestRunResponse,
    responses={
        200: {"description": "回测执行完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="触发回测",
    description="对历史分析记录进行回测评估，并写入 backtest_results/backtest_summaries",
)
def run_backtest(
    request: BacktestRunRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BacktestRunResponse:
    try:
        _validate_analysis_date_range(request.analysis_date_from, request.analysis_date_to)
        service = BacktestService(db_manager)
        stats = service.run_backtest(
            code=request.code,
            force=request.force,
            eval_window_days=request.eval_window_days,
            min_age_days=request.min_age_days,
            analysis_date_from=request.analysis_date_from,
            analysis_date_to=request.analysis_date_to,
            limit=request.limit,
        )
        return BacktestRunResponse(**stats)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"回测执行失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"回测执行失败: {str(exc)}"},
        )


@router.post(
    "/strategy/dsa-signal",
    response_model=StrategyBacktestResponse,
    responses={
        200: {"description": "DSA历史信号交易策略回测完成"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="运行DSA历史信号交易策略回测",
    description=(
        "将DSA历史分析转换为目标仓位，在下一交易日开盘执行，连续计算"
        "资金、持仓、交易成本、净值、回撤和买入持有基准。"
    ),
)
def run_dsa_signal_strategy_backtest(
    request: StrategyBacktestRunRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> StrategyBacktestResponse:
    try:
        _validate_analysis_date_range(request.start_date, request.end_date)
        service = BacktestService(db_manager)
        report = service.run_dsa_signal_strategy(
            code=request.code,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            side_cost_pct=request.side_cost_pct,
            lot_size=request.lot_size,
        )
        return StrategyBacktestResponse(**report)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"DSA交易策略回测失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"DSA交易策略回测失败: {str(exc)}"},
        )


@router.post(
    "/strategy/run",
    response_model=StrategyBacktestResponse,
    responses={
        200: {"description": "条件交易策略回测完成"},
        400: {"description": "请求参数或数据错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="运行条件交易策略回测",
    description="运行明确选择的通用交易策略；规则在收盘后产生目标仓位，下一共同交易日开盘执行。",
)
def run_form_strategy_backtest(
    request: FormStrategyBacktestRunRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> StrategyBacktestResponse:
    try:
        _validate_analysis_date_range(request.start_date, request.end_date)
        report = BacktestService(db_manager).run_strategy(
            strategy_id=request.strategy_id,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            side_cost_pct=request.side_cost_pct,
            lot_size=request.lot_size,
            lookback_days=request.lookback_days,
            moving_average_days=request.moving_average_days,
            fast_ma_days=request.fast_ma_days,
            slow_ma_days=request.slow_ma_days,
            top_n=request.top_n,
        )
        return StrategyBacktestResponse(**report)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("条件交易策略回测失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "条件交易策略回测失败"},
        )


@router.post(
    "/strategy/script",
    response_model=StrategyBacktestResponse,
    responses={
        200: {"description": "脚本交易策略回测完成"},
        400: {"description": "脚本、参数或数据错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="运行DSA Python脚本回测",
    description="静态检查脚本后在受限子进程中逐日生成目标仓位，再由统一回测引擎执行。",
)
def run_script_strategy_backtest(
    request: ScriptStrategyBacktestRunRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> StrategyBacktestResponse:
    try:
        _validate_analysis_date_range(request.start_date, request.end_date)
        report = BacktestService(db_manager).run_script_strategy(
            script=request.script,
            pool=request.pool,
            symbols=request.symbols,
            benchmarks=request.benchmarks,
            comparison_strategy_ids=request.comparison_strategy_ids,
            comparison_presets=[
                {"name": item.name, "script": item.script}
                for item in request.comparison_presets
            ],
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            side_cost_pct=request.side_cost_pct,
            lot_size=request.lot_size,
        )
        try:
            run_id = _persist_script_backtest_run(request, report)
            if run_id:
                report["run_id"] = run_id
        except Exception:
            # Persistence is best-effort: a store failure must never break a
            # completed backtest or its response.
            logger.warning("脚本回测结果落库失败（不影响本次回测）", exc_info=True)
        return StrategyBacktestResponse(**report)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_script_or_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("脚本交易策略回测失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "脚本交易策略回测失败"},
        )


def _persist_script_backtest_run(
    request: ScriptStrategyBacktestRunRequest,
    report: dict,
) -> Optional[str]:
    """Persist a completed script backtest as a durable run record.

    A missing strategy asset id degrades to an ad-hoc run (``strategy_id``
    NULL) so the history still groups by ``script_sha256``.
    """
    repo = get_strategy_assets_repo()
    strategy_id = request.strategy_asset_id
    if strategy_id is not None and repo.get_strategy(strategy_id) is None:
        strategy_id = None
    strategy_meta = report.get("strategy") or {}
    run = repo.create_run(
        strategy_id=strategy_id,
        strategy_name=str(report.get("strategy_name") or "python_script"),
        script_sha256=str(strategy_meta.get("script_sha256") or ""),
        script=request.script or "",
        symbols=list(report.get("symbols") or []),
        params={
            "pool": request.pool,
            "symbols": list(request.symbols or []),
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
            "initial_capital": request.initial_capital,
            "side_cost_pct": request.side_cost_pct,
            "lot_size": request.lot_size,
            # Freeze the comparison experiment so a historical run can be
            # replayed exactly: benchmark selection and compared strategies.
            "benchmarks": list(request.benchmarks or []),
            "comparison_strategy_ids": list(request.comparison_strategy_ids or []),
            "comparison_presets": [
                {"name": item.name, "script": item.script}
                for item in request.comparison_presets
            ],
        },
        data_quality=report.get("data_quality") or {},
        metrics=report.get("metrics") or {},
        report=report,
        status=str(report.get("status") or "completed"),
    )
    return run["id"]


@router.post(
    "/strategy/script/stability",
    response_model=StabilityTestResponse,
    responses={
        200: {"description": "时间均匀性/稳定性测试完成"},
        400: {"description": "脚本、参数或数据错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="运行DSA Python脚本的时间均匀性测试",
    description="跑一次全区间沙箱后，按随机抽样的多个子时间段分别回测，统计收益与回撤的均匀性。",
)
def run_script_stability_test(
    request: StabilityTestRequest,
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> StabilityTestResponse:
    try:
        _validate_analysis_date_range(request.start_date, request.end_date)
        report = BacktestService(db_manager).run_script_stability_test(
            script=request.script,
            pool=request.pool,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            side_cost_pct=request.side_cost_pct,
            lot_size=request.lot_size,
            min_window_days=request.min_window_days,
            iterations=request.iterations,
            seed=request.seed,
        )
        return StabilityTestResponse(**report)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_script_or_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("脚本时间均匀性测试失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "脚本时间均匀性测试失败"},
        )


@router.post(
    "/strategy/trained",
    response_model=TrainedStrategyBacktestResponse,
    responses={
        200: {"description": "已保存选股模型与交易策略重新回测完成"},
        400: {"description": "Run、交易策略、参数或本地选股模型训练产物不可用", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="重新运行已训练选股模型的交易策略回测",
    description="读取指定本地 MLflow Run 的 pred.pkl 和 label.pkl，按页面参数重新计算，不重新训练选股模型。",
)
def run_trained_strategy_backtest(
    request: TrainedStrategyBacktestRunRequest,
) -> TrainedStrategyBacktestResponse:
    try:
        if request.replay_mode == "custom":
            if (
                request.start_date is None
                or request.end_date is None
                or request.topk is None
                or request.cost_bps is None
            ):
                raise ValueError(
                    "自定义回测必须提供开始日期、结束日期、TopK 和成本"
                )
            _validate_analysis_date_range(request.start_date, request.end_date)
        report = TrainedStrategyBacktestService().run(
            run_id=request.run_id,
            model_id=request.model_id,
            strategy_id=request.strategy_id,
            replay_mode=request.replay_mode,
            start_date=(request.start_date.isoformat() if request.start_date else None),
            end_date=(request.end_date.isoformat() if request.end_date else None),
            topk=request.topk,
            cost_bps=(float(request.cost_bps) if request.cost_bps is not None else None),
        )
        return TrainedStrategyBacktestResponse(**report)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "trained_backtest_unavailable", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("选股模型结果的交易策略回测失败: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": "选股模型结果的交易策略回测失败"},
        )


@router.get(
    "/results",
    response_model=BacktestResultsResponse,
    responses={
        200: {"description": "回测结果列表"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取回测结果",
    description="分页获取回测结果，支持按股票代码过滤",
)
def get_backtest_results(
    code: Optional[str] = Query(None, description="股票代码筛选"),
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    analysis_date_from: Optional[date] = Query(None, description="分析日期起始（含）"),
    analysis_date_to: Optional[date] = Query(None, description="分析日期结束（含）"),
    analysis_phase: Optional[BacktestAnalysisPhaseQuery] = Query(None, description="分析阶段过滤：premarket/intraday/postmarket/unknown"),
    page: int = Query(1, ge=1, description="页码"),
    limit: int = Query(20, ge=1, le=200, description="每页数量"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> BacktestResultsResponse:
    try:
        _validate_analysis_date_range(analysis_date_from, analysis_date_to)
        service = BacktestService(db_manager)
        data = service.get_recent_evaluations(
            code=code,
            eval_window_days=eval_window_days,
            limit=limit,
            page=page,
            analysis_date_from=analysis_date_from,
            analysis_date_to=analysis_date_to,
            analysis_phase=analysis_phase,
        )
        items = [BacktestResultItem(**item) for item in data.get("items", [])]
        return BacktestResultsResponse(
            total=int(data.get("total", 0)),
            page=page,
            limit=limit,
            items=items,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询回测结果失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询回测结果失败: {str(exc)}"},
        )


@router.get(
    "/performance",
    response_model=Optional[PerformanceMetrics],
    responses={
        200: {"description": "整体回测表现"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "无回测汇总", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取整体回测表现",
)
def get_overall_performance(
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    analysis_date_from: Optional[date] = Query(None, description="分析日期起始（含）"),
    analysis_date_to: Optional[date] = Query(None, description="分析日期结束（含）"),
    analysis_phase: Optional[BacktestAnalysisPhaseQuery] = Query(None, description="分析阶段过滤：premarket/intraday/postmarket/unknown"),
    empty_ok: bool = Query(False, description="无汇总时返回 null，供网页空状态使用"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Optional[PerformanceMetrics]:
    try:
        _validate_analysis_date_range(analysis_date_from, analysis_date_to)
        service = BacktestService(db_manager)
        summary = service.get_summary(
            scope="overall",
            code=None,
            eval_window_days=eval_window_days,
            analysis_date_from=analysis_date_from,
            analysis_date_to=analysis_date_to,
            analysis_phase=analysis_phase,
        )
        if summary is None:
            if empty_ok:
                return None
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": "未找到整体回测汇总"},
            )
        return PerformanceMetrics(**summary)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询整体表现失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询整体表现失败: {str(exc)}"},
        )


@router.get(
    "/performance/{code}",
    response_model=Optional[PerformanceMetrics],
    responses={
        200: {"description": "单股回测表现"},
        400: {"description": "请求参数错误", "model": ErrorResponse},
        404: {"description": "无回测汇总", "model": ErrorResponse},
        500: {"description": "服务器错误", "model": ErrorResponse},
    },
    summary="获取单股回测表现",
)
def get_stock_performance(
    code: str,
    eval_window_days: Optional[int] = Query(None, ge=1, le=120, description="评估窗口过滤"),
    analysis_date_from: Optional[date] = Query(None, description="分析日期起始（含）"),
    analysis_date_to: Optional[date] = Query(None, description="分析日期结束（含）"),
    analysis_phase: Optional[BacktestAnalysisPhaseQuery] = Query(None, description="分析阶段过滤：premarket/intraday/postmarket/unknown"),
    empty_ok: bool = Query(False, description="无汇总时返回 null，供网页空状态使用"),
    db_manager: DatabaseManager = Depends(get_database_manager),
) -> Optional[PerformanceMetrics]:
    try:
        _validate_analysis_date_range(analysis_date_from, analysis_date_to)
        service = BacktestService(db_manager)
        summary = service.get_summary(
            scope="stock",
            code=code,
            eval_window_days=eval_window_days,
            analysis_date_from=analysis_date_from,
            analysis_date_to=analysis_date_to,
            analysis_phase=analysis_phase,
        )
        if summary is None:
            if empty_ok:
                return None
            raise HTTPException(
                status_code=404,
                detail={"error": "not_found", "message": f"未找到 {code} 的回测汇总"},
            )
        return PerformanceMetrics(**summary)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "invalid_params", "message": str(exc)},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"查询单股表现失败: {exc}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"error": "internal_error", "message": f"查询单股表现失败: {str(exc)}"},
        )
