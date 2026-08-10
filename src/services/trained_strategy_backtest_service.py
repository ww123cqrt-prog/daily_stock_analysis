# -*- coding: utf-8 -*-
"""Rerun an allowlisted MLflow prediction with a selected portfolio strategy."""

from __future__ import annotations

import json
import math
import re
import ast
import hashlib
import copy
import io
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from quant_workflow.scripts._optimized_strategy_core import OptimizedStrategy
from quant_workflow.scripts._paths import init_qlib_workflow
from quant_workflow.scripts._mlflow_run_provenance import (
    RunProvenanceError,
    normalize_provider_lineage,
    provider_lineage_is_complete,
    provider_lineages_match,
    resolve_immutable_provider_identity,
    validate_file_store_run_artifact,
)
from src.services.quant_workflow_paths import (
    mlruns_root as resolve_mlruns_root,
    quant_data_root as resolve_quant_data_root,
    reports_root as resolve_reports_root,
    workflow_root as resolve_workflow_root,
)


RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXPERIMENT_ID_RE = re.compile(r"^[0-9]+$")
MAX_REPORT_BYTES = 10 * 1024 * 1024
MAX_PICKLE_BYTES = 512 * 1024 * 1024
ALLOWED_STRATEGY_KWARGS = {
    "n_drop",
    "rebalance_freq",
    "turnover_cap",
    "trend_filter",
    "trend_ma",
    "trend_reduce_pct",
    "vol_sizing",
    "vol_reduce_threshold",
    "max_drawdown_stop",
    "max_dd_cooldown_days",
    "max_single_weight",
    "max_sector_weight",
}

MarketLoader = Callable[[Path, str, str], pd.DataFrame]
ProviderResolver = Callable[[dict[str, Any], Path], Path]
ArtifactValidator = Callable[..., dict[str, Any]]
QlibBacktestRunner = Callable[[Path, dict[str, Any], pd.DataFrame], pd.DataFrame]
LEGACY_BASELINE_STRATEGY = {
    "id": "baseline_daily",
    "label": "每日 TopK 等权基线（旧版）",
    "config": {
        "rebalance_freq": 1,
        "turnover_cap": 1.0,
        "max_drawdown_stop": 0.0,
    },
}
QLIB_RISK_ANNUALIZATION = 238.0
EXACT_METRIC_TOLERANCE = 1e-10
SUPPORTED_REPLAY_ENGINES = {"qlib_portana", "dsa_optimized_strategy"}
QLIB_EXACT_ARTIFACTS = ("pred.pkl", "label.pkl", "task")
CUSTOM_REPLAY_ARTIFACTS = ("pred.pkl", "label.pkl")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required training report is missing: {path.name}")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise ValueError(f"Training report is too large: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Training report must contain an object: {path.name}")
    return payload


def _single_column(frame: pd.DataFrame | pd.Series, name: str) -> pd.DataFrame:
    result = frame.to_frame(name) if isinstance(frame, pd.Series) else frame.copy()
    if result.shape[1] != 1:
        raise ValueError(f"Saved {name} artifact must contain exactly one column")
    result.columns = [name]
    return result


def _slice_frame(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.MultiIndex) or "datetime" not in frame.index.names:
        raise ValueError("Saved prediction artifacts must use a datetime/instrument MultiIndex")
    dates = pd.to_datetime(frame.index.get_level_values("datetime"), errors="coerce")
    if dates.isna().any():
        raise ValueError("Saved prediction artifacts contain invalid dates")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return frame.loc[(dates >= start_ts) & (dates <= end_ts)]


def _default_market_loader(provider_uri: Path, start: str, end: str) -> pd.DataFrame:
    if not provider_uri.is_dir():
        raise FileNotFoundError("The Qlib data provider used by this training run is unavailable")

    from qlib.data import D

    init_qlib_workflow(provider_uri=provider_uri)
    buffered_end = (pd.Timestamp(end) + pd.Timedelta(days=14)).date().isoformat()
    raw = D.features(["SH000300"], ["$close"], start_time=start, end_time=buffered_end)
    if raw.empty:
        raise ValueError("SH000300 benchmark data is unavailable for the selected period")
    market = raw.droplevel("instrument").rename(columns={"$close": "close"}).sort_index()
    market.index = pd.to_datetime(market.index)
    close = pd.to_numeric(market["close"], errors="coerce")
    market["benchmark_return"] = close.shift(-2) / close.shift(-1) - 1.0
    for days in (20, 60, 120, 200):
        market[f"ma{days}"] = close.rolling(days).mean()
        market[f"trend_{days}"] = (close > market[f"ma{days}"]).astype(float)
    historical_return = close.pct_change(fill_method=None)
    market["volatility_20"] = historical_return.rolling(20).std() * math.sqrt(252)
    market["vol_percentile"] = market["volatility_20"].rolling(120).rank(pct=True)
    return market


def _default_qlib_backtest_runner(
    provider_uri: Path,
    portana_config: dict[str, Any],
    pred: pd.DataFrame,
) -> pd.DataFrame:
    """Run the frozen Qlib portfolio configuration against saved predictions."""

    from qlib.backtest import backtest as qlib_backtest

    init_qlib_workflow(provider_uri=provider_uri)
    config = copy.deepcopy(portana_config)
    strategy = config["strategy"]
    strategy["kwargs"]["signal"] = pred
    executor = config.get("executor") or {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    }
    portfolio_metrics, _ = qlib_backtest(
        strategy=strategy,
        executor=executor,
        **config["backtest"],
    )
    if "1day" not in portfolio_metrics:
        raise ValueError("Qlib exact replay did not produce a daily portfolio report")
    report, _positions = portfolio_metrics["1day"]
    if not isinstance(report, pd.DataFrame):
        raise ValueError("Qlib exact replay returned an invalid portfolio report")
    return report.copy()


def _annualized(series: pd.Series) -> float:
    total = float((1.0 + series).prod())
    return total ** (252.0 / len(series)) - 1.0 if total > 0 else -1.0


def _finite_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _qlib_excess_metrics(excess: pd.Series) -> dict[str, float | None]:
    cumulative = excess.cumsum()
    std = float(excess.std(ddof=1))
    return {
        "annual_excess_return": float(excess.mean()) * QLIB_RISK_ANNUALIZATION,
        "information_ratio": _finite_or_none(
            float(excess.mean()) / std * math.sqrt(QLIB_RISK_ANNUALIZATION)
            if std > 0
            else None
        ),
        "excess_max_drawdown": float((cumulative - cumulative.cummax()).min()),
    }


def _contract_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "config_hash"}
    return hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sanitized_provider_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    try:
        return normalize_provider_lineage(value)
    except RunProvenanceError:
        return {}


def _safe_periods(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[str]] = {}
    for name, dates in value.items():
        if not isinstance(name, str) or not isinstance(dates, (list, tuple)):
            continue
        cleaned = [str(item) for item in dates if isinstance(item, str)]
        if cleaned:
            result[name] = cleaned
    return result


class TrainedStrategyBacktestService:
    """Load only locally registered training artifacts and rerun their strategy."""

    def __init__(
        self,
        *,
        reports_root: Path | None = None,
        mlruns_root: Path | None = None,
        quant_data_root: Path | None = None,
        strategy_config_path: Path | None = None,
        market_loader: MarketLoader | None = None,
        qlib_backtest_runner: QlibBacktestRunner | None = None,
        provider_resolver: ProviderResolver | None = None,
        artifact_validator: ArtifactValidator | None = None,
    ) -> None:
        self.reports_root = Path(reports_root or resolve_reports_root()).resolve()
        self.mlruns_root = Path(mlruns_root or resolve_mlruns_root()).resolve()
        self.quant_data_root = Path(
            quant_data_root or resolve_quant_data_root()
        ).resolve()
        self.strategy_config_path = Path(
            strategy_config_path
            or (resolve_workflow_root() / "configs" / "candidate_pipeline.yaml")
        ).resolve()
        self.market_loader = market_loader or _default_market_loader
        self.qlib_backtest_runner = (
            qlib_backtest_runner or _default_qlib_backtest_runner
        )
        self.provider_resolver = provider_resolver or (
            lambda identity, root: resolve_immutable_provider_identity(
                identity,
                quant_data_root=root,
            )
        )
        self.artifact_validator = (
            artifact_validator or validate_file_store_run_artifact
        )

    def catalog(self) -> dict[str, Any]:
        """Return historical replayable Runs without exposing local paths."""

        rows: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        baseline_paths = sorted(
            (self.reports_root / "training_runs").glob("*.json"),
            reverse=True,
        )[:500]
        latest_baseline = self.reports_root / "latest_training_result.json"
        if latest_baseline.is_file():
            baseline_paths.append(latest_baseline)
        for path in baseline_paths:
            try:
                report = _load_json(path)
                row = self._baseline_catalog_row(report)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                continue
            if row is None:
                continue
            key = (row["run_id"], row["model_id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)

        candidate_paths = sorted(
            self.reports_root.glob("candidate_pipeline_[0-9]*.json"),
            reverse=True,
        )[:500]
        latest_candidate = self.reports_root / "candidate_pipeline_latest.json"
        if latest_candidate.is_file():
            candidate_paths.append(latest_candidate)
        for path in candidate_paths:
            try:
                report = _load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                continue
            if report.get("dry_run") is True:
                continue
            for row in self._candidate_catalog_rows(report):
                key = (row["run_id"], row["model_id"])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)

        rows.sort(
            key=lambda item: (str(item.get("generated_at") or ""), item["run_id"]),
            reverse=True,
        )
        return {
            "status": "completed" if rows else "missing",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "runs": rows,
        }

    def run(
        self,
        *,
        run_id: str,
        model_id: str,
        strategy_id: str,
        start_date: str | None,
        end_date: str | None,
        topk: int | None,
        cost_bps: float | None,
        replay_mode: str = "custom",
    ) -> dict[str, Any]:
        if not RUN_ID_RE.fullmatch(run_id):
            raise ValueError("Run ID must contain exactly 32 lowercase hexadecimal characters")
        if replay_mode not in {"exact", "custom"}:
            raise ValueError("replay_mode must be exact or custom")

        selection = self._resolve_selection(run_id, model_id, strategy_id)
        original_parameters = selection.get("original_parameters") or {}
        if replay_mode == "exact":
            if not selection.get("exact_replay_available"):
                raise ValueError(
                    "The selected strategy does not contain a valid exact replay contract"
                )
            start_date = str(original_parameters["start_date"])
            end_date = str(original_parameters["end_date"])
            topk = int(original_parameters["topk"])
            cost_bps = float(original_parameters["cost_bps"])

        if start_date is None or end_date is None or topk is None or cost_bps is None:
            raise ValueError(
                "Custom replay requires start_date, end_date, topk and cost_bps"
            )

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        if start_ts > end_ts:
            raise ValueError("start_date cannot be after end_date")
        if not 1 <= int(topk) <= 500:
            raise ValueError("topk must be between 1 and 500")
        if not 0 <= float(cost_bps) <= 1000:
            raise ValueError("cost_bps must be between 0 and 1000")

        source_replay_engine = str(
            selection.get("replay_engine") or "dsa_optimized_strategy"
        )
        use_qlib_exact_replay = (
            replay_mode == "exact" and source_replay_engine == "qlib_portana"
        )
        replay_engine = (
            "qlib_portana" if use_qlib_exact_replay else "dsa_optimized_strategy"
        )
        report_artifact_sha256 = selection.get("artifact_sha256") or {}
        required_artifacts = list(CUSTOM_REPLAY_ARTIFACTS)
        if use_qlib_exact_replay:
            required_artifacts.append("task")
        expected_artifact_sha256 = {
            name: self._required_artifact_sha256(report_artifact_sha256, name)
            for name in required_artifacts
        }
        pred, pred_provenance = self._read_artifact(
            selection["experiment_id"],
            run_id,
            "pred.pkl",
            expected_provider_identity=selection["provider_identity"],
            expected_provider_path=selection["provider_uri"],
            expected_artifact_sha256=expected_artifact_sha256["pred.pkl"],
        )
        label, label_provenance = self._read_artifact(
            selection["experiment_id"],
            run_id,
            "label.pkl",
            expected_provider_identity=selection["provider_identity"],
            expected_provider_path=selection["provider_uri"],
            expected_artifact_sha256=expected_artifact_sha256["label.pkl"],
        )
        pred_slice = _single_column(_slice_frame(pred, start_date, end_date), "score")
        label_slice = _single_column(_slice_frame(label, start_date, end_date), "label")
        if pred_slice.empty or label_slice.empty:
            raise ValueError("The selected run has no prediction and label data in this date range")

        artifact_sha256 = {
            "pred.pkl": pred_provenance["artifact_sha256"],
            "label.pkl": label_provenance["artifact_sha256"],
        }
        if use_qlib_exact_replay:
            task, task_provenance = self._read_pickle_artifact(
                selection["experiment_id"],
                run_id,
                "task",
                expected_provider_identity=selection["provider_identity"],
                expected_provider_path=selection["provider_uri"],
                expected_artifact_sha256=expected_artifact_sha256["task"],
            )
            portana_config = self._validated_qlib_portana_config(
                task,
                selection["contract"],
            )
            qlib_report = self.qlib_backtest_runner(
                selection["provider_uri"],
                portana_config,
                pred,
            )
            backtest, benchmark = self._normalize_qlib_report(qlib_report)
            artifact_sha256["task"] = task_provenance["artifact_sha256"]
        else:
            market = self.market_loader(
                selection["provider_uri"], start_date, end_date
            ).copy()
            market.index = pd.to_datetime(market.index)
            if "benchmark_return" not in market.columns:
                raise ValueError("Market data is missing benchmark_return")

            strategy = OptimizedStrategy(
                topk=int(topk),
                cost_bps=float(cost_bps),
                **selection["strategy_kwargs"],
            )
            backtest = strategy.run(pred_slice, label_slice, market)
            benchmark = pd.to_numeric(
                market["benchmark_return"], errors="coerce"
            ).reindex(backtest.index)

        if backtest.empty:
            raise ValueError("The selected TopK is larger than the available stocks in this period")

        returns = pd.to_numeric(backtest["return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if returns.isna().any():
            raise ValueError("The recalculated strategy contains invalid daily returns")
        benchmark = pd.to_numeric(benchmark, errors="coerce").reindex(returns.index)
        if benchmark.isna().any():
            missing = int(benchmark.isna().sum())
            raise ValueError(f"SH000300 benchmark is missing {missing} selected trading days")

        strategy_nav = (1.0 + returns).cumprod()
        benchmark_nav = (1.0 + benchmark).cumprod()
        strategy_drawdown = strategy_nav / strategy_nav.cummax() - 1.0
        benchmark_drawdown = benchmark_nav / benchmark_nav.cummax() - 1.0
        excess = returns - benchmark
        return_std = float(returns.std(ddof=1))
        excess_std = float(excess.std(ddof=1))
        annual_return = _annualized(returns)
        benchmark_annual_return = _annualized(benchmark)
        qlib_excess_metrics = (
            _qlib_excess_metrics(excess)
            if use_qlib_exact_replay
            else None
        )
        exact_metric_validation = (
            self._validate_exact_qlib_metrics(
                selection.get("saved_metrics"), qlib_excess_metrics
            )
            if qlib_excess_metrics is not None
            else None
        )

        equity_curve = []
        for day in returns.index:
            equity_curve.append(
                {
                    "date": pd.Timestamp(day).date().isoformat(),
                    "strategy_cumulative_return": float(strategy_nav.loc[day] - 1.0),
                    "benchmark_cumulative_return": float(benchmark_nav.loc[day] - 1.0),
                    "drawdown": float(strategy_drawdown.loc[day]),
                    "benchmark_drawdown": float(benchmark_drawdown.loc[day]),
                    "daily_return": float(returns.loc[day]),
                    "benchmark_return": float(benchmark.loc[day]),
                    "turnover": float(backtest.loc[day, "turnover"]),
                    "position_pct": float(backtest.loc[day, "position_pct"]),
                }
            )

        warnings: list[str] = []
        actual_start = equity_curve[0]["date"]
        actual_end = equity_curve[-1]["date"]
        if actual_start != start_date or actual_end != end_date:
            warnings.append(
                f"The saved run contains evaluable trading days from {actual_start} to {actual_end}."
            )

        return {
            "status": "completed",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "run_id": run_id,
            "model_id": model_id,
            "model_label": selection["model_label"],
            "strategy_id": selection["strategy_id"],
            "strategy_label": selection["strategy_label"],
            "strategy_config": selection["strategy_kwargs"],
            "replay_mode": replay_mode,
            "replay_engine": replay_engine,
            "source_replay_engine": source_replay_engine,
            "modified_from_original": replay_mode == "custom",
            "config_hash": selection.get("config_hash"),
            "contract_version": selection.get("contract_version"),
            "original_parameters": original_parameters,
            "provider": selection["provider"],
            "provider_identity": _sanitized_provider_identity(
                selection["provider_identity"]
            ),
            "artifact_sha256": artifact_sha256,
            "execution_assumptions": selection.get("execution_assumptions") or {},
            "exact_metric_validation": exact_metric_validation,
            "period": {"start_date": actual_start, "end_date": actual_end},
            "parameters": {
                "start_date": start_date,
                "end_date": end_date,
                "topk": int(topk),
                "cost_bps": float(cost_bps),
            },
            "metrics": {
                "days": int(len(returns)),
                "total_return": float(strategy_nav.iloc[-1] - 1.0),
                "benchmark_total_return": float(benchmark_nav.iloc[-1] - 1.0),
                "annual_return": annual_return,
                "benchmark_annual_return": benchmark_annual_return,
                "annual_excess_return": (
                    qlib_excess_metrics["annual_excess_return"]
                    if qlib_excess_metrics is not None
                    else annual_return - benchmark_annual_return
                ),
                "information_ratio": (
                    qlib_excess_metrics["information_ratio"]
                    if qlib_excess_metrics is not None
                    else _finite_or_none(
                        float(excess.mean()) / excess_std * math.sqrt(252)
                        if excess_std > 0
                        else None
                    )
                ),
                "sharpe": _finite_or_none(
                    float(returns.mean()) / return_std * math.sqrt(252) if return_std > 0 else None
                ),
                "max_drawdown": float(strategy_drawdown.min()),
                "avg_turnover": float(backtest["turnover"].mean()),
                "avg_position": float(backtest["position_pct"].mean()),
            },
            "equity_curve": equity_curve,
            "warnings": warnings,
        }

    @staticmethod
    def _normalize_qlib_report(
        raw: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        required = {"return", "cost", "bench", "turnover", "account", "value"}
        if not isinstance(raw, pd.DataFrame) or not required.issubset(raw.columns):
            missing = sorted(required - set(getattr(raw, "columns", [])))
            raise ValueError(
                "Qlib exact replay report is missing required columns: "
                + ", ".join(missing)
            )
        report = raw.copy()
        report.index = pd.to_datetime(report.index, errors="coerce")
        if report.index.isna().any() or report.index.has_duplicates:
            raise ValueError("Qlib exact replay report contains invalid trading dates")
        report = report.sort_index()
        numeric: dict[str, pd.Series] = {}
        for column in required:
            series = pd.to_numeric(report[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            if series.isna().any():
                raise ValueError(
                    f"Qlib exact replay report contains invalid {column} values"
                )
            numeric[column] = series.astype(float)
        if (numeric["account"] <= 0).any():
            raise ValueError("Qlib exact replay report contains a non-positive account")

        backtest = pd.DataFrame(index=report.index)
        backtest["return"] = numeric["return"] - numeric["cost"]
        backtest["gross_return"] = numeric["return"]
        backtest["turnover"] = numeric["turnover"]
        backtest["position_pct"] = (
            numeric["value"] / numeric["account"]
        ).clip(lower=0.0, upper=1.0)
        backtest["n_stocks"] = np.nan
        return backtest, numeric["bench"]

    @classmethod
    def _validated_qlib_portana_config(
        cls,
        task: Any,
        contract: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(task, dict) or not isinstance(task.get("record"), list):
            raise ValueError("Saved Qlib task does not contain a record list")
        records = [
            item
            for item in task["record"]
            if isinstance(item, dict) and item.get("class") == "PortAnaRecord"
        ]
        if len(records) != 1:
            raise ValueError("Saved Qlib task must contain exactly one PortAnaRecord")
        record = records[0]
        if record.get("module_path") not in {None, "qlib.workflow.record_temp"}:
            raise ValueError("Saved Qlib task contains an unsupported record module")
        kwargs = record.get("kwargs") if isinstance(record.get("kwargs"), dict) else {}
        config = kwargs.get("config") if isinstance(kwargs.get("config"), dict) else None
        if not isinstance(config, dict):
            raise ValueError("Saved Qlib task does not contain a PortAnaRecord config")
        config = copy.deepcopy(config)

        strategy = config.get("strategy")
        if (
            not isinstance(strategy, dict)
            or strategy.get("class") != "TopkDropoutStrategy"
            or strategy.get("module_path") not in {None, "qlib.contrib.strategy"}
            or not isinstance(strategy.get("kwargs"), dict)
        ):
            raise ValueError("Saved Qlib task contains an unsupported trading strategy")
        strategy_kwargs = strategy["kwargs"]
        allowed_strategy_kwargs = {
            "signal",
            "topk",
            "n_drop",
            "method_sell",
            "method_buy",
            "hold_thresh",
            "only_tradable",
            "forbid_all_trade_at_limit",
        }
        if set(strategy_kwargs) - allowed_strategy_kwargs:
            raise ValueError("Saved Qlib strategy contains unsupported parameters")
        if strategy_kwargs.get("signal") != "<PRED>":
            raise ValueError("Saved Qlib strategy does not use the prediction placeholder")
        try:
            topk = int(strategy_kwargs["topk"])
            n_drop = int(strategy_kwargs["n_drop"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Saved Qlib strategy has invalid TopK parameters") from exc
        if not 1 <= topk <= 500 or not 0 <= n_drop <= topk:
            raise ValueError("Saved Qlib strategy has out-of-range TopK parameters")

        executor = config.get("executor")
        if executor is not None:
            if (
                not isinstance(executor, dict)
                or executor.get("class") != "SimulatorExecutor"
                or executor.get("module_path") not in {None, "qlib.backtest.executor"}
                or not isinstance(executor.get("kwargs"), dict)
                or set(executor["kwargs"])
                - {"time_per_step", "generate_portfolio_metrics", "track_data"}
            ):
                raise ValueError("Saved Qlib task contains an unsupported executor")

        backtest = config.get("backtest")
        if not isinstance(backtest, dict):
            raise ValueError("Saved Qlib task does not contain backtest parameters")
        if set(backtest) - {
            "start_time",
            "end_time",
            "account",
            "benchmark",
            "exchange_kwargs",
            "pos_type",
        }:
            raise ValueError("Saved Qlib backtest contains unsupported parameters")
        exchange = backtest.get("exchange_kwargs")
        if not isinstance(exchange, dict) or set(exchange) - {
            "limit_threshold",
            "deal_price",
            "open_cost",
            "close_cost",
            "min_cost",
            "trade_unit",
        }:
            raise ValueError("Saved Qlib backtest contains unsupported exchange parameters")

        if not isinstance(contract, dict) or contract.get("replay_engine") != "qlib_portana":
            raise ValueError("The Run does not contain a Qlib exact replay contract")
        original = contract.get("original_parameters")
        assumptions = contract.get("qlib_backtest_assumptions")
        if not isinstance(original, dict) or not isinstance(assumptions, dict):
            raise ValueError("The Qlib exact replay contract is incomplete")
        try:
            actual = {
                "account": float(backtest["account"]),
                "benchmark": str(backtest["benchmark"]),
                "topk": topk,
                "n_drop": n_drop,
                "deal_price": str(exchange["deal_price"]),
                "limit_threshold": float(exchange["limit_threshold"]),
                "open_cost": float(exchange["open_cost"]),
                "close_cost": float(exchange["close_cost"]),
                "min_cost": float(exchange["min_cost"]),
            }
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Saved Qlib backtest parameters are incomplete") from exc
        for name, value in actual.items():
            expected = assumptions.get(name)
            matches = (
                math.isclose(float(expected), float(value), rel_tol=0.0, abs_tol=1e-12)
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else expected == value
            )
            if not matches:
                raise ValueError(
                    f"Saved Qlib task does not match the frozen {name} assumption"
                )
        expected_cost_bps = (actual["open_cost"] + actual["close_cost"]) * 10_000
        if (
            str(backtest.get("start_time") or "") != str(original.get("start_date") or "")
            or str(backtest.get("end_time") or "") != str(original.get("end_date") or "")
            or int(original.get("topk") or 0) != topk
            or not math.isclose(
                float(original.get("cost_bps") or 0.0),
                expected_cost_bps,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("Saved Qlib task does not match the frozen original parameters")
        return config

    @staticmethod
    def _validate_exact_qlib_metrics(
        saved_metrics: Any,
        replayed_metrics: dict[str, float | None],
    ) -> dict[str, Any]:
        if not isinstance(saved_metrics, dict):
            raise ValueError("The Run does not contain saved Qlib metrics for validation")
        mapping = {
            "annual_excess_return": "1day.excess_return_with_cost.annualized_return",
            "information_ratio": "1day.excess_return_with_cost.information_ratio",
            "excess_max_drawdown": "1day.excess_return_with_cost.max_drawdown",
        }
        saved: dict[str, float] = {}
        replayed: dict[str, float] = {}
        differences: dict[str, float] = {}
        for public_name, saved_name in mapping.items():
            expected = _finite_or_none(saved_metrics.get(saved_name))
            actual = _finite_or_none(replayed_metrics.get(public_name))
            if expected is None or actual is None:
                raise ValueError("The Run contains incomplete Qlib metrics for validation")
            saved[public_name] = expected
            replayed[public_name] = actual
            differences[public_name] = abs(actual - expected)
        max_difference = max(differences.values(), default=0.0)
        if max_difference > EXACT_METRIC_TOLERANCE:
            raise ValueError("Exact replay does not match the saved Qlib metrics")
        return {
            "status": "pass",
            "tolerance": EXACT_METRIC_TOLERANCE,
            "max_abs_difference": max_difference,
            "saved": saved,
            "replayed": replayed,
        }

    def _baseline_catalog_row(self, report: dict[str, Any]) -> dict[str, Any] | None:
        if report.get("result_scope") == "candidate_child":
            return None
        run_id = str(report.get("run_id") or "")
        experiment_id = str(report.get("experiment_id") or "")
        if (
            report.get("status") != "completed"
            or not RUN_ID_RE.fullmatch(run_id)
            or not EXPERIMENT_ID_RE.fullmatch(experiment_id)
        ):
            return None
        contract = self._validated_replay_contract(report.get("replay_contract"))
        strategies = (
            list(contract["strategies"])
            if contract is not None
            else [self._legacy_baseline_strategy(report)]
        )
        return self._catalog_row(
            report=report,
            run_id=run_id,
            model_id="latest",
            model_label=str(report.get("experiment_name") or "LightGBM baseline"),
            source_type="baseline",
            contract=contract,
            strategies=strategies,
            artifact_sha256=report.get("artifact_sha256"),
            original_parameters=(
                dict(contract["original_parameters"])
                if contract is not None
                else self._legacy_baseline_parameters(report)
            ),
        )

    def _candidate_catalog_rows(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        models = report.get("models") if isinstance(report.get("models"), list) else []
        evaluations = (
            report.get("strategy_evaluations")
            if isinstance(report.get("strategy_evaluations"), list)
            else []
        )
        contract = self._validated_replay_contract(report.get("backtest_contract"))
        contract_strategies = {
            item["id"]: item for item in contract["strategies"]
        } if contract is not None else {}
        rows: list[dict[str, Any]] = []
        for model in models:
            if not isinstance(model, dict) or model.get("status") != "completed":
                continue
            run_id = str(model.get("run_id") or "")
            experiment_id = str(model.get("experiment_id") or "")
            if not RUN_ID_RE.fullmatch(run_id) or not EXPERIMENT_ID_RE.fullmatch(experiment_id):
                continue
            strategies: list[dict[str, Any]] = []
            for evaluation in evaluations:
                if not isinstance(evaluation, dict) or evaluation.get("model_id") != model.get("id"):
                    continue
                strategy_id = str(evaluation.get("strategy_id") or "")
                frozen = contract_strategies.get(strategy_id)
                if frozen is None:
                    try:
                        config = self._validated_strategy_kwargs(
                            evaluation.get("strategy_config")
                        )
                    except ValueError:
                        continue
                    frozen = {
                        "id": strategy_id,
                        "label": str(evaluation.get("strategy_label") or strategy_id),
                        "config": config,
                    }
                strategies.append(dict(frozen))
            original = (
                dict(contract["original_parameters"])
                if contract is not None
                else self._legacy_candidate_parameters(report)
            )
            rows.append(
                self._catalog_row(
                    report=report,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    model_id=str(model.get("id") or ""),
                    model_label=str(model.get("label") or model.get("id") or "Model"),
                    source_type="candidate",
                    contract=contract,
                    strategies=strategies,
                    artifact_sha256=model.get("artifact_sha256"),
                    original_parameters=original,
                )
            )
        return rows

    def _catalog_row(
        self,
        *,
        report: dict[str, Any],
        run_id: str,
        experiment_id: str | None = None,
        model_id: str,
        model_label: str,
        source_type: str,
        contract: dict[str, Any] | None,
        strategies: list[dict[str, Any]],
        artifact_sha256: Any,
        original_parameters: dict[str, Any],
    ) -> dict[str, Any]:
        identity = _sanitized_provider_identity(report.get("provider_identity"))
        provider_complete = provider_lineage_is_complete(identity)
        artifact_hashes_complete = self._artifact_hashes_complete(
            artifact_sha256,
            CUSTOM_REPLAY_ARTIFACTS,
        )
        custom_available = provider_complete and bool(strategies) and artifact_hashes_complete
        exact_strategy_ids = (
            set()
            if source_type == "candidate"
            else self._exact_strategy_ids_for_report(contract, report)
        )
        catalog_strategies = [
            {
                **strategy,
                "exact_replay_available": strategy.get("id") in exact_strategy_ids,
                "replay_engine": (
                    contract.get("replay_engine")
                    if contract is not None
                    and strategy.get("id") in exact_strategy_ids
                    else "dsa_optimized_strategy"
                ),
            }
            for strategy in strategies
        ]
        exact_available = custom_available and bool(exact_strategy_ids)
        if exact_available:
            capability = "exact_and_custom"
            reason = None
        elif custom_available:
            capability = "custom_only"
            if (
                source_type == "candidate"
                or (
                    contract is not None
                    and contract.get("replay_capability") == "custom_only"
                )
            ):
                reason = (
                    "候选 DSA 交易策略使用当前引擎重建回测；未冻结历史引擎版本，"
                    "不能按原条件重放。"
                )
            elif (
                contract is not None
                and contract.get("replay_engine") == "qlib_portana"
                and not self._qlib_exact_provenance_complete(report)
            ):
                reason = (
                    "该 Run 缺少 Qlib 原样回放所需的 pred.pkl、label.pkl 或 task 哈希，"
                    "只能自定义回测。"
                )
            else:
                reason = "旧版 Run 没有可验证的精确重放契约，只能明确标记后自定义回测。"
        else:
            capability = "unavailable"
            if provider_complete and strategies and not artifact_hashes_complete:
                reason = "该 Run 缺少冻结的 pred.pkl 或 label.pkl 哈希，不能回测。"
            else:
                reason = (
                    "该 Run 缺少不可变数据身份或冻结交易策略配置，不能回测。"
                )
        return {
            "run_id": run_id,
            "experiment_id": experiment_id
            or str(report.get("experiment_id") or "")
            or None,
            "model_id": model_id,
            "model_label": model_label,
            "source_type": source_type,
            "generated_at": str(report.get("generated_at") or ""),
            "provider": str(report.get("provider") or "unknown"),
            "provider_identity": identity,
            "periods": (
                _safe_periods(contract.get("periods"))
                if contract is not None
                else _safe_periods(report.get("periods"))
            ),
            "original_parameters": self._sanitized_original_parameters(
                original_parameters
            ),
            "strategies": catalog_strategies,
            "config_hash": contract.get("config_hash") if contract else None,
            "contract_version": contract.get("contract_version") if contract else None,
            "exact_replay_available": exact_available,
            "custom_replay_available": custom_available,
            "replay_capability": capability,
            "replay_unavailable_reason": reason,
        }

    @classmethod
    def _validated_replay_contract(cls, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict) or value.get("contract_version") not in {1, 2}:
            return None
        config_hash = str(value.get("config_hash") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", config_hash):
            return None
        if _contract_hash(value) != config_hash:
            return None
        parameters = cls._sanitized_original_parameters(value.get("original_parameters"))
        if not parameters.get("start_date") or not parameters.get("end_date"):
            return None
        strategies: list[dict[str, Any]] = []
        raw_strategies = value.get("strategies")
        if not isinstance(raw_strategies, list):
            return None
        for item in raw_strategies:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                return None
            try:
                config = cls._validated_strategy_kwargs(item.get("config"))
            except ValueError:
                return None
            strategies.append(
                {
                    "id": item["id"],
                    "label": str(item.get("label") or item["id"]),
                    "config": config,
                }
            )
        if not strategies:
            return None
        replay_engine = value.get("replay_engine")
        replay_capability = value.get("replay_capability")
        exact_strategy_ids = value.get("exact_strategy_ids")
        if value.get("contract_version") == 2:
            if replay_engine not in SUPPORTED_REPLAY_ENGINES:
                return None
            if (
                not isinstance(exact_strategy_ids, list)
                or any(not isinstance(item, str) for item in exact_strategy_ids)
                or not set(exact_strategy_ids).issubset(
                    {item["id"] for item in strategies}
                )
            ):
                return None
            if replay_capability == "custom_only":
                if exact_strategy_ids:
                    return None
            elif replay_capability in {None, "exact_and_custom"}:
                if not exact_strategy_ids:
                    return None
            else:
                return None
        else:
            replay_engine = None
            replay_capability = None
            exact_strategy_ids = []
        return {
            **value,
            "periods": _safe_periods(value.get("periods")),
            "original_parameters": parameters,
            "strategies": strategies,
            "replay_engine": replay_engine,
            "replay_capability": replay_capability,
            "exact_strategy_ids": exact_strategy_ids,
        }

    @staticmethod
    def _artifact_hashes_complete(value: Any, names: tuple[str, ...]) -> bool:
        return isinstance(value, dict) and all(
            re.fullmatch(r"[0-9a-f]{64}", str(value.get(name) or ""))
            for name in names
        )

    @classmethod
    def _required_artifact_sha256(cls, value: Any, name: str) -> str:
        if not cls._artifact_hashes_complete(value, (name,)):
            raise ValueError(
                f"The selected training result does not contain a frozen artifact hash: {name}"
            )
        return str(value[name])

    @staticmethod
    def _exact_strategy_ids(contract: dict[str, Any] | None) -> set[str]:
        if not isinstance(contract, dict) or contract.get("contract_version") != 2:
            return set()
        values = contract.get("exact_strategy_ids")
        return {item for item in values if isinstance(item, str)} if isinstance(values, list) else set()

    @staticmethod
    def _qlib_exact_provenance_complete(report: dict[str, Any]) -> bool:
        hashes = report.get("artifact_sha256")
        return isinstance(hashes, dict) and all(
            re.fullmatch(r"[0-9a-f]{64}", str(hashes.get(name) or ""))
            for name in QLIB_EXACT_ARTIFACTS
        )

    @classmethod
    def _exact_strategy_ids_for_report(
        cls,
        contract: dict[str, Any] | None,
        report: dict[str, Any],
    ) -> set[str]:
        exact_strategy_ids = cls._exact_strategy_ids(contract)
        if (
            isinstance(contract, dict)
            and contract.get("replay_engine") == "qlib_portana"
            and not cls._qlib_exact_provenance_complete(report)
        ):
            return set()
        return exact_strategy_ids

    @staticmethod
    def _sanitized_original_parameters(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        try:
            topk = int(value.get("topk"))
            cost_bps = float(value.get("cost_bps"))
            start_date = str(value.get("start_date") or "")
            end_date = str(value.get("end_date") or "")
            if not 1 <= topk <= 500 or not 0 <= cost_bps <= 1000:
                return {}
            if start_date and end_date and pd.Timestamp(start_date) > pd.Timestamp(end_date):
                return {}
        except (TypeError, ValueError, OverflowError):
            return {}
        result: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "topk": topk,
            "cost_bps": cost_bps,
        }
        if isinstance(value.get("strategy_id"), str):
            result["strategy_id"] = value["strategy_id"]
        return result

    @staticmethod
    def _legacy_baseline_parameters(report: dict[str, Any]) -> dict[str, Any]:
        params = report.get("params") if isinstance(report.get("params"), dict) else {}
        start_date = ""
        end_date = ""
        topk = 50
        cost_bps = 15.0
        try:
            test_period = ast.literal_eval(
                str(params.get("dataset.kwargs.segments.test") or "[]")
            )
            if isinstance(test_period, (list, tuple)) and len(test_period) == 2:
                start_date, end_date = str(test_period[0]), str(test_period[1])
            records = ast.literal_eval(str(params.get("record") or "[]"))
            portana = next(
                item
                for item in records
                if isinstance(item, dict) and item.get("class") == "PortAnaRecord"
            )
            config = portana.get("kwargs", {}).get("config", {})
            strategy_kwargs = config.get("strategy", {}).get("kwargs", {})
            backtest = config.get("backtest", {})
            topk = int(strategy_kwargs.get("topk") or topk)
            start_date = str(backtest.get("start_time") or start_date)
            end_date = str(backtest.get("end_time") or end_date)
            exchange = backtest.get("exchange_kwargs", {})
            cost_bps = float(exchange.get("open_cost", 0.0) + exchange.get("close_cost", 0.0)) * 10000
        except (ValueError, SyntaxError, StopIteration, AttributeError, TypeError):
            pass
        return {
            "strategy_id": "baseline_daily",
            "start_date": start_date,
            "end_date": end_date,
            "topk": topk,
            "cost_bps": cost_bps,
        }

    @staticmethod
    def _legacy_baseline_strategy(report: dict[str, Any]) -> dict[str, Any]:
        strategy = json.loads(json.dumps(LEGACY_BASELINE_STRATEGY))
        params = report.get("params") if isinstance(report.get("params"), dict) else {}
        try:
            records = ast.literal_eval(str(params.get("record") or "[]"))
            portana = next(
                item
                for item in records
                if isinstance(item, dict) and item.get("class") == "PortAnaRecord"
            )
            n_drop = int(
                portana.get("kwargs", {})
                .get("config", {})
                .get("strategy", {})
                .get("kwargs", {})
                .get("n_drop", 0)
            )
            strategy["config"]["n_drop"] = n_drop
        except (ValueError, SyntaxError, StopIteration, AttributeError, TypeError):
            pass
        return strategy

    @staticmethod
    def _legacy_candidate_parameters(report: dict[str, Any]) -> dict[str, Any]:
        periods = _safe_periods(report.get("periods"))
        selection = periods.get("selection") or []
        return {
            "start_date": selection[0] if len(selection) == 2 else "",
            "end_date": selection[1] if len(selection) == 2 else "",
            "topk": 50,
            "cost_bps": 15.0,
        }

    def _resolve_selection(self, run_id: str, model_id: str, strategy_id: str) -> dict[str, Any]:
        if model_id == "latest":
            immutable_report = self.reports_root / "training_runs" / f"{run_id}.json"
            report = _load_json(
                immutable_report
                if immutable_report.is_file()
                else self.reports_root / "latest_training_result.json"
            )
            if report.get("status") != "completed" or report.get("run_id") != run_id:
                raise ValueError("Run is not available in the current training results")
            experiment_id = str(report.get("experiment_id") or "")
            self._validate_experiment_id(experiment_id)
            normalized_strategy_id = "baseline_daily" if strategy_id == "qlib_default" else strategy_id
            contract = self._validated_replay_contract(report.get("replay_contract"))
            available_strategies = (
                contract["strategies"]
                if contract is not None
                else [self._legacy_baseline_strategy(report)]
            )
            strategy = next(
                (
                    item
                    for item in available_strategies
                    if item["id"] == normalized_strategy_id
                ),
                None,
            )
            if strategy is None:
                raise ValueError("Strategy is not frozen in the selected training Run")
            exact_strategy_ids = self._exact_strategy_ids_for_report(contract, report)
            return {
                "experiment_id": experiment_id,
                "model_label": str(report.get("experiment_name") or "Latest training result"),
                "strategy_id": normalized_strategy_id,
                "strategy_label": strategy["label"],
                "strategy_kwargs": strategy["config"],
                "provider": str(report.get("provider") or "unknown"),
                "provider_uri": self._resolve_report_provider(report),
                "provider_identity": dict(report["provider_identity"]),
                "exact_replay_available": normalized_strategy_id in exact_strategy_ids,
                "replay_engine": (
                    contract.get("replay_engine")
                    if contract is not None
                    and normalized_strategy_id in exact_strategy_ids
                    else "dsa_optimized_strategy"
                ),
                "contract_version": contract.get("contract_version") if contract else None,
                "config_hash": contract.get("config_hash") if contract else None,
                "contract": contract,
                "saved_metrics": (
                    dict(report.get("metrics"))
                    if isinstance(report.get("metrics"), dict)
                    else {}
                ),
                "execution_assumptions": (
                    dict(contract.get("qlib_backtest_assumptions") or {})
                    if contract is not None
                    else {}
                ),
                "original_parameters": (
                    contract["original_parameters"]
                    if contract is not None
                    else self._legacy_baseline_parameters(report)
                ),
                "artifact_sha256": (
                    dict(report.get("artifact_sha256"))
                    if isinstance(report.get("artifact_sha256"), dict)
                    else {}
                ),
            }

        report, model = self._candidate_report_for_run(run_id, model_id)
        evaluations = (
            report.get("strategy_evaluations")
            if isinstance(report.get("strategy_evaluations"), list)
            else []
        )
        evaluation = next(
            (
                item
                for item in evaluations
                if isinstance(item, dict)
                and item.get("model_id") == model_id
                and item.get("strategy_id") == strategy_id
            ),
            None,
        )
        if evaluation is None:
            raise ValueError("Strategy is not available for the selected trained model")
        experiment_id = str(model.get("experiment_id") or "")
        self._validate_experiment_id(experiment_id)
        contract = self._validated_replay_contract(report.get("backtest_contract"))
        frozen_strategy = next(
            (
                item
                for item in (contract.get("strategies") if contract else [])
                if item["id"] == strategy_id
            ),
            None,
        )
        if frozen_strategy is not None:
            strategy_label = frozen_strategy["label"]
            strategy_kwargs = frozen_strategy["config"]
        else:
            strategy_label = str(evaluation.get("strategy_label") or strategy_id)
            strategy_kwargs = self._validated_strategy_kwargs(
                evaluation.get("strategy_config")
            )
        exact_strategy_ids: set[str] = set()
        return {
            "experiment_id": experiment_id,
            "model_label": str(model.get("label") or model_id),
            "strategy_id": strategy_id,
            "strategy_label": strategy_label,
            "strategy_kwargs": strategy_kwargs,
            "provider": str(report.get("provider") or "unknown"),
            "provider_uri": self._resolve_report_provider(report),
            "provider_identity": dict(report["provider_identity"]),
            "exact_replay_available": strategy_id in exact_strategy_ids,
            "replay_engine": (
                contract.get("replay_engine")
                if contract is not None and strategy_id in exact_strategy_ids
                else "dsa_optimized_strategy"
            ),
            "contract_version": contract.get("contract_version") if contract else None,
            "config_hash": contract.get("config_hash") if contract else None,
            "contract": contract,
            "saved_metrics": {},
            "execution_assumptions": {},
            "original_parameters": (
                contract["original_parameters"]
                if contract is not None
                else self._legacy_candidate_parameters(report)
            ),
            "artifact_sha256": (
                dict(model.get("artifact_sha256"))
                if isinstance(model.get("artifact_sha256"), dict)
                else {}
            ),
        }

    def _candidate_report_for_run(
        self, run_id: str, model_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        paths = [self.reports_root / "candidate_pipeline_latest.json"]
        paths.extend(
            sorted(
                self.reports_root.glob("candidate_pipeline_[0-9]*.json"),
                reverse=True,
            )[:500]
        )
        seen: set[Path] = set()
        saw_dry_run = False
        for path in paths:
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                report = _load_json(path)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
                continue
            models = (
                report.get("models")
                if isinstance(report.get("models"), list)
                else []
            )
            model = next(
                (
                    item
                    for item in models
                    if isinstance(item, dict)
                    and item.get("id") == model_id
                    and item.get("run_id") == run_id
                    and item.get("status") == "completed"
                ),
                None,
            )
            if model is None:
                continue
            if report.get("dry_run") is True:
                saw_dry_run = True
                continue
            return report, model
        if saw_dry_run:
            raise ValueError("Synthetic test result cannot be backtested")
        raise ValueError("Run is not available in the current training results")

    def _resolve_report_provider(self, report: dict[str, Any]) -> Path:
        identity = report.get("provider_identity")
        if not isinstance(identity, dict):
            raise ValueError(
                "Training result does not record an immutable Qlib Provider; "
                "rerun training before recalculating this strategy"
            )
        try:
            lineage = normalize_provider_lineage(identity)
            return Path(
                self.provider_resolver(lineage, self.quant_data_root)
            ).resolve()
        except RunProvenanceError as exc:
            raise ValueError(
                "Training result provider identity cannot be verified; "
                "the strategy backtest was not run"
            ) from exc

    @staticmethod
    def _validated_strategy_kwargs(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError(
                "Training result does not contain a frozen strategy configuration"
            )
        kwargs = dict(value)
        unsupported = set(kwargs) - ALLOWED_STRATEGY_KWARGS
        if unsupported:
            raise ValueError(f"Strategy configuration contains unsupported fields: {', '.join(sorted(unsupported))}")
        return kwargs

    def _read_artifact(
        self,
        experiment_id: str,
        run_id: str,
        filename: str,
        *,
        expected_provider_identity: dict[str, Any],
        expected_provider_path: Path,
        expected_artifact_sha256: str | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        value, provenance = self._read_pickle_artifact(
            experiment_id,
            run_id,
            filename,
            expected_provider_identity=expected_provider_identity,
            expected_provider_path=expected_provider_path,
            expected_artifact_sha256=expected_artifact_sha256,
        )
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise ValueError(f"Saved training artifact has an unsupported type: {filename}")
        frame = value.to_frame() if isinstance(value, pd.Series) else value
        return frame, provenance

    def _read_pickle_artifact(
        self,
        experiment_id: str,
        run_id: str,
        filename: str,
        *,
        expected_provider_identity: dict[str, Any],
        expected_provider_path: Path,
        expected_artifact_sha256: str | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        root = self.mlruns_root.resolve()
        path = (root / experiment_id / run_id / "artifacts" / filename).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Resolved artifact path escapes the MLflow data root")
        if not path.is_file():
            raise FileNotFoundError(f"Saved training artifact is missing: {filename}")
        try:
            with path.open("rb") as handle:
                artifact_bytes = handle.read(MAX_PICKLE_BYTES + 1)
        except OSError as exc:
            raise ValueError(
                f"Saved training artifact cannot be read: {filename}"
            ) from exc
        if len(artifact_bytes) > MAX_PICKLE_BYTES:
            raise ValueError(f"Saved training artifact is too large: {filename}")
        loaded_artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        try:
            provenance = self.artifact_validator(
                artifact_path=path,
                artifact_relative_path=Path(filename),
                run_id=run_id,
                mlruns_dir=root,
                quant_data_root=self.quant_data_root,
            )
        except (RunProvenanceError, OSError, UnicodeError) as exc:
            raise ValueError(
                f"Saved training artifact provenance cannot be verified: {filename}"
            ) from exc
        actual_identity = provenance.get("provider_identity")
        if not provider_lineages_match(actual_identity, expected_provider_identity):
            raise ValueError(
                f"Saved training artifact does not match the training report provider: {filename}"
            )
        actual_provider_path = Path(
            str(provenance.get("provider_path") or "")
        ).expanduser().resolve()
        if actual_provider_path != Path(expected_provider_path).expanduser().resolve():
            raise ValueError(
                f"Saved training artifact does not match the training report provider: {filename}"
            )
        artifact_sha256 = str(provenance.get("artifact_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256):
            raise ValueError(
                f"Saved training artifact provenance is incomplete: {filename}"
            )
        if artifact_sha256 != loaded_artifact_sha256:
            raise ValueError(
                f"Saved training artifact changed while being verified: {filename}"
            )
        if expected_artifact_sha256 is not None and artifact_sha256 != str(
            expected_artifact_sha256
        ):
            raise ValueError(
                f"Saved training artifact does not match the frozen training report: {filename}"
            )
        value = pd.read_pickle(io.BytesIO(artifact_bytes))
        return value, dict(provenance)

    @staticmethod
    def _validate_experiment_id(experiment_id: str) -> None:
        if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
            raise ValueError("Training result contains an invalid experiment ID")
