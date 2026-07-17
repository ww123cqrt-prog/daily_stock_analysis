#!/usr/bin/env python3
"""Run one fail-closed day of frozen robust-strategy paper tracking."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import cloudpickle
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from prepare_robust_provider import prepare_runtime_provider  # noqa: E402
from robust_backtest import simulate_strategy  # noqa: E402
from robust_market_data import (
    build_market_regimes,
    load_benchmark_frame,
    load_market_panel,
)  # noqa: E402
from robust_shadow_tracking import (  # noqa: E402
    _atomic_write,
    _file_guard,
    _read_json,
    append_observation,
    append_signal_predictions,
    new_shadow_state,
)
from robust_strategy_core import RollingSplit  # noqa: E402
from run_robust_development import (
    TREE_MODELS,
    build_dataset_config,
    selection_contract_hash,
)  # noqa: E402
from run_robust_strategy import (
    DEFAULT_CONFIG,
    file_sha256,
    git_worktree_state,
    load_config,
)  # noqa: E402


REPLAY_START = "2026-01-01"
SPLIT_FIELDS = (
    "train_start",
    "train_end",
    "validation_start",
    "validation_end",
    "test_start",
    "test_end",
)


class ShadowBlocked(RuntimeError):
    def __init__(self, reason_code: str):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


@contextmanager
def _claim_lock(report_dir: Path):
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / ".shadow-day.lock").open("a+") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            with _file_guard(report_dir / "shadow_state_latest.json"):
                yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _state_summary(state: Mapping[str, Any] | None) -> dict[str, Any]:
    state = state or {}
    return {
        "state_status": state.get("status"),
        "state_metrics": state.get("metrics")
        if isinstance(state.get("metrics"), dict)
        else None,
    }


def _daily_payload(
    *,
    status: str,
    reason_code: str | None,
    as_of_date: str,
    inference_success: bool,
    appended_days: int,
    state: Mapping[str, Any] | None,
    outcome: str = "completed",
    error_type: str | None = None,
) -> dict[str, Any]:
    payload = {
        "status": status,
        "reason_code": reason_code,
        "as_of": str(pd.Timestamp(as_of_date).date()),
        "outcome": outcome,
        "inference_success": bool(inference_success),
        "appended_days": int(appended_days),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **_state_summary(state),
    }
    if error_type:
        payload["error_type"] = str(error_type)
    return payload


def _write_daily(report_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    _write_json(report_dir / "shadow_daily_latest.json", result)
    return result


def _read_provider_metadata(provider: Path) -> dict[str, Any]:
    metadata = _read_json(Path(provider) / "ROBUST_PROVIDER.json")
    if metadata.get("format_version") != 2 or not metadata.get("canonical_provider"):
        raise ShadowBlocked("frozen_provider_metadata_invalid")
    return metadata


def _expected_artifact_keys(
    config: Mapping[str, Any], selected_models: Sequence[str]
) -> set[tuple[str, int, int | None]]:
    expected: set[tuple[str, int, int | None]] = set()
    for model_id in selected_models:
        seeds: Sequence[int | None] = (
            config["training"]["tree_seeds"] if model_id in TREE_MODELS else [None]
        )
        for window in config["training"]["windows_trading_days"]:
            for seed in seeds:
                expected.add(
                    (str(model_id), int(window), None if seed is None else int(seed))
                )
    return expected


def _validate_frozen_inputs(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    provider_state: Mapping[str, Any],
    git_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], Path, Path]:
    if (
        confirmation.get("status") != "shadow_observation"
        or frozen.get("confirmation_status") != "shadow_observation"
    ):
        raise ShadowBlocked("shadow_not_confirmed")
    if git_state.get("clean") is not True or git_state.get("commit") != frozen.get(
        "git_commit"
    ):
        raise ShadowBlocked("frozen_git_state_mismatch")
    config_hash = file_sha256(config_path)
    if (
        len(str(frozen.get("config_hash") or "")) != 64
        or frozen.get("config_hash") != config_hash
    ):
        raise ShadowBlocked("frozen_config_mismatch")

    selected_models = [str(value) for value in frozen.get("selected_models") or []]
    selected_configs = frozen.get("selected_model_configs")
    selected_strategy = str(frozen.get("selected_strategy") or "")
    known_models = {str(item["id"]) for item in config["models"]}
    if (
        not selected_models
        or len(selected_models) != len(set(selected_models))
        or not set(selected_models).issubset(known_models)
        or not isinstance(selected_configs, dict)
        or not selected_strategy
    ):
        raise ShadowBlocked("frozen_selection_mismatch")
    expected_selection_hash = selection_contract_hash(
        selected_models, selected_configs, selected_strategy
    )
    if frozen.get("selection_hash") != expected_selection_hash:
        raise ShadowBlocked("frozen_selection_mismatch")
    if (
        confirmation.get("config_hash") != config_hash
        or confirmation.get("selected_models") != selected_models
        or confirmation.get("selected_strategy") != selected_strategy
        or confirmation.get("run_id") != frozen.get("confirmation_run_id")
        or confirmation.get("latest_market_day") != frozen.get("shadow_start_after")
    ):
        raise ShadowBlocked("confirmation_contract_mismatch")

    artifacts = frozen.get("confirmation_model_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ShadowBlocked("model_artifact_invalid")
    observed_keys: set[tuple[str, int, int | None]] = set()
    normalized_artifacts: list[dict[str, Any]] = []
    for raw in artifacts:
        if not isinstance(raw, dict):
            raise ShadowBlocked("model_artifact_invalid")
        required = {
            "job_key",
            "model_id",
            "window_days",
            "seed",
            "model_path",
            "model_sha256",
            "split",
        }
        if not required.issubset(raw):
            raise ShadowBlocked("model_artifact_invalid")
        split = raw.get("split")
        if not isinstance(split, dict) or any(
            not split.get(field) for field in SPLIT_FIELDS
        ):
            raise ShadowBlocked("model_artifact_invalid")
        model_path = Path(str(raw["model_path"]))
        expected_hash = str(raw["model_sha256"])
        if (
            len(expected_hash) != 64
            or not model_path.is_file()
            or file_sha256(model_path) != expected_hash
        ):
            raise ShadowBlocked("model_artifact_invalid")
        key = (
            str(raw["model_id"]),
            int(raw["window_days"]),
            None if raw["seed"] is None else int(raw["seed"]),
        )
        if key in observed_keys:
            raise ShadowBlocked("model_artifact_invalid")
        observed_keys.add(key)
        normalized_artifacts.append(dict(raw))
    if observed_keys != _expected_artifact_keys(config, selected_models):
        raise ShadowBlocked("model_artifact_invalid")
    if confirmation.get("confirmation_model_artifact_count") != len(
        normalized_artifacts
    ):
        raise ShadowBlocked("confirmation_contract_mismatch")

    ensemble_path = Path(str(frozen.get("confirmation_ensemble_prediction_path") or ""))
    ensemble_hash = str(frozen.get("confirmation_ensemble_prediction_hash") or "")
    if (
        len(ensemble_hash) != 64
        or not ensemble_path.is_file()
        or file_sha256(ensemble_path) != ensemble_hash
        or confirmation.get("confirmation_ensemble_prediction_path")
        != str(ensemble_path)
        or confirmation.get("confirmation_ensemble_prediction_hash") != ensemble_hash
    ):
        raise ShadowBlocked("confirmation_artifact_invalid")

    frozen_provider = Path(str(frozen.get("provider_uri") or ""))
    if (
        provider_state.get("status") != "completed"
        or not frozen_provider.is_dir()
        or Path(str(provider_state.get("provider_uri") or "")).resolve()
        != frozen_provider.resolve()
        or provider_state.get("provider_key") != frozen.get("provider_key")
    ):
        raise ShadowBlocked("frozen_provider_mismatch")
    return normalized_artifacts, ensemble_path, frozen_provider


def _file_is_prefix(prefix_path: Path, current_path: Path) -> bool:
    try:
        if prefix_path.stat().st_size > current_path.stat().st_size:
            return False
        with prefix_path.open("rb") as prefix, current_path.open("rb") as current:
            while True:
                expected = prefix.read(1024 * 1024)
                if not expected:
                    return True
                if current.read(len(expected)) != expected:
                    return False
    except OSError:
        return False


def verify_provider_is_append_only(
    frozen_provider: Path, current_provider: Path
) -> None:
    frozen_provider = Path(frozen_provider)
    current_provider = Path(current_provider)
    frozen_calendar = frozen_provider / "calendars" / "day.txt"
    current_calendar = current_provider / "calendars" / "day.txt"
    if (
        frozen_calendar.is_symlink()
        or current_calendar.is_symlink()
        or not frozen_calendar.is_file()
        or not current_calendar.is_file()
        or not _file_is_prefix(frozen_calendar, current_calendar)
    ):
        raise ValueError(
            "frozen provider calendar is not a byte prefix of the current calendar"
        )
    frozen_features = frozen_provider / "features"
    current_features = current_provider / "features"
    if (
        frozen_features.is_symlink()
        or current_features.is_symlink()
        or not frozen_features.is_dir()
    ):
        raise ValueError("provider features are missing or unsafe")
    for frozen_path in sorted(frozen_features.rglob("*")):
        if frozen_path.is_symlink():
            raise ValueError("frozen provider features contain a symlink")
        if not frozen_path.is_file():
            continue
        relative = frozen_path.relative_to(frozen_features)
        current_path = current_features / relative
        if (
            current_path.is_symlink()
            or not current_path.is_file()
            or not _file_is_prefix(frozen_path, current_path)
        ):
            raise ValueError(
                "a frozen provider feature is not a byte prefix of the current feature"
            )


def _normalize_prediction_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if {"datetime", "instrument", "score"}.issubset(frame.columns):
        result = frame.loc[:, ["datetime", "instrument", "score"]].copy()
    elif isinstance(frame.index, pd.MultiIndex) and {"datetime", "instrument"}.issubset(
        frame.index.names
    ):
        if "score" not in frame.columns:
            raise ValueError(f"prediction {name} has no score")
        result = (
            frame.loc[:, ["score"]]
            .reset_index()
            .loc[:, ["datetime", "instrument", "score"]]
        )
    else:
        raise ValueError(f"prediction {name} must use datetime/instrument keys")
    result["datetime"] = pd.to_datetime(
        result["datetime"], errors="coerce"
    ).dt.normalize()
    result["instrument"] = result["instrument"].astype(str).str.strip().str.upper()
    result["score"] = pd.to_numeric(result["score"], errors="coerce")
    if (
        result[["datetime", "score"]].isna().any(axis=None)
        or result["instrument"].eq("").any()
        or result.duplicated(["datetime", "instrument"]).any()
        or not result["score"].map(math.isfinite).all()
    ):
        raise ValueError(f"prediction {name} must have unique keys and finite scores")
    return result.sort_values(["datetime", "instrument"], kind="stable").reset_index(
        drop=True
    )


def aggregate_daily_predictions(
    frames: Mapping[str, tuple[str, pd.DataFrame]],
    *,
    selected_models: Sequence[str],
) -> pd.DataFrame:
    if not frames:
        raise ValueError("at least one frozen prediction is required")
    normalized: dict[str, tuple[str, pd.DataFrame]] = {
        name: (str(model_id), _normalize_prediction_frame(frame, name))
        for name, (model_id, frame) in frames.items()
    }
    first = next(iter(normalized.values()))[1]
    expected_keys = pd.MultiIndex.from_frame(first.loc[:, ["datetime", "instrument"]])
    by_model: dict[str, list[pd.Series]] = {}
    for name, (model_id, frame) in normalized.items():
        keys = pd.MultiIndex.from_frame(frame.loc[:, ["datetime", "instrument"]])
        if not keys.equals(expected_keys):
            raise ValueError("prediction indexes differ across frozen jobs")
        indexed = frame.set_index(["datetime", "instrument"])["score"]
        ranked = (
            indexed.groupby(level="datetime")
            .rank(method="average", pct=True)
            .rename(name)
        )
        by_model.setdefault(model_id, []).append(ranked)
    if set(by_model) != set(str(value) for value in selected_models):
        raise ValueError("prediction model set differs from selected models")

    model_scores: list[pd.Series] = []
    for model_id in selected_models:
        within_model = pd.concat(by_model[str(model_id)], axis=1).mean(axis=1)
        model_scores.append(
            within_model.groupby(level="datetime")
            .rank(method="average", pct=True)
            .rename(str(model_id))
        )
    score = pd.concat(model_scores, axis=1).mean(axis=1).rename("score")
    return score.reset_index().loc[:, ["datetime", "instrument", "score"]]


def _split_for_inference(record: Mapping[str, Any], as_of_date: str) -> RollingSplit:
    split = record.get("split")
    if not isinstance(split, dict):
        raise ValueError("frozen artifact split is missing")
    as_of = pd.Timestamp(as_of_date).normalize()
    return RollingSplit(
        train_start=pd.Timestamp(split["train_start"]).normalize(),
        train_end=pd.Timestamp(split["train_end"]).normalize(),
        validation_start=pd.Timestamp(split["validation_start"]).normalize(),
        validation_end=pd.Timestamp(split["validation_end"]).normalize(),
        test_start=as_of,
        test_end=as_of,
    )


def _default_qlib_initializer(provider_uri: Path) -> None:
    import qlib

    qlib.init(provider_uri=str(provider_uri), region="cn", kernels=8)


def _default_dataset_factory(
    provider_uri: Path, dataset_config: dict[str, Any], split: RollingSplit
):
    from qlib.utils import init_instance_by_config

    return init_instance_by_config(dataset_config)


def infer_frozen_models(
    *,
    provider_uri: Path,
    artifacts: Sequence[Mapping[str, Any]],
    selected_models: Sequence[str],
    as_of_date: str,
    dataset_factory: Any = _default_dataset_factory,
    qlib_initializer: Any = _default_qlib_initializer,
) -> pd.DataFrame:
    qlib_initializer(Path(provider_uri))
    frames: dict[str, tuple[str, pd.DataFrame]] = {}
    as_of = pd.Timestamp(as_of_date).normalize()
    for artifact in artifacts:
        model_path = Path(str(artifact["model_path"]))
        if not model_path.is_file() or file_sha256(model_path) != artifact.get(
            "model_sha256"
        ):
            raise ValueError("frozen model hash changed")
        split = _split_for_inference(artifact, as_of_date)
        dataset_config = build_dataset_config(split)
        dataset = dataset_factory(Path(provider_uri), dataset_config, split)
        with model_path.open("rb") as stream:
            model = cloudpickle.load(stream)
        raw = model.predict(dataset, segment="test")
        if isinstance(raw, pd.Series):
            raw_frame = raw.rename("score").to_frame()
        elif isinstance(raw, pd.DataFrame) and raw.shape[1] == 1:
            raw_frame = raw.rename(columns={raw.columns[0]: "score"})
        else:
            raise ValueError("model prediction must contain exactly one score column")
        frame = _normalize_prediction_frame(raw_frame, str(artifact["job_key"]))
        if not frame["datetime"].eq(as_of).all():
            raise ValueError("model prediction contains a date other than as_of_date")
        frames[str(artifact["job_key"])] = (str(artifact["model_id"]), frame)
    return aggregate_daily_predictions(frames, selected_models=selected_models)


def _load_replay_predictions(
    confirmation_path: Path, shadow_path: Path, *, as_of_date: str
) -> pd.DataFrame:
    frames = [
        _normalize_prediction_frame(pd.read_parquet(confirmation_path), "confirmation")
    ]
    if shadow_path.is_file():
        frames.append(
            _normalize_prediction_frame(pd.read_parquet(shadow_path), "shadow")
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined["datetime"] <= pd.Timestamp(as_of_date).normalize()
    ].copy()
    duplicate = combined.duplicated(["datetime", "instrument"], keep=False)
    if duplicate.any():
        grouped = (
            combined.loc[duplicate]
            .groupby(["datetime", "instrument"])["score"]
            .nunique(dropna=False)
        )
        if (grouped > 1).any():
            raise ValueError(
                "replay predictions contain conflicting date/instrument scores"
            )
        combined = combined.drop_duplicates(["datetime", "instrument"], keep="first")
    return combined.sort_values(["datetime", "instrument"], kind="stable").reset_index(
        drop=True
    )


def _load_state(report_dir: Path, frozen: Mapping[str, Any]) -> dict[str, Any]:
    path = report_dir / "shadow_state_latest.json"
    state = _read_json(path)
    if not state:
        return new_shadow_state(
            confirmation_run_id=str(frozen["confirmation_run_id"]),
            frozen_config_hash=str(frozen["config_hash"]),
            start_after=str(frozen["shadow_start_after"]),
        )
    if (
        state.get("confirmation_run_id") != frozen.get("confirmation_run_id")
        or state.get("frozen_config_hash") != frozen.get("config_hash")
        or state.get("start_after") != frozen.get("shadow_start_after")
    ):
        raise ShadowBlocked("shadow_state_contract_mismatch")
    return state


def _normalize_daily_returns(daily: pd.DataFrame, *, as_of_date: str) -> pd.DataFrame:
    required = {"date", "portfolio_return", "benchmark_return"}
    if not required.issubset(daily.columns):
        raise ValueError("strategy replay daily output is incomplete")
    result = daily.loc[:, ["date", "portfolio_return", "benchmark_return"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result["portfolio_return"] = pd.to_numeric(
        result["portfolio_return"], errors="coerce"
    )
    result["benchmark_return"] = pd.to_numeric(
        result["benchmark_return"], errors="coerce"
    )
    result = result[result["date"] <= pd.Timestamp(as_of_date).normalize()]
    if (
        result.empty
        or result.isna().any(axis=None)
        or result.duplicated(["date"]).any()
        or not result["portfolio_return"].map(math.isfinite).all()
        or not result["benchmark_return"].map(math.isfinite).all()
    ):
        raise ValueError("strategy replay daily returns are invalid")
    return result.sort_values("date", kind="stable").reset_index(drop=True)


def _run_replay(
    *,
    config: Mapping[str, Any],
    frozen: Mapping[str, Any],
    quant_data_dir: Path,
    provider_uri: Path,
    predictions: pd.DataFrame,
    as_of_date: str,
    market_loader: Any,
    benchmark_loader: Any,
    regime_builder: Any,
    strategy_runner: Any,
) -> pd.DataFrame:
    paths = config["paths"]
    market = market_loader(
        pit_dir=quant_data_dir / paths["pit_membership"],
        price_dir=quant_data_dir / paths["price_parquet"],
        reference_dir=quant_data_dir / Path(paths["security_master"]).parent,
        start_date=REPLAY_START,
        end_date=as_of_date,
    )
    benchmark = benchmark_loader(
        provider_uri, start_date=REPLAY_START, end_date=as_of_date
    )
    portfolio = config["portfolio"]
    costs = config["costs"]
    result = strategy_runner(
        predictions=predictions,
        market=market,
        benchmark=benchmark,
        strategy_id=str(frozen["selected_strategy"]),
        regimes=regime_builder(market),
        regime_positions=config["market_regime"]["positions"],
        initial_cash=10_000_000.0,
        topk=int(portfolio["topk"]),
        fixed_position=float(config["market_regime"]["positions"]["risk_on"]),
        single_stock_max=float(portfolio["single_stock_weight_max"]),
        industry_max=float(portfolio["industry_weight_max"]),
        growth_star_max=float(portfolio["growth_and_star_weight_max"]),
        commission_rate=float(costs["commission_rate_each_side"]),
        minimum_commission=float(costs["minimum_commission_cny"]),
        slippage_bps=0.0,
        all_in_round_trip_bps=float(costs["scenarios"]["baseline_total_bps"]),
        defer_unfilled_open_orders=bool(
            config["execution_rules"]["defer_unfilled_open_orders"]
        ),
    )
    return _normalize_daily_returns(result["daily"], as_of_date=as_of_date)


def run_shadow_day(
    *,
    config_path: Path,
    quant_data_dir: Path,
    report_dir: Path,
    frozen_path: Path,
    confirmation_path: Path,
    provider_state_path: Path,
    as_of_date: str,
    provider_preparer: Any = prepare_runtime_provider,
    inference_runner: Any = infer_frozen_models,
    market_loader: Any = load_market_panel,
    benchmark_loader: Any = load_benchmark_frame,
    regime_builder: Any = build_market_regimes,
    strategy_runner: Any = simulate_strategy,
    git_state_provider: Any = git_worktree_state,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    quant_data_dir = Path(quant_data_dir).resolve()
    report_dir = Path(report_dir).resolve()
    frozen_path = Path(frozen_path).resolve()
    confirmation_path = Path(confirmation_path).resolve()
    provider_state_path = Path(provider_state_path).resolve()
    as_of = str(pd.Timestamp(as_of_date).date())

    with _claim_lock(report_dir):
        state: dict[str, Any] | None = None
        try:
            config = load_config(config_path)
            frozen = _read_json(frozen_path)
            confirmation = _read_json(confirmation_path)
            provider_state = _read_json(provider_state_path)
            artifacts, ensemble_path, frozen_provider = _validate_frozen_inputs(
                config_path=config_path,
                config=config,
                frozen=frozen,
                confirmation=confirmation,
                provider_state=provider_state,
                git_state=dict(git_state_provider()),
            )
            provider_metadata = _read_provider_metadata(frozen_provider)
        except ShadowBlocked as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code=exc.reason_code,
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                ),
            )
        except Exception as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="preflight_failed",
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                    error_type=type(exc).__name__,
                ),
            )

        intervals_path = (
            quant_data_dir
            / config["paths"]["pit_membership"]
            / "membership_intervals.parquet"
        )
        manifest_path = quant_data_dir / "DATA_MANIFEST.json"
        try:
            current_provider_state = provider_preparer(
                canonical_provider=Path(str(provider_metadata["canonical_provider"])),
                intervals_path=intervals_path,
                runtime_root=quant_data_dir / "runtime" / "robust_strategy" / "shadow",
                data_manifest_path=manifest_path,
            )
            current_provider = Path(
                str(current_provider_state.get("provider_uri") or "")
            )
            if (
                current_provider_state.get("status") != "completed"
                or not current_provider.is_dir()
            ):
                raise ValueError("current provider snapshot is unavailable")
            current_metadata = _read_provider_metadata(current_provider)
            if (
                Path(str(current_metadata["canonical_provider"])).resolve()
                != Path(str(provider_metadata["canonical_provider"])).resolve()
            ):
                raise ShadowBlocked("provider_identity_mismatch")
            verify_provider_is_append_only(frozen_provider, current_provider)
        except ShadowBlocked as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code=exc.reason_code,
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                ),
            )
        except ValueError as exc:
            reason = (
                "provider_history_changed"
                if "prefix" in str(exc)
                else "provider_prepare_failed"
            )
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code=reason,
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                    error_type=type(exc).__name__,
                ),
            )
        except Exception as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="provider_prepare_failed",
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                    error_type=type(exc).__name__,
                ),
            )

        calendar_path = current_provider / "calendars" / "day.txt"
        calendar = (
            pd.DatetimeIndex(
                pd.to_datetime(
                    calendar_path.read_text(encoding="utf-8").splitlines(),
                    errors="coerce",
                )
            )
            .dropna()
            .normalize()
        )
        if pd.Timestamp(as_of) not in calendar:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="shadow_observation",
                    reason_code="non_trading_day",
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=_read_json(report_dir / "shadow_state_latest.json"),
                    outcome="skipped",
                ),
            )

        try:
            state = _load_state(report_dir, frozen)
        except ShadowBlocked as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code=exc.reason_code,
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=None,
                ),
            )
        if state["status"] in {"robust_strategy_qualified", "development_failed"}:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status=str(state["status"]),
                    reason_code="shadow_tracking_finalized",
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=state,
                    outcome="skipped",
                ),
            )

        existing_observation = next(
            (
                item
                for item in state.get("observations") or []
                if str(item.get("date")) == as_of
            ),
            None,
        )
        recorded_success_retry = bool(
            existing_observation and existing_observation.get("task_success")
        )
        if existing_observation is not None and not recorded_success_retry:
            previous_daily = _read_json(report_dir / "shadow_daily_latest.json")
            previous_reason = (
                previous_daily.get("reason_code")
                if previous_daily.get("as_of") == as_of
                else None
            )
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code=str(previous_reason or "shadow_day_already_failed"),
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=state,
                    outcome="skipped",
                    error_type=previous_daily.get("error_type"),
                ),
            )

        start_after = pd.Timestamp(state["start_after"]).normalize()
        as_of_timestamp = pd.Timestamp(as_of).normalize()
        replay_expected_dates = [
            str(day.date())
            for day in calendar[
                (calendar > start_after) & (calendar <= as_of_timestamp)
            ]
        ]
        expected_dates = replay_expected_dates[:60]
        state_dates = [
            str(item.get("date")) for item in state.get("observations") or []
        ]
        if state_dates != expected_dates[: len(state_dates)]:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="shadow_calendar_incomplete",
                    as_of_date=as_of,
                    inference_success=False,
                    appended_days=0,
                    state=state,
                ),
            )

        inference_success = recorded_success_retry
        inference_reason: str | None = None
        inference_error_type: str | None = None
        signal_path = report_dir / "shadow_signal_predictions.parquet"
        if not recorded_success_retry:
            try:
                today = inference_runner(
                    provider_uri=current_provider,
                    artifacts=artifacts,
                    selected_models=[str(value) for value in frozen["selected_models"]],
                    as_of_date=as_of,
                )
                today = _normalize_prediction_frame(today, "today")
                if not today["datetime"].eq(pd.Timestamp(as_of)).all():
                    raise ValueError("inference returned a date other than as_of_date")
                append_signal_predictions(signal_path, today, as_of_date=as_of)
                inference_success = True
            except Exception as exc:
                inference_reason = "inference_failed"
                inference_error_type = type(exc).__name__

        try:
            replay_predictions = _load_replay_predictions(
                ensemble_path, signal_path, as_of_date=as_of
            )
            daily = _run_replay(
                config=config,
                frozen=frozen,
                quant_data_dir=quant_data_dir,
                provider_uri=current_provider,
                predictions=replay_predictions,
                as_of_date=as_of,
                market_loader=market_loader,
                benchmark_loader=benchmark_loader,
                regime_builder=regime_builder,
                strategy_runner=strategy_runner,
            )
        except Exception as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="strategy_replay_failed",
                    as_of_date=as_of,
                    inference_success=inference_success,
                    appended_days=0,
                    state=state,
                    error_type=type(exc).__name__,
                ),
            )

        eligible = daily[
            (daily["date"] > start_after) & (daily["date"] <= as_of_timestamp)
        ]
        eligible_dates = [str(day.date()) for day in eligible["date"]]
        if eligible_dates != replay_expected_dates:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="shadow_calendar_incomplete",
                    as_of_date=as_of,
                    inference_success=inference_success,
                    appended_days=0,
                    state=state,
                ),
            )
        existing_dates = {str(item["date"]) for item in state.get("observations") or []}
        existing_by_date = {
            str(item["date"]): item for item in state.get("observations") or []
        }
        appended_days = 0
        try:
            for row in eligible.itertuples(index=False):
                day = str(pd.Timestamp(row.date).date())
                if day in existing_dates:
                    existing = existing_by_date[day]
                    if float(existing["portfolio_return"]) != float(
                        row.portfolio_return
                    ) or float(existing["benchmark_return"]) != float(
                        row.benchmark_return
                    ):
                        raise ValueError(
                            "replay changed an existing shadow observation"
                        )
                    continue
                state = append_observation(
                    state,
                    {
                        "date": day,
                        "portfolio_return": float(row.portfolio_return),
                        "benchmark_return": float(row.benchmark_return),
                        "task_success": bool(
                            pd.Timestamp(row.date) == as_of_timestamp
                            and inference_success
                        ),
                    },
                )
                existing_dates.add(day)
                appended_days += 1
                if state["status"] in {
                    "robust_strategy_qualified",
                    "development_failed",
                }:
                    break
        except Exception as exc:
            return _write_daily(
                report_dir,
                _daily_payload(
                    status="development_failed",
                    reason_code="shadow_observation_conflict",
                    as_of_date=as_of,
                    inference_success=inference_success,
                    appended_days=appended_days,
                    state=state,
                    error_type=type(exc).__name__,
                ),
            )
        if appended_days:
            _write_json(report_dir / "shadow_state_latest.json", state)

        as_of_complete = bool((daily["date"] == as_of_timestamp).any())
        reason_code = inference_reason
        error_type = inference_error_type
        status = str(state["status"])
        if not as_of_complete:
            status = "development_failed"
            reason_code = "as_of_market_data_incomplete"
            error_type = None
        elif not inference_success:
            status = "development_failed"
        return _write_daily(
            report_dir,
            _daily_payload(
                status=status,
                reason_code=reason_code,
                as_of_date=as_of,
                inference_success=inference_success,
                appended_days=appended_days,
                state=state,
                error_type=error_type,
            ),
        )


def exit_code_for_result(result: Mapping[str, Any]) -> int:
    if (
        result.get("outcome") == "skipped"
        and result.get("reason_code") == "non_trading_day"
    ):
        return 0
    if result.get("status") in {
        "shadow_observation",
        "robust_strategy_qualified",
    } and not result.get("reason_code"):
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one frozen robust-strategy shadow observation day."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument(
        "--report-dir", type=Path, default=reports_root() / "robust_strategy"
    )
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--confirmation-latest", type=Path, required=True)
    parser.add_argument("--provider-latest", type=Path, required=True)
    parser.add_argument("--as-of-date", required=True)
    args = parser.parse_args(argv)
    result = run_shadow_day(
        config_path=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        frozen_path=args.frozen,
        confirmation_path=args.confirmation_latest,
        provider_state_path=args.provider_latest,
        as_of_date=args.as_of_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return exit_code_for_result(result)


if __name__ == "__main__":
    raise SystemExit(main())
