# -*- coding: utf-8 -*-
"""Read-only status of the selected, frozen QMT experiment input."""
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from fastapi import APIRouter
from src.services.quant_workflow_paths import qmt_dataset_dir

router = APIRouter()


def inspect_qmt_input(*, training_config: Path | None = None) -> tuple[dict[str, Any], dict[str, str] | None]:
    """Load once at the file boundary; keep the filesystem identity internal."""
    directory = qmt_dataset_dir()
    status = {
        "backend": "qmt_native", "status": "unconfigured",
        "validation_scope": "frozen_experiment", "available_datasets": 0,
        "datasets": [], "experiment": None,
        "sdk": {"status": "not_checked", "required_for_offline": False},
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "summary": "请先只读导出实验输入，并配置 QMT_DATASET_DIR。",
    }
    if directory is None:
        return status, None
    try:
        from quant_qmt_data import load_dataset

        frame = load_dataset(directory)
        meta = frame.attrs
        if (frame.empty or not meta["calendar"] or not meta["universe"]
                or not {"date", "symbol", "open", "high", "low", "close", "volume"}.issubset(frame.columns)):
            raise ValueError("Missing training input fields or observations")
    except ImportError:
        status.update(status="unavailable", summary="本机尚未安装 QMT 数据读取包。")
        return status, None
    except (OSError, ValueError, TypeError, KeyError):
        status.update(status="invalid", summary="实验输入缺失、不完整或校验失败，请选择完整的冻结输入。")
        return status, None
    identity = {
        "backend": "qmt_native", "experiment_id": meta["experiment_id"],
        "bars_sha256": meta["bars_sha256"], "inputs_sha256": meta["inputs_sha256"],
        "price_basis": meta["price_basis"],
        "input_dir": str(directory),
    }
    experiment = {
        key: identity[key] for key in ("backend", "experiment_id", "bars_sha256", "inputs_sha256", "price_basis")
    }
    experiment.update(
        actual_start=meta["actual_start"], actual_end=meta["actual_end"],
        calendar_start=meta["calendar"][0], calendar_end=meta["calendar"][-1],
        calendar_days=len(meta["calendar"]), symbol_count=len(meta["universe"]),
        universe_kind=meta["universe_kind"], volume_units=meta["volume_units"],
        rows=len(frame), missing_symbols=meta["missing_symbols"],
        contains_partial=False, completed_through=meta["completed_through"],
    )
    status.update(
        status="ready", available_datasets=1, experiment=experiment,
        summary="冻结实验输入可离线读取；实际训练仍按配置检查日期与特征范围。",
    )
    if training_config is not None:
        import yaml
        from quant_workflow.scripts.train_baseline_lgb import apply_information_boundaries

        try:
            config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
            apply_information_boundaries(config, meta["calendar"])
        except (OSError, ValueError, KeyError, IndexError, TypeError, yaml.YAMLError):
            status["training_preflight"] = {"status": "blocked",
                "summary": "训练配置的日期或标签窗口没有完整可用样本。"}
        else:
            status["training_preflight"] = {"status": "passed",
                "summary": "冻结输入和本次训练的标签窗口检查通过。",
                "time_semantics": config["task"]["time_semantics"]}
    return status, identity


@router.get("/status")
def quant_data_status() -> dict[str, Any]:
    """Read the input without probing QMT or claiming full-history coverage."""
    status, _identity = inspect_qmt_input()
    return status
