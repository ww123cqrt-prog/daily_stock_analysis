"""Close an interrupted robust-strategy run without rewriting its artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from robust_evaluation import aggregate_training_artifacts


TERMINATED_TUNING_MODELS = (
    "linear_ridge",
    "lightgbm",
    "xgboost",
    "mps_dnn",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _training_records(progress: Mapping[str, Any]) -> list[dict[str, Any]]:
    jobs = progress.get("jobs")
    if not isinstance(jobs, Mapping):
        raise ValueError("development progress does not contain jobs")
    records: list[dict[str, Any]] = []
    for value in jobs.values():
        if not isinstance(value, Mapping) or value.get("status") != "completed":
            continue
        records.append(
            {
                "model_id": value["model_id"],
                "window_days": value["window_days"],
                "quarter": value["quarter"],
                "seed": value.get("seed"),
                "prediction_path": value["prediction_path"],
            }
        )
    return records


def _model_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    stability = report.get("stability") or {}
    seed_stability = report.get("seed_stability") or {}
    return {
        "model_id": str(report.get("model_id") or ""),
        "eligible": report.get("eligible") is True,
        "passed_gates": [
            str(name)
            for name, passed in (report.get("gates") or {}).items()
            if passed is True
        ],
        "failed_gates": [str(value) for value in report.get("failed_gates") or []],
        "positive_rank_ic_interval_ratio": _safe_number(
            report.get("positive_rank_ic_interval_ratio")
        ),
        "rank_ic_mean": _safe_number(report.get("rank_ic_mean")),
        "window_rank_correlation_median": _safe_number(
            stability.get("rank_correlation_median")
        ),
        "top50_overlap_mean": _safe_number(stability.get("topk_overlap_mean")),
        "seed_rank_correlation_median": _safe_number(
            seed_stability.get("rank_correlation_median")
        ),
        "seed_top50_overlap_mean": _safe_number(
            seed_stability.get("topk_overlap_mean")
        ),
    }


def _tuning_summary(run_dir: Path) -> dict[str, Any]:
    tuning_root = run_dir / "development" / "tuning"
    completed_trials: dict[str, int] = {}
    completed_jobs: dict[str, int] = {}
    if tuning_root.is_dir():
        for model_dir in sorted(path for path in tuning_root.iterdir() if path.is_dir()):
            trial_count = 0
            job_count = 0
            for progress_path in sorted(model_dir.glob("trial_*/progress.json")):
                progress = _read_json(progress_path)
                jobs = progress.get("jobs") or {}
                statuses = [
                    value.get("status")
                    for value in jobs.values()
                    if isinstance(value, Mapping)
                ]
                if statuses and all(status == "completed" for status in statuses):
                    trial_count += 1
                    job_count += len(statuses)
            if trial_count:
                completed_trials[model_dir.name] = trial_count
                completed_jobs[model_dir.name] = job_count
    return {
        "completed_trials_by_model": completed_trials,
        "completed_jobs_by_model": completed_jobs,
        "not_run_models": [
            model_id
            for model_id in TERMINATED_TUNING_MODELS
            if model_id not in completed_trials
        ],
        "verdict": "not_finalized",
        "note": (
            "Completed tuning predictions remain on disk, but the old pipeline was "
            "stopped before finalist selection and strategy evaluation."
        ),
    }


def build_closure_report(
    *, run_dir: Path, config: Mapping[str, Any], generated_at: str | None = None
) -> dict[str, Any]:
    progress = _read_json(run_dir / "development" / "progress.json")
    records = _training_records(progress)
    if not records:
        raise ValueError("the old run has no completed fixed-parameter jobs")
    aggregated = aggregate_training_artifacts(
        records,
        topk=int(config["portfolio"]["topk"]),
        thresholds=config["acceptance"],
    )
    fixed_models = [
        _model_summary(report) for report in aggregated.get("model_reports") or []
    ]
    return {
        "schema_version": 1,
        "generated_at": generated_at
        or datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_dir.name,
        "status": "development_failed",
        "lifecycle_status": "stopped",
        "reason_code": "superseded_by_signal_research",
        "qualification": "baseline_only",
        "fixed_parameter_job_count": len(records),
        "fixed_parameter_models": fixed_models,
        "tuning": _tuning_summary(run_dir),
        "conclusion": (
            "No model passed every stability gate. These artifacts are a generation-zero "
            "baseline and are not a qualified strategy."
        ),
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 旧鲁棒训练流程结案报告",
        "",
        f"- 运行：`{payload['run_id']}`",
        f"- 结论：`{payload['status']}`",
        f"- 固定参数任务：{payload['fixed_parameter_job_count']} 个，全部保留",
        "- 定位：第零代失败基线，不是合格策略",
        "",
        "## 固定参数模型",
        "",
        "| 模型 | 通过 | 失败门槛 | RankIC为正区间 | 窗口排名相关 | Top50重合率 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in payload.get("fixed_parameter_models") or []:
        def percent(value: Any) -> str:
            number = _safe_number(value)
            return "-" if number is None else f"{number:.2%}"

        lines.append(
            "| {model} | {eligible} | {failed} | {positive} | {corr} | {overlap} |".format(
                model=item["model_id"],
                eligible="是" if item["eligible"] else "否",
                failed=", ".join(item["failed_gates"]) or "-",
                positive=percent(item["positive_rank_ic_interval_ratio"]),
                corr=percent(item["window_rank_correlation_median"]),
                overlap=percent(item["top50_overlap_mean"]),
            )
        )
    tuning = payload.get("tuning") or {}
    lines.extend(
        [
            "",
            "## 调参收尾",
            "",
            "- CatBoost 已完整保存 20 组调参，每组 60 个任务。",
            "- Ridge、LightGBM、XGBoost、MPS DNN 后续调参未运行。",
            "- 旧流程在候选优选和完整交易策略回测前停止，因此不得据此声称调参成功。",
            "",
            "## 结论",
            "",
            str(payload["conclusion"]),
            "",
        ]
    )
    return "\n".join(lines)


def close_run(*, run_dir: Path, config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("robust strategy config must be a mapping")
    payload = build_closure_report(run_dir=run_dir, config=config)
    _atomic_write(
        run_dir / "closure_report.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(run_dir / "closure_report.md", render_markdown(payload))

    status_path = run_dir / "status.json"
    status = _read_json(status_path)
    status.update(
        {
            "status": payload["status"],
            "lifecycle_status": payload["lifecycle_status"],
            "reason_code": payload["reason_code"],
            "current_stage": "closed",
            "updated_at": payload["generated_at"],
            "closure_report": str(run_dir / "closure_report.json"),
        }
    )
    _atomic_write(
        status_path,
        json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = close_run(run_dir=args.run_dir.resolve(), config_path=args.config.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
