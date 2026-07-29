#!/usr/bin/env python3
"""Generate a read-only summary of the latest Qlib prediction scores.

The report reads Qlib/MLflow `pred.pkl` artifacts, extracts the latest trading
date, and writes a small top/bottom score summary for the DSA web workbench.
It does not connect to broker software, place orders, or mutate Qlib data.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _mlflow_run_provenance import (  # noqa: E402
    RunProvenanceError,
    validate_file_store_run_artifact,
)
from _paths import (  # noqa: E402
    mlruns_root,
    projects_root,
    quant_data_root,
    reports_root,
    workflow_root,
)


WORKFLOW = workflow_root()
ROOT = projects_root()
REPORT_DIR = reports_root()
MLRUNS_DIR = mlruns_root()
LATEST_TRAINING_RESULT = REPORT_DIR / "latest_training_result.json"
DEFAULT_TOPK = 20


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _safe_run_id(value: Any) -> str:
    raw = str(value or "").strip()
    if raw and not re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        raise ValueError("run_id may only contain letters, numbers, underscores, or hyphens.")
    return raw


def find_prediction_path(
    latest_training_path: Path = LATEST_TRAINING_RESULT,
    mlruns_dir: Path = MLRUNS_DIR,
    run_id: str | None = None,
) -> Path:
    """Locate the Qlib `pred.pkl` artifact for the latest or requested run."""
    requested_run_id = _safe_run_id(run_id)
    if not requested_run_id:
        latest = _read_json_object(latest_training_path) or {}
        requested_run_id = _safe_run_id(latest.get("run_id"))

    if not requested_run_id:
        raise FileNotFoundError(
            "Could not determine a Qlib training run id; refusing to select an "
            "unrelated pred.pkl artifact."
        )

    matches = sorted(mlruns_dir.glob(f"*/{requested_run_id}/artifacts/pred.pkl"))
    if len(matches) == 1 and matches[0].is_file() and not matches[0].is_symlink():
        return matches[0]
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Found multiple Qlib pred.pkl artifacts for run {requested_run_id} "
            f"under {mlruns_dir}."
        )
    raise FileNotFoundError(
        f"Could not find Qlib pred.pkl for run {requested_run_id} under {mlruns_dir}."
    )


def validate_prediction_provenance(
    *,
    prediction_path: Path,
    run_id: str,
    mlruns_dir: Path,
    quant_data_root_path: Path,
) -> dict[str, Any]:
    """Bind one prediction artifact to its exact immutable V2 provider."""

    safe_run_id = _safe_run_id(run_id)
    if not safe_run_id:
        raise ValueError("prediction run id is required for provider validation")
    try:
        return validate_file_store_run_artifact(
            artifact_path=prediction_path,
            artifact_relative_path=Path("pred.pkl"),
            run_id=safe_run_id,
            mlruns_dir=mlruns_dir,
            quant_data_root=quant_data_root_path,
        )
    except RunProvenanceError as exc:
        raise ValueError(
            f"prediction run provider identity is invalid: {exc}"
        ) from exc


def load_prediction_scores(path: Path) -> pd.DataFrame:
    """Load a Qlib prediction artifact into datetime/instrument/score rows."""
    payload = pd.read_pickle(path)
    if isinstance(payload, pd.Series):
        frame = payload.to_frame(name=payload.name or "score")
    elif isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        raise ValueError(f"Prediction artifact must be a pandas DataFrame or Series: {path}")

    score_column = "score" if "score" in frame.columns else str(frame.columns[0])
    if isinstance(frame.index, pd.MultiIndex):
        rows = frame[[score_column]].reset_index()
    else:
        rows = frame[[score_column]].reset_index()
        if "datetime" not in rows.columns and "date" in rows.columns:
            rows = rows.rename(columns={"date": "datetime"})

    if "datetime" not in rows.columns or "instrument" not in rows.columns:
        raise ValueError("Prediction artifact must include datetime and instrument index/columns.")

    rows = rows.rename(columns={score_column: "score"})
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["instrument"] = rows["instrument"].astype(str)
    rows["score"] = pd.to_numeric(rows["score"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "instrument", "score"])
    rows = rows.replace([math.inf, -math.inf], pd.NA).dropna(subset=["score"])
    if rows.empty:
        raise ValueError(f"Prediction artifact has no usable scores: {path}")
    return rows[["datetime", "instrument", "score"]].sort_values(["datetime", "instrument"]).reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "instrument": str(row.instrument),
            "score": _safe_float(row.score),
        }
        for row in frame.itertuples(index=False)
    ]


def summarize_latest_predictions(scores: pd.DataFrame, topk: int = DEFAULT_TOPK) -> dict[str, Any]:
    if topk < 1:
        raise ValueError("topk must be at least 1.")

    latest_date = pd.Timestamp(scores["datetime"].max()).normalize()
    latest_rows = scores.loc[scores["datetime"].dt.normalize() == latest_date].copy()
    latest_rows = latest_rows.sort_values("score", ascending=False)
    if latest_rows.empty:
        raise ValueError("Prediction artifact has no rows for the latest date.")

    latest_scores = latest_rows["score"]
    return {
        "latest_date": str(latest_date.date()),
        "score_column": "score",
        "top": _records(latest_rows.head(topk)),
        "bottom": _records(latest_rows.sort_values("score", ascending=True).head(topk)),
        "summary": {
            "rows": int(len(scores)),
            "latest_rows": int(len(latest_rows)),
            "date_count": int(scores["datetime"].dt.normalize().nunique()),
            "score_min": _safe_float(latest_scores.min()),
            "score_max": _safe_float(latest_scores.max()),
            "score_mean": _safe_float(latest_scores.mean()),
        },
    }


def _markdown_table(items: list[dict[str, Any]]) -> str:
    lines = ["| Instrument | Score |", "| --- | ---: |"]
    for item in items:
        score = item.get("score")
        score_text = f"{score:.8f}" if isinstance(score, (int, float)) else "n/a"
        lines.append(f"| `{item.get('instrument')}` | {score_text} |")
    return "\n".join(lines)


def build_markdown(payload: dict[str, Any]) -> str:
    return f"""# Qlib Prediction Summary

Generated: {payload.get("generated_at")}

## Scope

This is a read-only model score summary from Qlib `pred.pkl`. It is not a
trading signal by itself, does not connect to broker software, 不读取交易密码、
不自动下单，也不改写 Qlib provider。

## Inputs

- Prediction artifact: `{payload.get("prediction_path")}`
- Run id: `{payload.get("run_id") or "n/a"}`
- Experiment: `{payload.get("experiment_name") or "n/a"}`
- Latest prediction date: `{payload.get("latest_date")}`
- Top/bottom count: `{payload.get("topk")}`

## Highest Scores

{_markdown_table(payload.get("top") or [])}

## Lowest Scores

{_markdown_table(payload.get("bottom") or [])}

## Interpretation

- Higher score means the model ranked the instrument more favorably for the
  learned label on this date.
- Use this together with liquidity, risk control, position sizing, and a
  validated backtest. Do not treat it as an automatic buy/sell instruction.
"""


def generate_report(
    prediction_path: Path,
    out_dir: Path,
    run_id: str = "",
    experiment_name: str = "",
    topk: int = DEFAULT_TOPK,
    preflight_unavailable_reason: str = "",
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    unavailable_reason = preflight_unavailable_reason
    if not unavailable_reason and prediction_path.exists() and not provenance:
        unavailable_reason = "prediction artifact provenance was not validated"
    if unavailable_reason:
        summary = {
            "latest_date": "n/a",
            "score_column": "score",
            "top": [],
            "bottom": [],
            "summary": {
                "rows": 0,
                "latest_rows": 0,
                "date_count": 0,
                "score_min": None,
                "score_max": None,
                "score_mean": None,
            },
        }
    else:
        try:
            scores = load_prediction_scores(prediction_path)
            summary = summarize_latest_predictions(scores, topk=topk)
        except (FileNotFoundError, ValueError) as exc:
            unavailable_reason = str(exc)
            summary = {
                "latest_date": "n/a",
                "score_column": "score",
                "top": [],
                "bottom": [],
                "summary": {
                    "rows": 0,
                    "latest_rows": 0,
                    "date_count": 0,
                    "score_min": None,
                    "score_max": None,
                    "score_mean": None,
                },
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"qlib_predictions_{ts}.json"
    md_path = out_dir / f"qlib_predictions_{ts}.md"
    latest_json = out_dir / "qlib_predictions_latest.json"
    latest_md = out_dir / "qlib_predictions_latest.md"

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "read_only_qlib_prediction_summary",
        "run_id": _safe_run_id(run_id),
        "experiment_name": str(experiment_name or ""),
        "prediction_path": str(prediction_path),
        "topk": int(topk),
        "unavailable_reason": unavailable_reason,
        "provenance_status": "validated" if provenance else "unavailable",
        "artifact_sha256": str((provenance or {}).get("artifact_sha256") or ""),
        "provider_identity": dict((provenance or {}).get("provider_identity") or {}),
        "provider_path": str((provenance or {}).get("provider_path") or ""),
        "artifacts": {
            "json": str(json_path),
            "markdown": str(md_path),
            "latest_json": str(latest_json),
            "latest_markdown": str(latest_md),
        },
        **summary,
    }
    markdown = build_markdown(payload)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Qlib prediction score summary.")
    parser.add_argument("--prediction-path", type=Path, default=None, help="Path to Qlib pred.pkl. Defaults to latest MLflow run.")
    parser.add_argument("--run-id", default="", help="Optional MLflow run id used to locate pred.pkl.")
    parser.add_argument("--latest-training", type=Path, default=LATEST_TRAINING_RESULT)
    parser.add_argument("--mlruns-dir", type=Path, default=MLRUNS_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    args = parser.parse_args()

    latest_training = _read_json_object(args.latest_training) or {}
    run_id = _safe_run_id(args.run_id or latest_training.get("run_id"))
    latest_training_run_id = _safe_run_id(latest_training.get("run_id"))
    experiment_name = (
        str(latest_training.get("experiment_name") or "")
        if run_id and run_id == latest_training_run_id
        else ""
    )
    preflight_unavailable_reason = ""
    provenance: dict[str, Any] | None = None
    try:
        prediction_path = args.prediction_path or find_prediction_path(
            latest_training_path=args.latest_training,
            mlruns_dir=args.mlruns_dir,
            run_id=run_id,
        )
        provenance = validate_prediction_provenance(
            prediction_path=prediction_path,
            run_id=run_id,
            mlruns_dir=args.mlruns_dir,
            quant_data_root_path=quant_data_root().expanduser().resolve(),
        )
    except (FileNotFoundError, OSError, UnicodeError, ValueError) as exc:
        preflight_unavailable_reason = str(exc)
        prediction_path = args.out_dir / "missing_pred.pkl"
    payload = generate_report(
        prediction_path=prediction_path,
        out_dir=args.out_dir,
        run_id=run_id,
        experiment_name=experiment_name,
        topk=args.topk,
        preflight_unavailable_reason=preflight_unavailable_reason,
        provenance=provenance,
    )
    print(f"Latest prediction date: {payload['latest_date']}")
    print(f"Latest markdown: {payload['artifacts']['latest_markdown']}")
    print(f"Latest JSON: {payload['artifacts']['latest_json']}")
    return 1 if payload["unavailable_reason"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
