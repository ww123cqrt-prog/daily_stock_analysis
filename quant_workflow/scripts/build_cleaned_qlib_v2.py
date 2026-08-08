#!/usr/bin/env python3
"""Build and atomically publish the native cleaned A-share Qlib V2 dataset."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from apply_data_cleaning import build_cleaned_parquet, load_cleaning_plan  # noqa: E402
from build_qlib_v2 import (  # noqa: E402
    _import_benchmarks,
    _install_instrument_pools,
    native_benchmark_source,
)
from merge_to_qlib import convert_to_bin  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    ValidatedResolutionCache,
    atomic_write_json,
    dataset_publication_lock,
    finalize_qlib_snapshot,
    publish_qlib_snapshot,
    resolve_dataset,
    sha256_file,
)
from report_data_anomalies import build_report, scan_merged_files  # noqa: E402


DATASET_ID = "qlib_ashare_cleaned_v2"
DATASET_ROOT = Path("qlib/cn_data_ashare_cleaned_v2")
ARTIFACT_FILENAMES = {
    "anomaly_stocks": "data_anomaly_stocks.csv",
    "anomaly_events": "data_anomaly_events.csv",
    "cleaning_strategy": "data_cleaning_strategy.md",
    "cleaning_summary": "cleaning_summary.json",
}


def _upstream_contract(resolved: Any) -> dict[str, str]:
    return {
        "dataset_id": str(resolved.dataset_id),
        "snapshot_id": str(resolved.manifest["snapshot_id"]),
        "content_sha256": str(resolved.manifest["content_sha256"]),
        "price_basis": str(resolved.manifest["price_basis"]),
    }


def _assert_build_inputs_still_current(
    root: Path,
    *,
    upstream: Mapping[str, str],
    canonical_dependency: Mapping[str, str],
) -> None:
    current_qfq = resolve_dataset(
        "ashare_daily_qfq",
        quant_data_root=root,
        require_basis="qfq",
    )
    if _upstream_contract(current_qfq) != dict(upstream):
        raise DataContractError("qfq snapshot changed during cleaned build")
    current_canonical = resolve_dataset(
        "qlib_ashare_v2",
        quant_data_root=root,
        require_basis="qfq",
    )
    if _upstream_contract(current_canonical) != dict(canonical_dependency):
        raise DataContractError("canonical Qlib changed during cleaned build")


def _write_anomaly_evidence(
    source_bars: Path,
    evidence_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    evidence_dir.mkdir(parents=True, exist_ok=False)
    summary, events, coverage = scan_merged_files(source_bars)
    if events.empty:
        events = pd.DataFrame(
            columns=[
                "symbol",
                "issue_type",
                "severity",
                "count",
                "sample_dates",
                "detail",
            ]
        )
    stocks_path = evidence_dir / ARTIFACT_FILENAMES["anomaly_stocks"]
    events_path = evidence_dir / ARTIFACT_FILENAMES["anomaly_events"]
    strategy_path = evidence_dir / ARTIFACT_FILENAMES["cleaning_strategy"]
    summary.to_csv(stocks_path, index=False)
    events.to_csv(events_path, index=False)
    strategy_path.write_text(
        build_report(
            summary_df=summary,
            events_df=events,
            coverage=coverage,
            stock_csv=Path(stocks_path.name),
            events_csv=Path(events_path.name),
            merged_dir=Path("ashare_daily_qfq/bars"),
        ),
        encoding="utf-8",
    )
    blocker_count = (
        int((summary["severity"] == "blocker").sum())
        if not summary.empty and "severity" in summary
        else 0
    )
    if blocker_count:
        raise DataContractError(
            f"native cleaned Qlib input has {blocker_count} structural blockers"
        )
    return stocks_path, events_path, coverage


def _cleaning_summary(summary: Mapping[str, Any]) -> dict[str, int]:
    return {
        "source_symbols": int(summary["source_symbols"]),
        "output_symbols": int(summary["written_symbols"]),
        "source_rows": int(summary["source_rows"]),
        "output_rows": int(summary["output_rows"]),
        "excluded_symbols": int(summary["excluded_symbols"]),
        "masked_volume_rows": int(summary["masked_volume_rows"]),
        "masked_event_rows": int(summary["masked_event_rows"]),
    }


def _copy_transformation_artifacts(
    source_dir: Path,
    snapshot: Path,
) -> dict[str, dict[str, Any]]:
    target_dir = snapshot / "evidence" / "cleaning"
    target_dir.mkdir(parents=True, exist_ok=False)
    artifacts: dict[str, dict[str, Any]] = {}
    for name, filename in ARTIFACT_FILENAMES.items():
        source = source_dir / filename
        target = target_dir / filename
        shutil.copy2(source, target)
        artifacts[name] = {
            "relative_path": target.relative_to(snapshot).as_posix(),
            "size_bytes": target.stat().st_size,
            "sha256": sha256_file(target),
        }
    return artifacts


def build_cleaned_qlib_v2(
    *,
    quant_data_root: Path,
    publish: bool = False,
    publication_lock_held: bool = False,
    validated_cache: ValidatedResolutionCache | None = None,
) -> Path:
    root = Path(quant_data_root).expanduser().resolve()
    qfq = resolve_dataset(
        "ashare_daily_qfq",
        quant_data_root=root,
        require_basis="qfq",
        validated_cache=validated_cache,
    )
    canonical = resolve_dataset(
        "qlib_ashare_v2",
        quant_data_root=root,
        require_basis="qfq",
        validated_cache=validated_cache,
    )
    upstream = _upstream_contract(qfq)
    canonical_dependency = _upstream_contract(canonical)
    if canonical.manifest.get("upstream_snapshots") != [upstream]:
        raise DataContractError(
            "native canonical Qlib does not match the current qfq snapshot"
        )
    if canonical.manifest.get("benchmark_complete") is not True:
        raise DataContractError("native canonical Qlib benchmarks are incomplete")

    run_id = (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    snapshot_id = f"qlib-ashare-cleaned-v2-{run_id}"
    run_root = root / "staging" / f"qlib-cleaned-v2-{run_id}"
    evidence_work = run_root / "evidence"
    cleaned_parquet = run_root / "cleaned_parquet"
    staging = run_root / snapshot_id

    try:
        stocks_path, events_path, coverage = _write_anomaly_evidence(
            qfq.snapshot_path / "bars",
            evidence_work,
        )
        plan = load_cleaning_plan(stocks_path, events_path)
        cleaning = build_cleaned_parquet(
            qfq.snapshot_path / "bars",
            cleaned_parquet,
            plan,
            overwrite=True,
        )
        normalized_summary = _cleaning_summary(cleaning)
        atomic_write_json(
            evidence_work / ARTIFACT_FILENAMES["cleaning_summary"],
            {
                "schema_version": 1,
                "source_snapshot": upstream,
                "coverage": coverage,
                "summary": normalized_summary,
                "execution": {
                    "masked_required_rows_are_excluded_from_qlib": True,
                    "factor_semantics": "unit_factor_prices_already_qfq",
                },
            },
        )
        result = convert_to_bin(
            cleaned_parquet,
            staging,
            overwrite=True,
            calendar_path=canonical.snapshot_path / "calendars" / "day.txt",
            source_contract=upstream,
            force_unit_factor=True,
        )
        if result != 0:
            raise DataContractError("cleaned Qlib v2 conversion failed")
        benchmark_evidence = _import_benchmarks(
            canonical.snapshot_path,
            staging,
            root=root,
            upstream=upstream,
            source_metadata=native_benchmark_source(canonical, root=root),
        )
        instrument_pool_evidence = _install_instrument_pools(staging, root=root)
        artifacts = _copy_transformation_artifacts(evidence_work, staging)
        transformation = {
            "name": "anomaly_masking",
            "algorithm": {
                "script": "build_cleaned_qlib_v2.py",
                "version": "2",
            },
            "source_snapshot": upstream,
            "summary": normalized_summary,
            "artifacts": artifacts,
        }
        finalize_qlib_snapshot(
            staging,
            dataset_id=DATASET_ID,
            snapshot_id=snapshot_id,
            source={
                "provider": "quant_workflow",
                "interface": "v2_anomaly_cleaning_to_qlib",
            },
            source_retrieved_at=str(qfq.manifest["source_retrieved_at"]),
            producer={"script": "build_cleaned_qlib_v2.py", "version": "6"},
            upstream_snapshot=upstream,
            benchmark_evidence=benchmark_evidence,
            instrument_pool_evidence=instrument_pool_evidence,
            transformation_evidence=transformation,
            dependency_snapshots=[canonical_dependency],
        )
        if not publish:
            return staging

        def publish_current_build() -> Path:
            _assert_build_inputs_still_current(
                root,
                upstream=upstream,
                canonical_dependency=canonical_dependency,
            )
            return publish_qlib_snapshot(
                staging,
                dataset_root=root / DATASET_ROOT,
                quant_data_root=root,
                update_root_catalog=True,
                publication_lock_held=True,
                validated_cache=validated_cache,
            )

        if publication_lock_held:
            published = publish_current_build()
        else:
            with dataset_publication_lock(root):
                published = publish_current_build()
    except Exception as exc:
        run_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            run_root / "failure.json",
            {
                "status": "failed",
                "failed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    shutil.rmtree(run_root, ignore_errors=True)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-data-root",
        type=Path,
        default=default_quant_data_root(),
    )
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = build_cleaned_qlib_v2(
            quant_data_root=args.quant_data_root,
            publish=args.publish,
        )
    except (DataContractError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(
            f"Cleaned Qlib v2 build failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
