#!/usr/bin/env python3
"""Launch the Goal 1 historical fundamental build through launchd."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, quant_data_root, reports_root  # noqa: E402
from launch_robust_strategy import launch  # noqa: E402
from run_robust_signal_research import DEFAULT_CONFIG  # noqa: E402


def new_run_id() -> str:
    return f"robust_fundamentals_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def build_command(
    *,
    quant_data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    report_dir: Path,
    run_id: str,
) -> list[str]:
    status_path = report_dir / "runs" / run_id / "status.json"
    return [
        sys.executable,
        str(SCRIPT_DIR / "download_robust_fundamentals.py"),
        "--quant-data-dir",
        str(quant_data_dir.resolve()),
        "--cache-dir",
        str(cache_dir.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--data-start",
        "2014-01-02",
        "--data-end",
        "2025-12-31",
        "--report-start",
        "2010-01-01",
        "--report-end",
        "2025-12-31",
        "--coverage-minimum",
        "0.98",
        "--status-path",
        str(status_path.resolve()),
    ]


def launch_fundamental_download(
    *,
    config: Path,
    quant_data_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    report_dir: Path,
    mlruns_dir: Path,
    run_id: str,
    warmup_seconds: float,
) -> dict[str, Any]:
    command = build_command(
        quant_data_dir=quant_data_dir,
        cache_dir=cache_dir,
        output_dir=output_dir,
        report_dir=report_dir,
        run_id=run_id,
    )
    return launch(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        mlruns_dir=mlruns_dir,
        as_of_date=date.today().isoformat(),
        run_id=run_id,
        resume=False,
        skip_membership_build=True,
        warmup_seconds=warmup_seconds,
        runner_namespace="fundamental_runner",
        command_override=command,
        mps_required_override=False,
        update_pipeline_state_on_reconcile=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    quant_data = quant_data_root().expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=quant_data / "cache" / "robust_fundamentals" / "v1",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=quant_data / "reference" / "fundamentals"
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=reports_root() / "robust_signal" / "fundamentals",
    )
    parser.add_argument("--mlruns-dir", type=Path, default=mlruns_root())
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    result = launch_fundamental_download(
        config=args.config.expanduser().resolve(),
        quant_data_dir=args.quant_data_dir.expanduser().resolve(),
        cache_dir=args.cache_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        report_dir=args.report_dir.expanduser().resolve(),
        mlruns_dir=args.mlruns_dir.expanduser().resolve(),
        run_id=args.run_id or new_run_id(),
        warmup_seconds=args.warmup_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
