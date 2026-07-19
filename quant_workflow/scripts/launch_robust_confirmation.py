#!/usr/bin/env python3
"""Launch the single-use robust confirmation as a durable launchd job."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import mlruns_root, quant_data_root, reports_root  # noqa: E402
from launch_robust_strategy import launch  # noqa: E402
from run_robust_strategy import DEFAULT_CONFIG, RUN_ID_RE  # noqa: E402


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def frozen_requires_mps(frozen: Mapping[str, Any]) -> bool:
    return "mps_dnn" in {str(value) for value in frozen.get("selected_models") or []}


def build_confirmation_command(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    frozen: Path,
    provider_state: Path,
    run_id: str,
    resume: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "run_robust_confirmation.py"),
        "--config",
        str(Path(config).expanduser().resolve()),
        "--quant-data-dir",
        str(Path(quant_data_dir).expanduser().resolve()),
        "--report-dir",
        str(Path(report_dir).expanduser().resolve()),
        "--frozen",
        str(Path(frozen).expanduser().resolve()),
        "--provider-state",
        str(Path(provider_state).expanduser().resolve()),
        "--run-id",
        run_id,
    ]
    if resume:
        command.append("--resume")
    return command


def launch_confirmation(
    *,
    config: Path,
    quant_data_dir: Path,
    report_dir: Path,
    mlruns_dir: Path,
    frozen: Path,
    provider_state: Path,
    run_id: str,
    resume: bool,
    warmup_seconds: float = 10.0,
    **launch_kwargs: Any,
) -> dict[str, Any]:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    frozen_payload = _read_json(Path(frozen).expanduser().resolve())
    if frozen_payload.get("status") != "research_candidate_ready":
        raise ValueError("a frozen research candidate is required")
    command = build_confirmation_command(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        frozen=frozen,
        provider_state=provider_state,
        run_id=run_id,
        resume=resume,
    )
    return launch(
        config=config,
        quant_data_dir=quant_data_dir,
        report_dir=report_dir,
        mlruns_dir=mlruns_dir,
        as_of_date=str(frozen_payload.get("as_of_date") or "2025-12-31"),
        run_id=run_id,
        resume=resume,
        skip_membership_build=True,
        warmup_seconds=warmup_seconds,
        runner_namespace="confirmation_runner",
        command_override=command,
        mps_required_override=frozen_requires_mps(frozen_payload),
        update_pipeline_state_on_reconcile=False,
        **launch_kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the frozen 2026 confirmation as a durable job."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--quant-data-dir", type=Path, default=quant_data_root())
    parser.add_argument(
        "--report-dir", type=Path, default=reports_root() / "robust_strategy"
    )
    parser.add_argument("--mlruns-dir", type=Path, default=mlruns_root())
    parser.add_argument(
        "--frozen",
        type=Path,
        default=reports_root() / "robust_strategy" / "frozen_strategy_latest.json",
    )
    parser.add_argument(
        "--provider-state",
        type=Path,
        default=reports_root() / "robust_strategy" / "provider_latest.json",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=10.0)
    args = parser.parse_args(argv)
    run_id = args.run_id or datetime.now().strftime("confirmation_%Y%m%d_%H%M%S")
    result = launch_confirmation(
        config=args.config,
        quant_data_dir=args.quant_data_dir,
        report_dir=args.report_dir,
        mlruns_dir=args.mlruns_dir,
        frozen=args.frozen,
        provider_state=args.provider_state,
        run_id=run_id,
        resume=args.resume,
        warmup_seconds=args.warmup_seconds,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
