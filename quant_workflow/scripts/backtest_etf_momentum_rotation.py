"""Backward-compatible entry point for the canonical ETF strategy report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402


DEFAULT_DATA_DIR = (
    quant_data_root() / "parquet" / "etf_1d" / "cn_etf_daily"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_momentum_rotation"


def run(
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
    side_cost_bps: float = 7.5,
) -> Path:
    """Generate the unified report; ``side_cost_bps`` is retained for CLI compatibility."""

    del side_cost_bps
    from generate_etf_momentum_strategy_report import run as generate_report

    return generate_report(data_dir=Path(data_dir), report_dir=Path(report_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the canonical ETF momentum report. The legacy "
            "--side-cost-bps option is accepted but the canonical report always "
            "uses its registered 0/10/20/40bp scenarios."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--side-cost-bps", type=float, default=7.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        side_cost_bps=args.side_cost_bps,
    )
    print(f"ETF momentum strategy report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
