#!/usr/bin/env python3
"""Build and smoke-test Qlib's metadata-only direct Parquet adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DSA_ROOT = SCRIPT_DIR.parents[1]
if str(DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from quant_workflow.quant_data_qlib_provider import (  # noqa: E402
    build_direct_provider_descriptor,
    init_qlib_from_quant_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Qlib Provider descriptor that reads quant_data Parquet "
            "directly; no features/*.bin copy is generated."
        )
    )
    parser.add_argument("--quant-data-root", type=Path, default=quant_data_root())
    parser.add_argument("--dataset-id", default="ashare_daily_qfq")
    parser.add_argument("--membership-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-symbol", default="SH600000")
    parser.add_argument("--smoke-start")
    parser.add_argument("--smoke-end")
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.quant_data_root.expanduser().resolve()
    descriptor = build_direct_provider_descriptor(
        quant_data_root=root,
        dataset_id=args.dataset_id,
        membership_path=args.membership_path,
        output_root=args.output_root,
    )
    payload = json.loads(descriptor.read_text(encoding="utf-8"))
    result = {
        "status": "ready",
        "storage_mode": "direct_parquet",
        "descriptor": descriptor.relative_to(root).as_posix(),
        "dataset_id": payload["dataset"]["dataset_id"],
        "snapshot_id": payload["dataset"]["snapshot_id"],
        "latest_trade_date": payload["dataset"]["latest_trade_date"],
        "market_data_bin_count": len(list(descriptor.parent.rglob("*.bin"))),
        "smoke_test": "skipped",
    }
    return_code = 0
    if not args.skip_smoke:
        start = args.smoke_start or payload["calendar"]["end"]
        end = args.smoke_end or start
        init_qlib_from_quant_data(
            descriptor_path=descriptor,
            quant_data_root=root,
            kernels=1,
        )
        from qlib.data import D

        frame = D.features(
            [args.smoke_symbol],
            ["$close", "$volume", "$factor"],
            start_time=start,
            end_time=end,
        )
        smoke_passed = not frame.empty and bool(frame.notna().any().any())
        result["smoke_test"] = "pass" if smoke_passed else "empty"
        result["smoke_rows"] = int(len(frame))
        result["smoke_symbol"] = args.smoke_symbol
        result["smoke_range"] = {"start": start, "end": end}
        if not smoke_passed:
            result["status"] = "smoke_failed"
            return_code = 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
