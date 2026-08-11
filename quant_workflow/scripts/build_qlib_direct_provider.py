#!/usr/bin/env python3
"""Build and smoke-test Qlib's metadata-only direct Parquet adapter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DSA_ROOT = SCRIPT_DIR.parents[1]
if str(DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root  # noqa: E402
from quant_workflow.quant_data_qlib_provider import (  # noqa: E402
    REQUIRED_INDEX_SYMBOLS,
    build_direct_provider_descriptor,
    init_qlib_from_quant_data,
)


def _sample_dates(
    descriptor: Path,
    payload: dict,
    quant_data_root: Path,
    symbol: str,
    sample_count: int,
) -> list[str]:
    if symbol in REQUIRED_INDEX_SYMBOLS:
        dataset = payload.get("index_dataset")
    else:
        dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError(f"smoke dataset identity is missing for {symbol}")
    snapshot = quant_data_root / str(dataset["snapshot_path"])
    file_index = pd.read_parquet(
        snapshot / str(dataset["file_index_path"])
    )
    rows = file_index[file_index["symbol"].astype(str) == symbol]
    if rows.empty:
        raise ValueError(f"smoke symbol is missing from its file index: {symbol}")
    row = rows.iloc[0]
    relative_path = str(row.get("relative_path", ""))
    if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError(f"smoke symbol has an invalid file index path: {symbol}")
    bars_path = snapshot / relative_path
    if not bars_path.is_file() or bars_path.is_symlink():
        raise ValueError(f"smoke symbol bar file is missing: {symbol}")
    bars = pd.read_parquet(bars_path, columns=["date"])
    dates = pd.to_datetime(bars["date"], errors="coerce").dropna().sort_values()
    dates = dates[(dates >= pd.Timestamp(row["first_date"])) & (dates <= pd.Timestamp(row["last_date"]))]
    if dates.empty:
        raise ValueError(f"smoke symbol has no dates on the frozen calendar: {symbol}")
    positions = np.linspace(0, len(dates) - 1, num=min(sample_count, len(dates)), dtype=int)
    return [dates.iloc[int(position)].date().isoformat() for position in sorted(set(positions))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Qlib Provider descriptor that reads quant_data Parquet "
            "directly; no features/*.bin copy is generated."
        )
    )
    parser.add_argument("--quant-data-root", type=Path, default=quant_data_root())
    parser.add_argument("--dataset-id", default="ashare_daily_qfq")
    parser.add_argument("--index-dataset-id", default="cn_index_daily")
    parser.add_argument(
        "--cleaning-mode",
        choices=("canonical", "cleaned"),
        default="canonical",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--smoke-symbol", default="SH600000")
    parser.add_argument("--smoke-start")
    parser.add_argument("--smoke-end")
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=5,
        help="Deterministic historical dates per symbol when no explicit range is supplied.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    root = args.quant_data_root.expanduser().resolve()
    descriptor = build_direct_provider_descriptor(
        quant_data_root=root,
        dataset_id=args.dataset_id,
        output_root=args.output_root,
        index_dataset_id=getattr(args, "index_dataset_id", "cn_index_daily"),
        cleaning_mode=getattr(args, "cleaning_mode", "canonical"),
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
    if result["market_data_bin_count"]:
        result["status"] = "invalid_storage"
        result["smoke_test"] = "failed"
        result["error"] = "direct Provider metadata directory contains .bin files"
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 1
    return_code = 0
    if not getattr(args, "skip_smoke", False):
        sample_count = int(getattr(args, "smoke_samples", 5))
        if sample_count < 1:
            raise ValueError("--smoke-samples must be at least 1")
        init_qlib_from_quant_data(
            descriptor_path=descriptor,
            quant_data_root=root,
            kernels=1,
        )
        from qlib.data import D

        stock_symbol = getattr(args, "smoke_symbol", "SH600000")
        explicit_start = getattr(args, "smoke_start", None)
        explicit_end = getattr(args, "smoke_end", None)
        symbols = [stock_symbol, *REQUIRED_INDEX_SYMBOLS]
        checks: list[dict[str, str]] = []
        failures: list[dict[str, str]] = []
        stock_sample_dates: list[str] = []
        for symbol in symbols:
            fields = (
                ["$close", "$volume", "$factor"]
                if symbol == stock_symbol
                else ["$open", "$close"]
            )
            dates = (
                [explicit_start or explicit_end or payload["calendar"]["end"]]
                if explicit_start or explicit_end
                else _sample_dates(
                    descriptor,
                    payload,
                    root,
                    symbol,
                    sample_count,
                )
            )
            if symbol == stock_symbol:
                stock_sample_dates = dates
            for day in dates:
                start = explicit_start or day
                end = explicit_end or day
                frame = D.features(
                    [symbol],
                    fields,
                    start_time=start,
                    end_time=end,
                )
                check = {"symbol": symbol, "start": start, "end": end}
                checks.append(check)
                if frame.empty:
                    failures.append({**check, "reason": "empty_or_all_null"})
                    continue
                missing_columns = [field for field in fields if field not in frame.columns]
                if missing_columns:
                    failures.append(
                        {
                            **check,
                            "reason": "missing_required_columns",
                            "fields": ",".join(missing_columns),
                        }
                    )
                    continue
                required = frame.loc[:, list(fields)]
                usable = required.dropna(how="all")
                if usable.empty:
                    failures.append({**check, "reason": "empty_or_all_null"})
                elif not bool(usable.notna().all().all()):
                    failures.append(
                        {**check, "reason": "missing_or_null_required_field"}
                    )
        result["smoke_sample_count"] = len(stock_sample_dates)
        result["smoke_checks"] = checks
        result["smoke_failures"] = failures
        result["smoke_test"] = "pass" if not failures else "empty"
        result["smoke_symbol"] = stock_symbol
        if failures:
            result["status"] = "smoke_failed"
            return_code = 1
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
