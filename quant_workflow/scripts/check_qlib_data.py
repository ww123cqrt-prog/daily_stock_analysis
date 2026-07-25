from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root  # noqa: E402


QUANT_DATA = quant_data_root()
PROVIDER_URI = qlib_provider_uri()


def read_provider_calendar_range(provider: Path) -> tuple[str, str]:
    calendar_path = provider / "calendars" / "day.txt"
    days = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not days:
        raise ValueError(f"empty Qlib calendar: {calendar_path}")
    return days[0], days[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check a Qlib provider.")
    parser.add_argument("--provider-dir", type=Path, default=PROVIDER_URI)
    return parser.parse_args(argv)


def read_instrument_sample(provider: Path, limit: int = 2) -> list[str]:
    path = provider / "instruments" / "all.txt"
    symbols = [
        line.split("\t", 1)[0].strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not symbols:
        raise ValueError(f"empty Qlib instruments: {path}")
    return symbols[:limit]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    multiprocessing.set_start_method("fork", force=True)

    import qlib
    from qlib.data import D

    start_date, end_date = read_provider_calendar_range(args.provider_dir)
    qlib.init(provider_uri=str(args.provider_dir), region="cn")

    calendar = D.calendar(start_time=start_date, end_time=end_date, freq="day")
    if len(calendar) < 1:
        raise ValueError(f"Qlib returned an empty calendar: {args.provider_dir}")
    print("calendar", len(calendar), calendar[0], calendar[-1])

    sample_start = str(calendar[-min(5, len(calendar))].date())
    sample_end = str(calendar[-1].date())
    instruments = read_instrument_sample(args.provider_dir)
    features = D.features(
        instruments,
        ["$close", "$volume"],
        start_time=sample_start,
        end_time=sample_end,
        freq="day",
        disk_cache=0,
    )
    print(features)
    if features.empty:
        raise ValueError(f"Qlib returned no sample features: {args.provider_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
