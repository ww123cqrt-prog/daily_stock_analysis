from __future__ import annotations

import argparse
import multiprocessing
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri  # noqa: E402


REQUIRED_INSTRUMENTS = (
    "SH600000",
    "SH000300",
    "SH000905",
    "SH000906",
    "SZ399006",
    "SH000688",
)
REQUIRED_FIELDS = ("$open", "$high", "$low", "$close", "$volume", "$factor")


def read_provider_calendar_range(provider: Path) -> tuple[str, str]:
    calendar_path = provider / "calendars" / "day.txt"
    days = [line.strip() for line in calendar_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not days:
        raise ValueError(f"empty Qlib calendar: {calendar_path}")
    return days[0], days[-1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check a Qlib provider.")
    parser.add_argument(
        "--provider",
        choices=("canonical", "cleaned"),
        default="canonical",
    )
    parser.add_argument("--provider-dir", type=Path)
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


def validate_feature_sample(features, sample_days: list[str]) -> None:
    import numpy as np
    import pandas as pd

    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("Qlib returned no required sample features")
    if not isinstance(features.index, pd.MultiIndex) or not {
        "instrument",
        "datetime",
    }.issubset(features.index.names):
        raise ValueError("Qlib sample index is invalid")
    if list(features.columns) != list(REQUIRED_FIELDS):
        raise ValueError("Qlib sample fields are incomplete")
    expected_days = set(sample_days)
    for instrument in REQUIRED_INSTRUMENTS:
        try:
            frame = features.xs(instrument, level="instrument")
        except KeyError as exc:
            raise ValueError(f"Qlib sample is missing {instrument}") from exc
        actual_days = {
            str(pd.Timestamp(value).date()) for value in frame.index.tolist()
        }
        missing_days = sorted(expected_days.difference(actual_days))
        if missing_days:
            raise ValueError(
                f"Qlib sample is missing dates for {instrument}: "
                + ", ".join(missing_days)
            )
        required = frame.loc[
            [str(pd.Timestamp(value).date()) in expected_days for value in frame.index]
        ]
        values = required.loc[:, list(REQUIRED_FIELDS)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Qlib sample contains non-finite values for {instrument}")
        if (required.loc[:, ["$open", "$high", "$low", "$close", "$factor"]] <= 0).any(
            axis=None
        ) or (required["$volume"] < 0).any():
            raise ValueError(f"Qlib sample contains invalid values for {instrument}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    provider_dir = args.provider_dir or qlib_provider_uri(args.provider)
    multiprocessing.set_start_method("fork", force=True)

    import qlib
    from qlib.data import D

    start_date, end_date = read_provider_calendar_range(provider_dir)
    qlib.init(provider_uri=str(provider_dir), region="cn")

    calendar = D.calendar(start_time=start_date, end_time=end_date, freq="day")
    if len(calendar) < 1:
        raise ValueError(f"Qlib returned an empty calendar: {provider_dir}")
    print("calendar", len(calendar), calendar[0], calendar[-1])

    sample_start = str(calendar[-min(5, len(calendar))].date())
    sample_end = str(calendar[-1].date())
    sample_days = [str(value.date()) for value in calendar[-min(5, len(calendar)) :]]
    features = D.features(
        list(REQUIRED_INSTRUMENTS),
        list(REQUIRED_FIELDS),
        start_time=sample_start,
        end_time=sample_end,
        freq="day",
        disk_cache=0,
    )
    validate_feature_sample(features, sample_days)
    print(features)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
