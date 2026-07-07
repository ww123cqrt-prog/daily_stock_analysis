from __future__ import annotations

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

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    import qlib
    from qlib.data import D

    start_date, end_date = read_provider_calendar_range(PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region="cn")

    calendar = D.calendar(start_time=start_date, end_time=end_date, freq="day")
    print("calendar", len(calendar), calendar[0], calendar[-1])

    sample_start = str(calendar[-5].date())
    sample_end = str(calendar[-1].date())
    features = D.features(
        ["SZ000001", "SH600519"],
        ["$close", "$volume"],
        start_time=sample_start,
        end_time=sample_end,
        freq="day",
        disk_cache=0,
    )
    print(features)
