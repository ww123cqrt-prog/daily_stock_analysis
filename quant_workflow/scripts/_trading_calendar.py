"""Read-only A-share trading-day calendar helpers."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from _paths import provider_calendar_days, qlib_workflow_provider_uri


QLIB_CALENDAR_SOURCE = "qlib_day_calendar"
WEEKDAY_FALLBACK_SOURCE = "weekday_only"


@dataclass(frozen=True)
class TradingDayDecision:
    day: date
    source: str


@dataclass(frozen=True)
class TradingCalendar:
    days: tuple[date, ...]
    path: Path

    def on_or_previous(self, value: date) -> TradingDayDecision:
        if self._covers(value):
            index = bisect_right(self.days, value) - 1
            return TradingDayDecision(self.days[index], QLIB_CALENDAR_SOURCE)
        return TradingDayDecision(_weekday_on_or_previous(value), WEEKDAY_FALLBACK_SOURCE)

    def previous(self, value: date) -> TradingDayDecision:
        if self._covers(value) and value > self.days[0]:
            index = bisect_left(self.days, value) - 1
            if index >= 0:
                return TradingDayDecision(self.days[index], QLIB_CALENDAR_SOURCE)
        return TradingDayDecision(_previous_weekday(value), WEEKDAY_FALLBACK_SOURCE)

    def next(self, value: date) -> TradingDayDecision:
        if self.days and self.days[0] <= value < self.days[-1]:
            index = bisect_right(self.days, value)
            if index < len(self.days):
                return TradingDayDecision(self.days[index], QLIB_CALENDAR_SOURCE)
        return TradingDayDecision(_next_weekday(value), WEEKDAY_FALLBACK_SOURCE)

    def _covers(self, value: date) -> bool:
        return bool(self.days and self.days[0] <= value <= self.days[-1])


def load_trading_calendar(path: Path | None = None) -> TradingCalendar:
    days: list[date] = []
    try:
        provider_path = Path(path) if path is not None else qlib_workflow_provider_uri()
        calendar_path = provider_path
        if (
            provider_path.name != "PROVIDER_DESCRIPTOR.json"
            and (provider_path.is_file() or (path is not None and provider_path.suffix))
        ):
            raw_days = (
                provider_path.read_text(encoding="utf-8").splitlines()
                if provider_path.is_file()
                else []
            )
        else:
            raw_days = provider_calendar_days(provider_path)
    except (FileNotFoundError, OSError, RuntimeError, UnicodeError, ValueError):
        calendar_path = Path(path) if path is not None else Path()
        raw_days = []
    for raw in raw_days:
        try:
            days.append(date.fromisoformat(raw))
        except ValueError:
            continue
    return TradingCalendar(tuple(sorted(set(days))), calendar_path)


def _is_weekday(value: date) -> bool:
    return value.weekday() < 5


def _weekday_on_or_previous(value: date) -> date:
    current = value
    while not _is_weekday(current):
        current -= timedelta(days=1)
    return current


def _previous_weekday(value: date) -> date:
    return _weekday_on_or_previous(value - timedelta(days=1))


def _next_weekday(value: date) -> date:
    current = value + timedelta(days=1)
    while not _is_weekday(current):
        current += timedelta(days=1)
    return current
