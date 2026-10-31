# -*- coding: utf-8 -*-
"""Static checks and deterministic execution for the DSA Python strategy API."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import date
import math
import statistics
from types import MappingProxyType
from typing import Any, Sequence

import numpy as np

from src.core.script_market import MarketView
from src.core.strategy_backtest import PriceBar, TargetAllocation, StrategyBacktestConfig, run_portfolio_backtest_view


SCRIPT_API_VERSION = "dsa-script-v1"
MAX_SCRIPT_BYTES = 50_000
MAX_AST_NODES = 5_000
_FORBIDDEN_CALLS = {
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "hasattr",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "delattr",
    "vars",
    "__import__",
}
_FORBIDDEN_NODES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Nonlocal,
    ast.With,
)


class _SafeStatistics:
    """Whitelisted statistics facade.

    The real ``statistics`` module imports ``sys`` (``statistics.sys``), which
    would let a script walk ``sys.modules['builtins'].open`` and read files
    outside the sandbox. Only the public aggregation functions are exposed.
    """

    mean = staticmethod(statistics.mean)
    median = staticmethod(statistics.median)
    median_low = staticmethod(statistics.median_low)
    median_high = staticmethod(statistics.median_high)
    median_grouped = staticmethod(statistics.median_grouped)
    mode = staticmethod(statistics.mode)
    stdev = staticmethod(statistics.stdev)
    pstdev = staticmethod(statistics.pstdev)
    variance = staticmethod(statistics.variance)
    pvariance = staticmethod(statistics.pvariance)
    fmean = staticmethod(statistics.fmean)


class ScriptValidationError(ValueError):
    pass


class ScriptExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class StrategyScriptResult:
    allocations: tuple[TargetAllocation, ...]
    logs: tuple[str, ...]
    check: dict[str, Any]
    backtest_report: dict[str, Any] | None = None


def validate_strategy_script(source: str) -> dict[str, Any]:
    encoded = str(source or "").encode("utf-8")
    if not encoded:
        raise ScriptValidationError("script must not be empty")
    if len(encoded) > MAX_SCRIPT_BYTES:
        raise ScriptValidationError(f"script must not exceed {MAX_SCRIPT_BYTES} bytes")
    try:
        tree = ast.parse(source, filename="strategy.py", mode="exec")
    except SyntaxError as exc:
        raise ScriptValidationError(f"syntax error at line {exc.lineno}: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise ScriptValidationError("script is too complex")

    function_names = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "on_bar" not in function_names:
        raise ScriptValidationError("script must define on_bar(context, data)")
    for node in nodes:
        if isinstance(node, _FORBIDDEN_NODES):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                raise ScriptValidationError("imports are not allowed; math and statistics are provided")
            raise ScriptValidationError(f"{type(node).__name__} is not allowed")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ScriptValidationError("private attributes are not allowed")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ScriptValidationError("private names are not allowed")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
            raise ScriptValidationError(f"{node.func.id} is not allowed")
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef):
            continue
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
            continue
        raise ScriptValidationError("top-level code is not allowed; define initialize and on_bar functions")
    return {
        **({"fill_feedback": True} if "on_fill" in function_names else {}),
        "status": "passed",
        "api_version": SCRIPT_API_VERSION,
        "ast_nodes": len(nodes),
        "script_bytes": len(encoded),
    }


def _normalize_history(
    bars: Sequence[PriceBar],
    symbols: Sequence[str],
) -> tuple[tuple[str, ...], list[date], dict[str, dict[date, PriceBar]]]:
    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not normalized_symbols or len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("symbols must be non-empty and unique")
    by_symbol: dict[str, dict[date, PriceBar]] = {symbol: {} for symbol in normalized_symbols}
    for bar in bars:
        symbol = str(bar.symbol).strip().upper()
        if symbol in by_symbol:
            by_symbol[symbol][bar.date] = bar
    if any(not by_symbol[symbol] for symbol in normalized_symbols):
        raise ValueError("every script symbol requires price bars")
    common = set(by_symbol[normalized_symbols[0]])
    for symbol in normalized_symbols[1:]:
        common &= set(by_symbol[symbol])
    calendar = sorted(common)
    if not calendar:
        raise ValueError("script symbols have no common trading dates")
    return normalized_symbols, calendar, by_symbol


class _StrategyData:
    """Script-facing data API backed by a vectorized MarketView."""

    def __init__(self, view: MarketView) -> None:
        self._view = view
        self.symbols = tuple(str(symbol) for symbol in view.symbols)
        self._symbol_set = set(self.symbols)
        self._index = 0
        # PIT pools (csi800/all_a) enforce that a script can only touch
        # symbols that are members on the current date — otherwise a script
        # could buy a stock before it enters the pool (look-ahead bias).
        self._pit_guard = str(view.meta.get("pool") or "") in ("csi800", "all_a")

    @property
    def current_date(self) -> date:
        return date.fromordinal(int(self._view.dates[self._index]))

    def _symbol(self, symbol: str) -> str:
        normalized = str(symbol).strip().upper()
        if normalized not in self._symbol_set:
            raise ValueError(f"symbol is outside the requested universe: {symbol}")
        if self._pit_guard and not bool(
            self._view.members[self._index, self._view.symbol_index(normalized)]
        ):
            raise ValueError(
                f"{normalized} is not a tradable member on this date"
            )
        return normalized

    def _history_value(self, symbol: str, field: str, ago: int) -> float:
        normalized = self._symbol(symbol)
        if isinstance(ago, bool) or not isinstance(ago, int) or ago < 0:
            raise ValueError("ago must be non-negative")
        target = self._index - ago
        if target < 0:
            raise ValueError("requested history is not available yet")
        symbol_index = self._view.symbol_index(normalized)
        matrix = self._view.open if field == "open" else self._view.close
        value = matrix[target, symbol_index]
        if bool(np.isnan(value)):
            raise ValueError(f"{field} is not available for {normalized} on this date")
        return float(value)

    def close(self, symbol: str, ago: int = 0) -> float:
        return self._history_value(symbol, "close", ago)

    def open(self, symbol: str, ago: int = 0) -> float:
        return self._history_value(symbol, "open", ago)

    def volume(self, symbol: str, ago: int = 0) -> float:
        normalized = self._symbol(symbol)
        if isinstance(ago, bool) or not isinstance(ago, int) or ago < 0:
            raise ValueError("ago must be non-negative")
        target = self._index - ago
        if target < 0:
            raise ValueError("requested history is not available yet")
        value = self._view.volume[target, self._view.symbol_index(normalized)]
        if bool(np.isnan(value)):
            raise ValueError(f"volume is not available for {normalized} on this date")
        return float(value)

    def index_close(self, ago: int = 0) -> float:
        if isinstance(ago, bool) or not isinstance(ago, int) or ago < 0:
            raise ValueError("ago must be non-negative")
        target = self._index - ago
        if target < 0:
            raise ValueError("requested history is not available yet")
        value = self._view.index_close[target]
        if bool(np.isnan(value)):
            raise ValueError("benchmark index close is not available on this date")
        return float(value)

    def index_history(self, count: int) -> tuple[float, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("history count must be a positive integer")
        start = self._index - count + 1
        if start < 0:
            return ()
        values = self._view.index_close[start:self._index + 1]
        values = values[~np.isnan(values)]
        return tuple(float(value) for value in values)

    def history(self, symbol: str, count: int, field: str = "close") -> tuple[float, ...]:
        if field not in {"open", "close", "volume"}:
            raise ValueError("history field must be open, close, or volume")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("history count must be a positive integer")
        start = self._index - count + 1
        if start < 0:
            return ()
        normalized = self._symbol(symbol)
        symbol_index = self._view.symbol_index(normalized)
        matrix = self._view.open if field == "open" else self._view.close if field == "close" else self._view.volume
        values = matrix[start:self._index + 1, symbol_index]
        values = values[~np.isnan(values)]
        return tuple(float(value) for value in values)

    def returns(self, symbol: str, days: int) -> float | None:
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise ValueError("return days must be a positive integer")
        if self._index < days:
            return None
        current = self.close(symbol)
        previous = self.close(symbol, ago=days)
        if previous == 0:
            return None
        return current / previous - 1.0

    def sma(self, symbol: str, days: int) -> float | None:
        """Simple moving average over the last ``days`` valid closes.

        Returns ``None`` when the window does not contain ``days`` valid
        closes yet — a suspended or newly listed symbol must not produce a
        "20-day average" out of three surviving observations.
        """
        values = self.history(symbol, days, "close")
        if len(values) < days:
            return None
        return sum(values) / len(values)

    def rank_by_return(self, days: int, descending: bool = True) -> list[str]:
        """Rank today's tradable universe members by trailing return.

        Only members eligible on the current date participate (PIT), and
        symbols without a valid previous close are excluded.
        """
        if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
            raise ValueError("return days must be a positive integer")
        if self._index < days:
            return []
        member_indices = np.flatnonzero(self._view.members[self._index])
        if len(member_indices) == 0:
            return []
        previous = self._view.close[self._index - days, member_indices]
        current = self._view.close[self._index, member_indices]
        valid = ~np.isnan(previous) & ~np.isnan(current) & (previous != 0.0)
        returns = np.full(len(member_indices), np.nan, dtype=np.float64)
        returns[valid] = current[valid] / previous[valid] - 1.0
        valid_indices = np.flatnonzero(valid)
        ordered = valid_indices[np.argsort(
            -returns[valid_indices] if descending else returns[valid_indices],
            kind="stable",
        )]
        return [str(self._view.symbols[member_indices[pos]]) for pos in ordered]

    def universe(self) -> list[str]:
        """Tradable universe members on the current date (PIT)."""
        return [
            str(self._view.symbols[index])
            for index in np.flatnonzero(self._view.members[self._index])
        ]


class _StrategyContext:
    def __init__(self, symbols: tuple[str, ...], logs: list[str]) -> None:
        self.symbols = symbols
        self.current_date: date | None = None
        self.current_weights = MappingProxyType({})
        self._target: dict[str, float] = {}
        self._touched = False
        self._logs = logs
        self._tradable: set[str] | None = None

    def _begin_bar(
        self,
        current_date: date,
        current_weights: dict[str, float],
        tradable: set[str] | None = None,
    ) -> None:
        self.current_date = current_date
        self.current_weights = MappingProxyType(dict(current_weights))
        self._target = dict(current_weights)
        self._touched = False
        # PIT pools pass the day's member set; targets must stay inside it.
        self._tradable = tradable

    def _require_tradable(self, symbol: str) -> None:
        if self._tradable is not None and symbol not in self._tradable:
            raise ValueError(f"{symbol} is not a tradable member on this date")

    def target_percent(self, symbol: str, weight: float) -> None:
        normalized = str(symbol).strip().upper()
        if normalized not in self.symbols:
            raise ValueError(f"symbol is outside the requested universe: {symbol}")
        self._require_tradable(normalized)
        numeric_weight = float(weight)
        if not math.isfinite(numeric_weight) or numeric_weight < 0 or numeric_weight > 1:
            raise ValueError("target weight must be between 0 and 1")
        if numeric_weight == 0:
            self._target.pop(normalized, None)
        else:
            self._target[normalized] = numeric_weight
        if sum(self._target.values()) > 1.0 + 1e-9:
            raise ValueError("target weights must sum to at most 1")
        self._touched = True

    def target_equal(self, symbols: Sequence[str]) -> None:
        selected = tuple(str(symbol).strip().upper() for symbol in symbols)
        if not selected:
            self.target_cash()
            return
        if len(set(selected)) != len(selected) or any(symbol not in self.symbols for symbol in selected):
            raise ValueError("target_equal symbols must be unique members of the requested universe")
        for symbol in selected:
            self._require_tradable(symbol)
        weight = 1.0 / len(selected)
        self._target = {symbol: weight for symbol in selected}
        self._touched = True

    def target_cash(self) -> None:
        self._target = {}
        self._touched = True

    def log(self, message: object) -> None:
        prefix = self.current_date.isoformat() if self.current_date else "initialize"
        self._logs.append(f"{prefix} {str(message)[:500]}")


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


def view_from_bars(bars: Sequence[PriceBar], symbols: Sequence[str]) -> MarketView:
    """Build a vectorized view from legacy PriceBar input (small universes only).

    Used by the compatibility path for callers that already hold in-memory
    bars; large universes must use a prebuilt MarketBundle.
    """
    normalized_symbols = tuple(str(symbol).strip().upper() for symbol in symbols)
    if not normalized_symbols or len(set(normalized_symbols)) != len(normalized_symbols):
        raise ValueError("symbols must be non-empty and unique")
    symbol_index = {symbol: i for i, symbol in enumerate(normalized_symbols)}
    dates_by_bar = sorted({bar.date for bar in bars})
    dates = np.array([value.toordinal() for value in dates_by_bar], dtype=np.int32)
    open_matrix = np.full((len(dates), len(normalized_symbols)), np.nan, dtype=np.float64)
    close_matrix = np.full_like(open_matrix, np.nan)
    members = np.zeros_like(open_matrix, dtype=bool)
    date_index = {value: i for i, value in enumerate(dates_by_bar)}
    for bar in bars:
        symbol = str(bar.symbol).strip().upper()
        if symbol not in symbol_index:
            continue
        row = date_index[bar.date]
        col = symbol_index[symbol]
        open_matrix[row, col] = float(bar.open)
        close_matrix[row, col] = float(bar.close)
        members[row, col] = True
    if any(not members[:, col].any() for col in range(len(normalized_symbols))):
        raise ValueError("every script symbol requires price bars")
    return MarketView(
        dates=dates,
        symbols=np.array(normalized_symbols, dtype=object),
        open_matrix=open_matrix,
        close_matrix=close_matrix,
        members=members,
        meta={},
    )


def execute_strategy_script(
    source: str,
    *,
    view: MarketView | None = None,
    bars: Sequence[PriceBar] | None = None,
    symbols: Sequence[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    execution_config: StrategyBacktestConfig | None = None,
) -> StrategyScriptResult:
    check = validate_strategy_script(source)
    if view is None:
        if bars is None or symbols is None:
            raise ValueError("either a market view or bars+symbols are required")
        view = view_from_bars(bars, symbols)
    normalized_symbols = tuple(str(symbol) for symbol in view.symbols)
    calendar = view.calendar
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "math": math,
        "statistics": _SafeStatistics(),
    }
    try:
        exec(compile(source, "strategy.py", "exec"), namespace, namespace)
        initialize = namespace.get("initialize")
        on_bar = namespace["on_bar"]
        if initialize is not None and not callable(initialize):
            raise ScriptExecutionError("initialize must be callable")
        if not callable(on_bar):
            raise ScriptExecutionError("on_bar must be callable")
        logs: list[str] = []
        context = _StrategyContext(normalized_symbols, logs)
        data = _StrategyData(view)
        if initialize is not None:
            initialize(context)

        on_fill = namespace.get("on_fill")
        if on_fill is not None and not callable(on_fill):
            raise ScriptExecutionError("on_fill must be callable")
        if on_fill is not None and execution_config is None:
            raise ScriptExecutionError("on_fill requires execution_config and actual trade feedback")
        if execution_config is not None:
            allocations: list[TargetAllocation] = []

            def filled(trade, positions, cash):
                context.current_date = date.fromisoformat(trade["execution_date"])
                context.positions = MappingProxyType({s: n for s, n in positions.items() if n})
                context.cash = cash
                if on_fill is not None:
                    on_fill(context, MappingProxyType(dict(trade)))

            def closed(index, positions, cash, weights):
                current_date = calendar[index]
                data._index = index
                context.positions = MappingProxyType({s: n for s, n in positions.items() if n})
                context.cash = cash
                tradable = set(view.members_on(index)) if str(view.meta.get("pool") or "") in ("csi800", "all_a") else None
                context._begin_bar(current_date, dict(weights), tradable)
                on_bar(context, data)
                if context._touched and context._target != weights:
                    allocation = TargetAllocation(
                        source_id=f"python_script:{current_date.isoformat()}", signal_date=current_date,
                        target_weights=dict(sorted(context._target.items())),
                        reason=f"DSA Python strategy target on {current_date.isoformat()}")
                    allocations.append(allocation)
                    return allocation
                return None

            report = run_portfolio_backtest_view(strategy_name="python_script", allocations=[],
                view=view, config=execution_config, period_start=start_date, period_end=end_date,
                signal_callback=closed, fill_callback=filled)
            report["assumptions"]["execution_feedback"] = "actual_fills"
            return StrategyScriptResult(tuple(allocations), tuple(logs[-500:]), check, report)

        allocations: list[TargetAllocation] = []
        current_target: dict[str, float] = {}
        pit_pool = str(view.meta.get("pool") or "") in ("csi800", "all_a")
        for index, current_date in enumerate(calendar):
            if start_date is not None and current_date < start_date:
                continue
            if end_date is not None and current_date > end_date:
                break
            data._index = index
            tradable = set(view.members_on(index)) if pit_pool else None
            context._begin_bar(current_date, current_target, tradable)
            on_bar(context, data)
            if context._touched and context._target != current_target:
                current_target = dict(sorted(context._target.items()))
                allocations.append(
                    TargetAllocation(
                        source_id=f"python_script:{current_date.isoformat()}",
                        signal_date=current_date,
                        target_weights=current_target,
                        reason=f"DSA Python strategy target on {current_date.isoformat()}",
                    )
                )
        return StrategyScriptResult(
            allocations=tuple(allocations),
            logs=tuple(logs[-500:]),
            check=check,
        )
    except ScriptValidationError:
        raise
    except ScriptExecutionError:
        raise
    except Exception as exc:
        raise ScriptExecutionError(f"script failed: {type(exc).__name__}: {exc}") from exc
