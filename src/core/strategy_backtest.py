# -*- coding: utf-8 -*-
"""Asset-agnostic target-allocation backtest engine.

Strategies produce target weights after a trading day's information is known.
The engine executes a changed target at the next common trading-day open and
tracks cash, board lots, costs, equity, drawdown, and a buy-and-hold benchmark.
It intentionally knows nothing about whether a symbol is a stock or an ETF.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence

import numpy as np


CASH = "CASH"


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    date: date
    open: float
    close: float


@dataclass(frozen=True)
class TargetAllocation:
    source_id: str
    signal_date: date
    target_weights: Mapping[str, float]
    reason: str | None = None


@dataclass(frozen=True)
class StrategyBacktestConfig:
    initial_capital: float = 1_000_000.0
    side_cost_pct: float = 0.0
    lot_size: int = 100

    def __post_init__(self) -> None:
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not math.isfinite(self.side_cost_pct) or self.side_cost_pct < 0:
            raise ValueError("side_cost_pct must be non-negative")
        if self.side_cost_pct > 5:
            raise ValueError("side_cost_pct must not exceed 5 percent")
        if (
            isinstance(self.lot_size, bool)
            or not isinstance(self.lot_size, int)
            or self.lot_size <= 0
        ):
            raise ValueError("lot_size must be a positive integer")


def _normalized_weights(
    weights: Mapping[str, float],
    *,
    available_symbols: set[str],
) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_symbol, raw_weight in weights.items():
        symbol = str(raw_symbol).strip().upper()
        weight = float(raw_weight)
        if not symbol:
            raise ValueError("target weight symbol must not be empty")
        if symbol not in available_symbols:
            raise ValueError(f"target symbol has no price bars: {symbol}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("target weights must be finite and non-negative")
        if weight > 0:
            normalized[symbol] = weight
    if sum(normalized.values()) > 1.0 + 1e-9:
        raise ValueError("target weights must sum to at most 1")
    return dict(sorted(normalized.items()))


def _normalize_bars(
    bars: Sequence[PriceBar],
) -> tuple[tuple[str, ...], list[date], dict[date, dict[str, PriceBar]]]:
    if not bars:
        raise ValueError("price bars must not be empty")

    by_symbol: dict[str, dict[date, PriceBar]] = {}
    for raw_bar in bars:
        symbol = str(raw_bar.symbol).strip().upper()
        open_price = float(raw_bar.open)
        close = float(raw_bar.close)
        if not symbol:
            raise ValueError("price bar symbol must not be empty")
        if (
            not math.isfinite(open_price)
            or not math.isfinite(close)
            or open_price <= 0
            or close <= 0
        ):
            raise ValueError("price bars require positive open and close values")
        symbol_bars = by_symbol.setdefault(symbol, {})
        if raw_bar.date in symbol_bars:
            raise ValueError(f"duplicate price bar: {symbol} {raw_bar.date.isoformat()}")
        symbol_bars[raw_bar.date] = PriceBar(
            symbol=symbol,
            date=raw_bar.date,
            open=open_price,
            close=close,
        )

    symbols = tuple(sorted(by_symbol))
    common_dates = set(by_symbol[symbols[0]])
    for symbol in symbols[1:]:
        common_dates &= set(by_symbol[symbol])
    calendar = sorted(common_dates)
    if not calendar:
        raise ValueError("symbols have no common trading dates")

    bars_by_date = {
        trading_date: {
            symbol: by_symbol[symbol][trading_date]
            for symbol in symbols
        }
        for trading_date in calendar
    }
    return symbols, calendar, bars_by_date


def _max_drawdown_pct(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    peak = float(values[0])
    worst = 0.0
    for value in values:
        peak = max(peak, float(value))
        drawdown = float(value) / peak - 1.0
        worst = min(worst, drawdown)
    return worst * 100.0


def _annualized_return_pct(
    *,
    initial_capital: float,
    final_equity: float,
    trading_days: int,
) -> float:
    if trading_days <= 0:
        return 0.0
    if initial_capital <= 0:
        return -100.0
    ratio = final_equity / initial_capital
    if ratio <= 0:
        return -100.0
    return (ratio ** (252.0 / trading_days) - 1.0) * 100.0


def _log_return(current: float, previous: float) -> float:
    """Day-over-day log return with an explicit total-loss floor.

    ``0.0`` is never a real return: a day where equity hits zero is a total
    loss (-100%), so its log return is floored at ``log(1e-12)`` (~-27.6)
    instead of being silently recorded as a zero-return day (which would
    understate risk). Days after the equity is already zero carry no new
    information and contribute 0.
    """
    if previous > 0 and current > 0:
        return math.log(current / previous)
    if previous > 0 and current <= 0:
        return math.log(1e-12)
    return 0.0


def _risk_metrics(
    strategy_values: Sequence[float],
    benchmark_values: Sequence[float],
    *,
    opening_equity: float,
    benchmark_opening_equity: float,
) -> dict[str, float | None]:
    """Risk metrics on day-over-day log returns (robust near zero equity).

    The return series starts at the window's opening equity, so the first
    simulated day's intraday return is included; a pre-window position's
    carry-in P&L never leaks into the window's risk numbers.
    """
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    previous_strategy = float(opening_equity)
    previous_benchmark = float(benchmark_opening_equity)
    for strategy_value, benchmark_value in zip(
        strategy_values, benchmark_values, strict=True
    ):
        strategy_returns.append(
            _log_return(float(strategy_value), previous_strategy)
        )
        benchmark_returns.append(
            _log_return(float(benchmark_value), previous_benchmark)
        )
        previous_strategy = float(strategy_value)
        previous_benchmark = float(benchmark_value)
    if not strategy_returns:
        return {
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": None,
            "alpha_pct": None,
            "beta": None,
            "win_rate_pct": 0.0,
            "benchmark_annualized_volatility_pct": 0.0,
            "benchmark_sharpe_ratio": None,
        }

    strategy_mean = sum(strategy_returns) / len(strategy_returns)
    benchmark_mean = sum(benchmark_returns) / len(benchmark_returns)
    if len(strategy_returns) > 1:
        strategy_variance = sum(
            (value - strategy_mean) ** 2 for value in strategy_returns
        ) / (len(strategy_returns) - 1)
        benchmark_variance = sum(
            (value - benchmark_mean) ** 2 for value in benchmark_returns
        ) / (len(benchmark_returns) - 1)
        covariance = sum(
            (strategy - strategy_mean) * (benchmark - benchmark_mean)
            for strategy, benchmark in zip(strategy_returns, benchmark_returns, strict=True)
        ) / (len(strategy_returns) - 1)
    else:
        strategy_variance = 0.0
        benchmark_variance = 0.0
        covariance = 0.0
    volatility = math.sqrt(max(strategy_variance, 0.0))
    benchmark_volatility = math.sqrt(max(benchmark_variance, 0.0))
    beta = covariance / benchmark_variance if benchmark_variance > 0 else None
    alpha = (
        (strategy_mean - beta * benchmark_mean) * 252.0 * 100.0
        if beta is not None
        else None
    )
    return {
        "annualized_volatility_pct": volatility * math.sqrt(252.0) * 100.0,
        "sharpe_ratio": (
            strategy_mean / volatility * math.sqrt(252.0)
            if volatility > 0
            else None
        ),
        "alpha_pct": alpha,
        "beta": beta,
        "win_rate_pct": sum(value > 0 for value in strategy_returns) / len(strategy_returns) * 100.0,
        # Benchmark-side risk metrics, same methodology as the strategy side
        # (log returns, 252-day annualization, zero risk-free rate), so the
        # two columns are directly comparable.
        "benchmark_annualized_volatility_pct": (
            benchmark_volatility * math.sqrt(252.0) * 100.0
        ),
        "benchmark_sharpe_ratio": (
            benchmark_mean / benchmark_volatility * math.sqrt(252.0)
            if benchmark_volatility > 0
            else None
        ),
    }


def _position_label(holdings: Mapping[str, int]) -> str:
    owned = sorted(symbol for symbol, shares in holdings.items() if shares > 0)
    return ",".join(owned) if owned else CASH


def _buy_allocation(
    *,
    cash: float,
    target_weights: Mapping[str, float],
    day_bars: Mapping[str, PriceBar],
    cost_rate: float,
    lot_size: int,
) -> tuple[float, dict[str, int], list[dict[str, float | int | str]], float]:
    holdings = {symbol: 0 for symbol in day_bars}
    trades: list[dict[str, float | int | str]] = []
    total_cost = 0.0
    allocation_cash = cash
    for symbol, weight in sorted(target_weights.items()):
        price = day_bars[symbol].open
        budget = allocation_cash * weight
        affordable = math.floor(budget / (price * (1.0 + cost_rate)))
        shares = (affordable // lot_size) * lot_size
        if shares <= 0:
            continue
        notional = shares * price
        fee = notional * cost_rate
        cash -= notional + fee
        holdings[symbol] = shares
        total_cost += fee
        trades.append(
            {
                "symbol": symbol,
                "action": "buy",
                "shares": shares,
                "price": price,
                "notional": notional,
                "fee": fee,
            }
        )
    return cash, holdings, trades, total_cost


def _empty_report(
    *,
    strategy_name: str,
    status: str,
    config: StrategyBacktestConfig,
    signal_count: int,
    skipped_signal_count: int,
    symbols: Sequence[str],
    warning: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "strategy_name": strategy_name,
        "symbols": list(symbols),
        "metrics": {
            "initial_capital": config.initial_capital,
            "final_equity": config.initial_capital,
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "benchmark_return_pct": 0.0,
            "excess_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "benchmark_max_drawdown_pct": 0.0,
            "trade_count": 0,
            "exposure_pct": 0.0,
            "total_cost": 0.0,
            "turnover_pct": 0.0,
            "signal_count": signal_count,
            "executed_signal_count": 0,
            "skipped_signal_count": skipped_signal_count,
            "trading_days": 0,
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": None,
            "alpha_pct": None,
            "beta": None,
            "win_rate_pct": 0.0,
        },
        "benchmark": {
            "kind": "single_asset_buy_and_hold" if len(symbols) == 1 else "equal_weight_buy_and_hold",
            "symbols": list(symbols),
        },
        "assumptions": {
            "execution": "next_common_trading_day_open",
            "initial_capital": config.initial_capital,
            "side_cost_pct": config.side_cost_pct,
            "lot_size": config.lot_size,
        },
        "equity_curve": [],
        "trades": [],
        "warnings": [warning],
    }


def run_portfolio_backtest(
    *,
    strategy_name: str,
    allocations: Sequence[TargetAllocation],
    bars: Sequence[PriceBar],
    config: StrategyBacktestConfig,
) -> dict[str, Any]:
    """Run a target-allocation strategy without asset-type assumptions."""

    symbols, calendar, bars_by_date = _normalize_bars(bars)
    available_symbols = set(symbols)

    scheduled: dict[date, tuple[TargetAllocation, dict[str, float]]] = {}
    skipped_signal_count = 0
    indexed_allocations = list(enumerate(allocations))
    indexed_allocations.sort(key=lambda item: (item[1].signal_date, item[0]))
    for _, allocation in indexed_allocations:
        weights = _normalized_weights(
            allocation.target_weights,
            available_symbols=available_symbols,
        )
        calendar_index = bisect_right(calendar, allocation.signal_date)
        if calendar_index >= len(calendar):
            skipped_signal_count += 1
            continue
        scheduled[calendar[calendar_index]] = (allocation, weights)

    if not scheduled:
        return _empty_report(
            strategy_name=strategy_name,
            status="no_executable_signals",
            config=config,
            signal_count=len(allocations),
            skipped_signal_count=skipped_signal_count,
            symbols=symbols,
            warning="No signal has a later common trading-day open in the supplied data.",
        )

    first_date = min(scheduled)
    simulation_dates = [trading_date for trading_date in calendar if trading_date >= first_date]
    cost_rate = config.side_cost_pct / 100.0

    cash = float(config.initial_capital)
    holdings = {symbol: 0 for symbol in symbols}
    current_target: dict[str, float] | None = None
    trades: list[dict[str, Any]] = []
    total_cost = 0.0
    turnover_notional = 0.0
    equity_curve: list[dict[str, Any]] = []
    invested_days = 0

    benchmark_weights = {symbol: 1.0 / len(symbols) for symbol in symbols}
    benchmark_cash, benchmark_holdings, _, benchmark_cost = _buy_allocation(
        cash=float(config.initial_capital),
        target_weights=benchmark_weights,
        day_bars=bars_by_date[first_date],
        cost_rate=cost_rate,
        lot_size=config.lot_size,
    )

    for trading_date in simulation_dates:
        day_bars = bars_by_date[trading_date]
        scheduled_target = scheduled.get(trading_date)
        if scheduled_target is not None:
            allocation, target_weights = scheduled_target
            if current_target != target_weights:
                equity_at_open = cash + sum(
                    holdings[symbol] * day_bars[symbol].open
                    for symbol in symbols
                )
                target_holdings: dict[str, int] = {}
                for symbol in symbols:
                    weight = target_weights.get(symbol, 0.0)
                    price = day_bars[symbol].open
                    affordable = math.floor(
                        equity_at_open * weight / (price * (1.0 + cost_rate))
                    )
                    target_holdings[symbol] = (
                        affordable // config.lot_size
                    ) * config.lot_size

                for symbol in symbols:
                    shares = holdings[symbol] - target_holdings[symbol]
                    if shares <= 0:
                        continue
                    price = day_bars[symbol].open
                    notional = shares * price
                    fee = notional * cost_rate
                    cash += notional - fee
                    total_cost += fee
                    turnover_notional += notional
                    holdings[symbol] -= shares
                    trades.append(
                        {
                            "source_id": allocation.source_id,
                            "signal_date": allocation.signal_date.isoformat(),
                            "execution_date": trading_date.isoformat(),
                            "symbol": symbol,
                            "action": "sell",
                            "shares": shares,
                            "price": price,
                            "notional": notional,
                            "fee": fee,
                            "reason": allocation.reason,
                        }
                    )

                for symbol in symbols:
                    shares_needed = target_holdings[symbol] - holdings[symbol]
                    if shares_needed <= 0:
                        continue
                    price = day_bars[symbol].open
                    affordable = math.floor(cash / (price * (1.0 + cost_rate)))
                    shares = min(
                        shares_needed,
                        (affordable // config.lot_size) * config.lot_size,
                    )
                    if shares <= 0:
                        continue
                    notional = shares * price
                    fee = notional * cost_rate
                    cash -= notional + fee
                    total_cost += fee
                    turnover_notional += notional
                    holdings[symbol] += shares
                    trades.append(
                        {
                            "source_id": allocation.source_id,
                            "signal_date": allocation.signal_date.isoformat(),
                            "execution_date": trading_date.isoformat(),
                            "symbol": symbol,
                            "action": "buy",
                            "shares": shares,
                            "price": price,
                            "notional": notional,
                            "fee": fee,
                            "reason": allocation.reason,
                        }
                    )
                current_target = target_weights

        position_value = sum(
            holdings[symbol] * day_bars[symbol].close
            for symbol in symbols
        )
        equity = cash + position_value
        benchmark_equity = benchmark_cash + sum(
            benchmark_holdings[symbol] * day_bars[symbol].close
            for symbol in symbols
        )
        if position_value > 0:
            invested_days += 1
        equity_curve.append(
            {
                "date": trading_date.isoformat(),
                "strategy_equity": equity,
                "benchmark_equity": benchmark_equity,
                "strategy_return_pct": (equity / config.initial_capital - 1.0) * 100.0,
                "benchmark_return_pct": (
                    benchmark_equity / config.initial_capital - 1.0
                )
                * 100.0,
                "position": _position_label(holdings),
            }
        )

    strategy_values = [float(point["strategy_equity"]) for point in equity_curve]
    benchmark_values = [float(point["benchmark_equity"]) for point in equity_curve]
    strategy_peak = float(config.initial_capital)
    benchmark_peak = float(config.initial_capital)
    for point, strategy_value, benchmark_value in zip(
        equity_curve,
        strategy_values,
        benchmark_values,
        strict=True,
    ):
        strategy_peak = max(strategy_peak, strategy_value)
        benchmark_peak = max(benchmark_peak, benchmark_value)
        point["drawdown_pct"] = (strategy_value / strategy_peak - 1.0) * 100.0
        point["benchmark_drawdown_pct"] = (
            benchmark_value / benchmark_peak - 1.0
        ) * 100.0

    final_equity = strategy_values[-1]
    benchmark_final = benchmark_values[-1]
    total_return_pct = (final_equity / config.initial_capital - 1.0) * 100.0
    benchmark_return_pct = (
        benchmark_final / config.initial_capital - 1.0
    ) * 100.0
    warnings: list[str] = []
    risk_metrics = _risk_metrics(
        strategy_values,
        benchmark_values,
        opening_equity=config.initial_capital,
        benchmark_opening_equity=config.initial_capital,
    )
    if len(scheduled) < 2:
        warnings.append("Fewer than two executable signals; the result is not enough to judge a strategy.")
    if not trades:
        warnings.append("Signals produced no actual trade.")
    if config.side_cost_pct == 0:
        warnings.append("Trading cost is set to 0%; returns are before fees and slippage.")

    return {
        "status": "completed",
        "strategy_name": strategy_name,
        "symbols": list(symbols),
        "period": {
            "start_date": first_date.isoformat(),
            "end_date": simulation_dates[-1].isoformat(),
        },
        "metrics": {
            "initial_capital": config.initial_capital,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": _annualized_return_pct(
                initial_capital=config.initial_capital,
                final_equity=final_equity,
                trading_days=len(simulation_dates),
            ),
            "benchmark_return_pct": benchmark_return_pct,
            "benchmark_annualized_return_pct": _annualized_return_pct(
                initial_capital=config.initial_capital,
                final_equity=benchmark_final,
                trading_days=len(simulation_dates),
            ),
            "excess_return_pct": total_return_pct - benchmark_return_pct,
            "max_drawdown_pct": _max_drawdown_pct(
                [config.initial_capital, *strategy_values]
            ),
            "benchmark_max_drawdown_pct": _max_drawdown_pct(
                [config.initial_capital, *benchmark_values]
            ),
            "trade_count": len(trades),
            "exposure_pct": invested_days / len(simulation_dates) * 100.0,
            "total_cost": total_cost,
            "turnover_pct": turnover_notional / config.initial_capital * 100.0,
            "signal_count": len(allocations),
            "executed_signal_count": len(scheduled),
            "skipped_signal_count": skipped_signal_count,
            "trading_days": len(simulation_dates),
            **risk_metrics,
        },
        "benchmark": {
            "kind": (
                "single_asset_buy_and_hold"
                if len(symbols) == 1
                else "equal_weight_buy_and_hold"
            ),
            "symbols": list(symbols),
            "entry_cost": benchmark_cost,
        },
        "assumptions": {
            "execution": "next_common_trading_day_open",
            "initial_capital": config.initial_capital,
            "side_cost_pct": config.side_cost_pct,
            "lot_size": config.lot_size,
        },
        "equity_curve": equity_curve,
        "trades": trades,
        "warnings": warnings,
    }


def _build_comparison_series(
    *,
    benchmarks: Mapping[str, Any],
    dates: Sequence[date],
    simulation_indices: Sequence[int],
    initial_capital: float,
) -> dict[str, Any]:
    """Build normalized comparison series aligned to the simulation calendar.

    Each benchmark (index or single symbol) is normalized to its first valid
    close inside the simulation window. Missing dates are ``None``.
    """
    ordinal_to_index = {
        dates[index].toordinal(): position
        for position, index in enumerate(simulation_indices)
    }
    comparisons: dict[str, Any] = {}
    for symbol, frame in benchmarks.items():
        close_by_ordinal: dict[int, float] = {}
        for value in frame.itertuples(index=False):
            close_by_ordinal[value.date.toordinal()] = float(value.close)
        if not close_by_ordinal:
            continue
        window = [
            close_by_ordinal.get(dates[index].toordinal())
            for index in simulation_indices
        ]
        first_valid = None
        first_valid_index = None
        for position, value in enumerate(window):
            if value is not None and not (isinstance(value, float) and value != value):
                first_valid = value
                first_valid_index = position
                break
        if first_valid is None or first_valid <= 0:
            continue
        normalized = [
            None if value is None or (isinstance(value, float) and value != value)
            else value / first_valid * initial_capital
            for value in window
        ]
        values = [value for value in normalized if value is not None]
        if not values:
            continue
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = min(max_drawdown, value / peak - 1.0)
        comparisons[symbol] = {
            "symbol": symbol,
            "equity": normalized,
            "total_return_pct": (values[-1] / initial_capital - 1.0) * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
            "first_valid_date": (
                dates[simulation_indices[first_valid_index]].isoformat()
                if first_valid_index is not None
                else None
            ),
        }
    return comparisons


def _align_strategy_comparisons(
    *,
    comparisons_reports: Sequence[Mapping[str, Any]],
    dates: Sequence[date],
    simulation_indices: Sequence[int],
    initial_capital: float,
) -> dict[str, Any]:
    """Align saved-strategy comparison runs to the main simulation calendar."""
    out: dict[str, Any] = {}
    for entry in comparisons_reports:
        name = str(entry.get("name") or "comparison")
        report = entry.get("report") or {}
        equity_by_date: dict[str, float] = {}
        for point in report.get("equity_curve") or []:
            value = point.get("strategy_equity")
            if value is None:
                continue
            value = float(value)
            if value != value:  # NaN
                continue
            date_key = str(point.get("date") or "")
            if date_key:
                equity_by_date[date_key] = value
        window = [
            equity_by_date.get(dates[index].isoformat())
            for index in simulation_indices
        ]
        first_valid = None
        first_valid_index = None
        for position, value in enumerate(window):
            if value is not None:
                first_valid = value
                first_valid_index = position
                break
        if first_valid is None or first_valid <= 0:
            continue
        normalized = [
            None if value is None else value / first_valid * initial_capital
            for value in window
        ]
        values = [value for value in normalized if value is not None]
        if not values:
            continue
        peak = values[0]
        max_drawdown = 0.0
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                max_drawdown = min(max_drawdown, value / peak - 1.0)
        out[f"strategy:{name}"] = {
            "symbol": f"strategy:{name}",
            "name": name,
            "equity": normalized,
            # Total return is computed on the same normalized window as the
            # plotted curve, so the headline number and the chart agree even
            # when the comparison strategy started earlier than the main one.
            "total_return_pct": (values[-1] / initial_capital - 1.0) * 100.0,
            "max_drawdown_pct": max_drawdown * 100.0,
            "first_valid_date": (
                dates[simulation_indices[first_valid_index]].isoformat()
                if first_valid_index is not None
                else None
            ),
        }
    return out


def run_portfolio_backtest_view(
    *,
    strategy_name: str,
    allocations: Sequence[TargetAllocation],
    view: "MarketView",
    config: StrategyBacktestConfig,
    benchmarks: Mapping[str, Any] | None = None,
    strategy_comparisons: Sequence[Mapping[str, Any]] | None = None,
    period_start: date | None = None,
    period_end: date | None = None,
    execution_price: str = "open",
) -> dict[str, Any]:
    """Vectorized portfolio backtest over a MarketView (large universes).

    ``execution_price`` selects the fill price on the signal's execution day:
    ``"open"`` (default, T+1 open) or ``"close"`` (T+1 close, avoids
    overnight-gap noise at the cost of one day of latency).

    Same semantics as ``run_portfolio_backtest`` but reads prices from the
    numpy matrices instead of ``PriceBar`` lists, and computes the benchmark
    as an equal-weight buy-and-hold of every tradable universe member.
    """
    from src.core.script_market import MarketView  # local import for clarity

    dates = view.calendar
    symbols = [str(symbol) for symbol in view.symbols]
    symbol_index = {symbol: i for i, symbol in enumerate(symbols)}
    close_matrix = view.close
    open_matrix = view.open
    if execution_price not in ("open", "close"):
        raise ValueError("execution_price must be 'open' or 'close'")
    trade_matrix = open_matrix if execution_price == "open" else close_matrix

    # Truncate the simulation at the last date with tradable members so a
    # stale membership window cannot poison the final benchmark value. This
    # is computed before scheduling so a signal whose execution day falls
    # past the end of the window is counted as skipped, never "executed".
    member_days = view.members.any(axis=1)
    member_positions = np.flatnonzero(member_days)
    if member_positions.size == 0:
        return _empty_report(
            strategy_name=strategy_name,
            status="no_executable_signals",
            config=config,
            signal_count=len(allocations),
            skipped_signal_count=0,
            symbols=symbols,
            warning="The universe has no tradable members during the requested period.",
        )
    last_member_index = int(member_positions[-1])
    truncation_warning: str | None = None
    if last_member_index < len(dates) - 1:
        truncation_warning = (
            f"Simulation truncated at {dates[last_member_index].isoformat()}: "
            "the universe membership data ends there."
        )
    window_end = min(len(dates), last_member_index + 1)
    if period_end is not None:
        window_end = min(window_end, bisect_right(dates, period_end))

    scheduled: dict[date, tuple[TargetAllocation, dict[str, float]]] = {}
    skipped_signal_count = 0
    indexed_allocations = list(enumerate(allocations))
    indexed_allocations.sort(key=lambda item: (item[1].signal_date, item[0]))
    for _, allocation in indexed_allocations:
        if period_end is not None and allocation.signal_date > period_end:
            skipped_signal_count += 1
            continue
        weights = _normalized_weights(
            allocation.target_weights,
            available_symbols=set(symbols),
        )
        calendar_index = bisect_right(dates, allocation.signal_date)
        if calendar_index >= window_end:
            # The signal would only execute after the window closes (e.g. a
            # period-end signal whose next trading-day open is outside the
            # window): it is skipped, not reported as executed.
            skipped_signal_count += 1
            continue
        scheduled[dates[calendar_index]] = (allocation, weights)

    if not scheduled:
        return _empty_report(
            strategy_name=strategy_name,
            status="no_executable_signals",
            config=config,
            signal_count=len(allocations),
            skipped_signal_count=skipped_signal_count,
            symbols=symbols,
            warning="No signal has a later common trading-day open in the supplied data.",
        )

    # The simulation window starts at period_start when one is given (it is
    # the true first simulated day, not silently the first signal day);
    # otherwise it starts at the first signal execution day. Signals dated at
    # or before period_start are kept and pre-executed so a sub-period opens
    # with the position the strategy already held.
    if period_start is not None:
        window_start_index = bisect_left(dates, period_start)
    else:
        window_start_index = dates.index(min(scheduled))
    if window_start_index >= window_end:
        return _empty_report(
            strategy_name=strategy_name,
            status="no_executable_signals",
            config=config,
            signal_count=len(allocations),
            skipped_signal_count=skipped_signal_count,
            symbols=symbols,
            warning="The universe has no tradable members during the requested period.",
        )
    simulation_indices = list(range(window_start_index, window_end))
    first_date = dates[window_start_index]
    cost_rate = config.side_cost_pct / 100.0

    cash = float(config.initial_capital)
    holdings = {symbol: 0 for symbol in symbols}
    current_target: dict[str, float] | None = None
    trades: list[dict[str, Any]] = []
    total_cost = 0.0
    turnover_notional = 0.0
    equity_curve: list[dict[str, Any]] = []
    invested_days = 0
    blocked_signal_count = 0

    # Initialize valuation carry-forward prices for positions already held.
    last_close: dict[str, float] = {}
    for symbol in symbols:
        if holdings[symbol]:
            past = close_matrix[:window_start_index, symbol_index[symbol]]
            valid_past = past[~np.isnan(past)]
            if valid_past.size:
                last_close[symbol] = float(valid_past[-1])

    pending_retry: tuple[TargetAllocation, dict[str, float]] | None = None

    def _apply_target(
        execution_date: date,
        allocation: TargetAllocation,
        weights: dict[str, float],
        *,
        is_retry: bool = False,
    ) -> None:
        """Execute one target allocation at the open of ``execution_date``.

        Suspended symbols (missing open) block that leg of the rebalance; when
        any leg is blocked the previous target is kept and the pending target
        is retried on every subsequent day until it executes, so a suspension
        never silently drops a rebalance even when no new signal arrives.
        """
        nonlocal cash, current_target, total_cost, turnover_notional, blocked_signal_count, pending_retry
        if current_target == weights:
            return
        t = dates.index(execution_date)
        equity_at_open = cash
        for symbol in symbols:
            price = trade_matrix[t, symbol_index[symbol]]
            if holdings[symbol] and not np.isnan(price):
                equity_at_open += holdings[symbol] * float(price)
        target_holdings: dict[str, int] = {}
        blocked = False
        for symbol in symbols:
            weight = weights.get(symbol, 0.0)
            if weight <= 0:
                continue
            price = trade_matrix[t, symbol_index[symbol]]
            if np.isnan(price) or price <= 0:
                blocked = True
                continue
            affordable = math.floor(
                equity_at_open * weight / (price * (1.0 + cost_rate))
            )
            target_holdings[symbol] = (
                affordable // config.lot_size
            ) * config.lot_size

        # Atomic rebalance: if any leg that needs to trade is suspended, the
        # WHOLE target is deferred — never a partial-leg execution that
        # leaves an unintended intermediate position and extra turnover.
        for symbol in symbols:
            weight = weights.get(symbol, 0.0)
            desired = target_holdings.get(symbol, 0)
            if weight <= 0 and holdings[symbol] <= 0:
                continue
            if weight > 0 or holdings[symbol] != desired:
                price = trade_matrix[t, symbol_index[symbol]]
                if np.isnan(price) or price <= 0:
                    blocked = True
                    break
        if blocked:
            if not is_retry:
                blocked_signal_count += 1
            pending_retry = (allocation, weights)
            return

        for symbol in symbols:
            shares = holdings[symbol] - target_holdings.get(symbol, 0)
            if shares <= 0:
                continue
            price = trade_matrix[t, symbol_index[symbol]]
            if np.isnan(price) or price <= 0:
                continue
            notional = shares * float(price)
            fee = notional * cost_rate
            cash += notional - fee
            total_cost += fee
            turnover_notional += notional
            holdings[symbol] -= shares
            trades.append(
                {
                    "source_id": allocation.source_id,
                    "signal_date": allocation.signal_date.isoformat(),
                    "execution_date": execution_date.isoformat(),
                    "symbol": symbol,
                    "action": "sell",
                    "shares": shares,
                    "price": float(price),
                    "notional": notional,
                    "fee": fee,
                    "reason": allocation.reason,
                }
            )

        for symbol in symbols:
            shares_needed = target_holdings.get(symbol, 0) - holdings[symbol]
            if shares_needed <= 0:
                continue
            price = trade_matrix[t, symbol_index[symbol]]
            if np.isnan(price) or price <= 0:
                blocked = True
                continue
            affordable = math.floor(cash / (price * (1.0 + cost_rate)))
            shares = min(
                shares_needed,
                (affordable // config.lot_size) * config.lot_size,
            )
            if shares <= 0:
                continue
            notional = shares * float(price)
            fee = notional * cost_rate
            cash -= notional + fee
            total_cost += fee
            turnover_notional += notional
            holdings[symbol] += shares
            last_close[symbol] = float(price)
            trades.append(
                {
                    "source_id": allocation.source_id,
                    "signal_date": allocation.signal_date.isoformat(),
                    "execution_date": execution_date.isoformat(),
                    "symbol": symbol,
                    "action": "buy",
                    "shares": shares,
                    "price": float(price),
                    "notional": notional,
                    "fee": fee,
                    "reason": allocation.reason,
                }
            )
        if blocked:
            if not is_retry:
                blocked_signal_count += 1
            pending_retry = (allocation, weights)
        else:
            pending_retry = None
            current_target = weights

    # Pre-execute signals whose execution day falls before the window so a
    # sub-period starts with the position the strategy already held, instead
    # of silently resetting to cash.
    for execution_date, (allocation, weights) in sorted(scheduled.items()):
        if execution_date >= first_date:
            break
        _apply_target(execution_date, allocation, weights)

    # Window-scoped metric baselines: opening trades (pre-window execution)
    # stay visible in the trade list but are excluded from the sub-period's
    # trade_count / total_cost / turnover.
    window_trade_count_base = len(trades)
    window_total_cost_base = total_cost
    window_turnover_base = turnover_notional

    # Vectorized benchmark: equal-weight buy-and-hold of the members valid on
    # the first simulated day with weights fixed at 1/N0. Suspended closes
    # carry the last valid close forward (the same valuation rule as the
    # strategy side), so delisted/suspended members keep their weight instead
    # of being renormalized away — the previous per-day renormalization
    # silently erased delisting losses (survivorship bias).
    window_close = close_matrix[window_start_index:window_end]
    window_members = view.members[window_start_index:window_end]
    base_valid = (
        window_members[0] & ~np.isnan(window_close[0]) & (window_close[0] > 0)
    )
    n0 = int(base_valid.sum())
    bench_returns = np.full(len(simulation_indices), np.nan, dtype=np.float64)
    if n0 > 0:
        base_close = window_close[0]
        filled = window_close.copy()
        rows, cols = filled.shape
        fill_positions = np.where(~np.isnan(filled), np.arange(rows)[:, None], 0)
        np.maximum.accumulate(fill_positions, axis=0, out=fill_positions)
        col_index = np.broadcast_to(np.arange(cols), (rows, cols))
        filled = filled[fill_positions, col_index]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = filled[:, base_valid] / base_close[base_valid]
            bench_returns = np.nanmean(ratios, axis=1)
    benchmark_equities = config.initial_capital * bench_returns

    opening_equity: float | None = None
    for t in simulation_indices:
        trading_date = dates[t]
        if opening_equity is None:
            # Opening equity = the portfolio value right BEFORE the first
            # window day's open execution: cash plus positions marked at the
            # day's open (or the carried-forward price when suspended). The
            # first day's intraday return therefore counts, instead of the
            # first day's close being silently treated as the base.
            opening_equity = cash
            for symbol in symbols:
                if holdings[symbol]:
                    price = open_matrix[t, symbol_index[symbol]]
                    if not np.isnan(price):
                        opening_equity += holdings[symbol] * float(price)
                    else:
                        opening_equity += holdings[symbol] * last_close.get(symbol, 0.0)
        scheduled_target = scheduled.get(trading_date)
        if scheduled_target is not None:
            allocation, target_weights = scheduled_target
            _apply_target(trading_date, allocation, target_weights)
        elif pending_retry is not None:
            # A suspension blocked an earlier rebalance; retry it every day
            # (even without a fresh signal) until the target is executed.
            retry_allocation, retry_weights = pending_retry
            _apply_target(
                trading_date, retry_allocation, retry_weights, is_retry=True
            )

        position_value = 0.0
        for symbol in symbols:
            if holdings[symbol]:
                price = close_matrix[t, symbol_index[symbol]]
                if not np.isnan(price):
                    last_close[symbol] = float(price)
                position_value += holdings[symbol] * last_close.get(symbol, 0.0)
        equity = cash + position_value
        if position_value > 0:
            invested_days += 1
        benchmark_equity = float(benchmark_equities[t - window_start_index])
        equity_curve.append(
            {
                "date": trading_date.isoformat(),
                "strategy_equity": equity,
                "benchmark_equity": benchmark_equity if benchmark_equity == benchmark_equity else config.initial_capital,
                "strategy_return_pct": (equity / config.initial_capital - 1.0) * 100.0,
                "benchmark_return_pct": (
                    benchmark_equity / config.initial_capital - 1.0
                )
                * 100.0
                if benchmark_equity == benchmark_equity
                else 0.0,
                "position": _position_label(holdings),
            }
        )

    strategy_values = [float(point["strategy_equity"]) for point in equity_curve]
    benchmark_values = [
        float(point["benchmark_equity"]) if point["benchmark_equity"] == point["benchmark_equity"]
        else float(config.initial_capital)
        for point in equity_curve
    ]
    # opening_equity is captured in the simulation loop (value right before
    # the first window day's open execution); fall back defensively.
    assert opening_equity is not None and opening_equity >= 0
    strategy_peak = opening_equity
    benchmark_peak = float(config.initial_capital)
    for point, strategy_value, benchmark_value in zip(
        equity_curve, strategy_values, benchmark_values, strict=True
    ):
        strategy_peak = max(strategy_peak, strategy_value)
        benchmark_peak = max(benchmark_peak, benchmark_value)
        point["drawdown_pct"] = (strategy_value / strategy_peak - 1.0) * 100.0
        point["benchmark_drawdown_pct"] = (
            benchmark_value / benchmark_peak - 1.0
        ) * 100.0
    # Returns are measured against the window's opening equity (the value
    # right before the first day's open execution), NOT the original capital:
    # the first day's intraday return counts, and a sub-period that opens
    # with a carried-in position reports the sub-period's own return.
    if opening_equity > 0:
        for point in equity_curve:
            point["strategy_return_pct"] = (
                float(point["strategy_equity"]) / opening_equity - 1.0
            ) * 100.0

    final_equity = strategy_values[-1]
    benchmark_final = benchmark_values[-1]
    if opening_equity > 0:
        total_return_pct = (final_equity / opening_equity - 1.0) * 100.0
    else:
        total_return_pct = -100.0 if final_equity > 0 else 0.0
    benchmark_return_pct = (benchmark_final / config.initial_capital - 1.0) * 100.0
    warnings: list[str] = []
    if truncation_warning:
        warnings.append(truncation_warning)
    risk_metrics = _risk_metrics(
        strategy_values,
        benchmark_values,
        opening_equity=opening_equity,
        benchmark_opening_equity=config.initial_capital,
    )
    comparisons: dict[str, Any] = {}
    if benchmarks:
        comparisons = _build_comparison_series(
            benchmarks=benchmarks,
            dates=dates,
            simulation_indices=simulation_indices,
            initial_capital=config.initial_capital,
        )
    if strategy_comparisons:
        comparisons.update(
            _align_strategy_comparisons(
                comparisons_reports=strategy_comparisons,
                dates=dates,
                simulation_indices=simulation_indices,
                initial_capital=config.initial_capital,
            )
        )
    in_window_signal_count = sum(
        1 for execution_date in scheduled if execution_date >= first_date
    )
    if in_window_signal_count < 2:
        warnings.append("Fewer than two executable signals; the result is not enough to judge a strategy.")
    if not trades:
        warnings.append("Signals produced no actual trade.")
    if blocked_signal_count:
        warnings.append(
            f"{blocked_signal_count} signal(s) were blocked by suspensions: "
            "context.current_weights reflects requested targets, not the "
            "actual holdings after lot rounding, insufficient cash or "
            "suspension delays."
        )
    if config.side_cost_pct == 0:
        warnings.append("Trading cost is set to 0%; returns are before fees and slippage.")

    return {
        "status": "completed",
        "strategy_name": strategy_name,
        "symbols": list(symbols),
        "period": {
            "start_date": first_date.isoformat(),
            "end_date": dates[simulation_indices[-1]].isoformat(),
        },
        "metrics": {
            "initial_capital": config.initial_capital,
            "opening_equity": opening_equity,
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": _annualized_return_pct(
                initial_capital=opening_equity,
                final_equity=final_equity,
                trading_days=len(simulation_indices),
            ),
            "benchmark_return_pct": benchmark_return_pct,
            "benchmark_annualized_return_pct": _annualized_return_pct(
                initial_capital=config.initial_capital,
                final_equity=benchmark_final,
                trading_days=len(simulation_indices),
            ),
            "excess_return_pct": total_return_pct - benchmark_return_pct,
            "max_drawdown_pct": _max_drawdown_pct(
                [opening_equity, *strategy_values]
            ),
            "benchmark_max_drawdown_pct": _max_drawdown_pct(
                [config.initial_capital, *benchmark_values]
            ),
            "trade_count": len(trades) - window_trade_count_base,
            "exposure_pct": invested_days / len(simulation_indices) * 100.0,
            "total_cost": total_cost - window_total_cost_base,
            "turnover_pct": (
                (turnover_notional - window_turnover_base)
                / config.initial_capital
                * 100.0
            ),
            "signal_count": len(allocations),
            "executed_signal_count": len(scheduled),
            "skipped_signal_count": skipped_signal_count,
            "blocked_signal_count": blocked_signal_count,
            "trading_days": len(simulation_indices),
            **risk_metrics,
        },
        "comparisons": comparisons,
        "benchmark": {
            "kind": "equal_weight_buy_and_hold",
            "symbols": list(symbols),
            "entry_cost": 0.0,
        },
        "assumptions": {
            "execution": (
                "next_common_trading_day_open"
                if execution_price == "open"
                else "next_common_trading_day_close"
            ),
            "initial_capital": config.initial_capital,
            "side_cost_pct": config.side_cost_pct,
            "lot_size": config.lot_size,
        },
        "equity_curve": equity_curve,
        "trades": trades,
        "warnings": warnings,
    }
