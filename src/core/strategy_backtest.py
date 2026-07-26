# -*- coding: utf-8 -*-
"""Asset-agnostic target-allocation backtest engine.

Strategies produce target weights after a trading day's information is known.
The engine executes a changed target at the next common trading-day open and
tracks cash, board lots, costs, equity, drawdown, and a buy-and-hold benchmark.
It intentionally knows nothing about whether a symbol is a stock or an ETF.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
import math
from typing import Any, Mapping, Sequence


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
    ratio = final_equity / initial_capital
    if ratio <= 0:
        return -100.0
    return (ratio ** (252.0 / trading_days) - 1.0) * 100.0


def _risk_metrics(
    strategy_values: Sequence[float],
    benchmark_values: Sequence[float],
    *,
    initial_capital: float,
) -> dict[str, float | None]:
    strategy_returns: list[float] = []
    benchmark_returns: list[float] = []
    previous_strategy = float(initial_capital)
    previous_benchmark = float(initial_capital)
    for strategy_value, benchmark_value in zip(strategy_values, benchmark_values, strict=True):
        strategy_returns.append(float(strategy_value) / previous_strategy - 1.0)
        benchmark_returns.append(float(benchmark_value) / previous_benchmark - 1.0)
        previous_strategy = float(strategy_value)
        previous_benchmark = float(benchmark_value)
    if not strategy_returns:
        return {
            "annualized_volatility_pct": 0.0,
            "sharpe_ratio": None,
            "alpha_pct": None,
            "beta": None,
            "win_rate_pct": 0.0,
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
                    holdings[symbol] = shares
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
        initial_capital=config.initial_capital,
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
