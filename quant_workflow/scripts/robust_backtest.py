"""Shares-and-cash A-share simulator for robust strategy evaluation."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from robust_strategy_core import (  # noqa: E402
    build_constrained_portfolio,
    calculate_order_costs,
    can_execute_open,
    price_limit_ratio,
)


STRATEGIES = {
    "weekly_top50",
    "low_turnover",
    "regime_scaled",
    "sector_constrained",
    "full_risk_control",
}


def weekly_signal_execution_pairs(
    calendar: Sequence[Any], *, additional_execution_delay_days: int = 0
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    days = (
        pd.DatetimeIndex(pd.to_datetime(calendar, errors="coerce"))
        .dropna()
        .normalize()
        .unique()
        .sort_values()
    )
    if len(days) < 2:
        return []
    positions = {day: index for index, day in enumerate(days)}
    grouped = pd.Series(days, index=days).groupby(days.to_period("W-FRI")).max()
    pairs: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    if additional_execution_delay_days < 0:
        raise ValueError("additional_execution_delay_days must be nonnegative")
    for signal in grouped:
        index = positions[pd.Timestamp(signal)]
        execution_index = index + 1 + additional_execution_delay_days
        if execution_index < len(days):
            pairs.append((pd.Timestamp(signal), pd.Timestamp(days[execution_index])))
    return pairs


def calculate_all_in_order_costs(
    orders: pd.DataFrame,
    *,
    trade_date: str | pd.Timestamp,
    commission_rate: float,
    minimum_commission: float,
    round_trip_total_bps: float,
) -> dict[str, float]:
    if round_trip_total_bps < 0:
        raise ValueError("round_trip_total_bps must be nonnegative")
    regulatory = calculate_order_costs(
        orders,
        trade_date=trade_date,
        commission_rate=commission_rate,
        minimum_commission=minimum_commission,
        slippage_bps=0.0,
    )
    target = (
        float(
            pd.to_numeric(orders["notional"], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .sum()
        )
        * (round_trip_total_bps / 2.0)
        / 10000.0
    )
    total = max(float(regulatory["total"]), target)
    return {
        "commission": float(regulatory["commission"]),
        "stamp_tax": float(regulatory["stamp_tax"]),
        "slippage": total - float(regulatory["total"]),
        "total": total,
    }


def select_symbols(
    candidates: pd.DataFrame,
    *,
    strategy_id: str,
    topk: int,
    current_symbols: set[str],
    dropout_buffer: int = 10,
    target_position: float = 0.9,
    industry_max: float = 0.25,
    growth_star_max: float = 0.30,
) -> list[str]:
    if strategy_id not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_id}")
    if topk <= 0 or dropout_buffer < 0:
        raise ValueError("topk must be positive and dropout_buffer must be nonnegative")
    ranked = candidates.copy()
    ranked["score"] = pd.to_numeric(ranked["score"], errors="coerce")
    ranked["symbol"] = ranked["symbol"].astype(str)
    ranked = ranked.dropna(subset=["score"]).sort_values(
        ["score", "symbol"], ascending=[False, True], kind="stable"
    )
    order = ranked["symbol"].drop_duplicates().tolist()
    if strategy_id in {"weekly_top50", "regime_scaled"}:
        return order[:topk]
    rank_by_symbol = {symbol: index + 1 for index, symbol in enumerate(order)}
    retained = [
        symbol
        for symbol in order
        if symbol in current_symbols and rank_by_symbol[symbol] <= topk + dropout_buffer
    ]
    preferred = retained + [symbol for symbol in order if symbol not in retained]
    if strategy_id == "low_turnover":
        return sorted(
            preferred[:topk], key=lambda symbol: rank_by_symbol.get(symbol, math.inf)
        )

    metadata = ranked.drop_duplicates("symbol").set_index("symbol")
    denominator = max(float(target_position), 1e-12)
    industry_slots = max(
        1, min(topk, math.ceil(topk * float(industry_max) / denominator))
    )
    growth_star_slots = max(
        1, min(topk, math.ceil(topk * float(growth_star_max) / denominator))
    )
    selected: list[str] = []
    industry_counts: dict[str, int] = {}
    growth_star_count = 0
    for symbol in preferred:
        if symbol not in metadata.index:
            continue
        row = metadata.loc[symbol]
        industry = str(row.get("industry") or "")
        board = str(row.get("board") or "")
        if not industry or not board:
            continue
        if industry_counts.get(industry, 0) >= industry_slots:
            continue
        is_growth_star = board in {"growth", "star"}
        if is_growth_star and growth_star_count >= growth_star_slots:
            continue
        selected.append(symbol)
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if is_growth_star:
            growth_star_count += 1
        if len(selected) >= topk:
            break
    return selected


def build_strategy_target(
    candidates: pd.DataFrame,
    *,
    strategy_id: str,
    current_symbols: set[str],
    topk: int,
    target_position: float,
    single_stock_max: float,
    industry_max: float,
    growth_star_max: float,
    dropout_buffer: int = 10,
) -> pd.DataFrame:
    if strategy_id not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_id}")
    selected_symbols = select_symbols(
        candidates,
        strategy_id=strategy_id,
        topk=topk,
        current_symbols=current_symbols,
        dropout_buffer=dropout_buffer,
        target_position=target_position,
        industry_max=industry_max,
        growth_star_max=growth_star_max,
    )
    selected = candidates[
        candidates["symbol"].astype(str).isin(selected_symbols)
    ].copy()
    rank = {symbol: index for index, symbol in enumerate(selected_symbols)}
    selected["_rank"] = selected["symbol"].astype(str).map(rank)
    selected = (
        selected.sort_values("_rank").drop_duplicates("symbol").drop(columns="_rank")
    )
    return build_constrained_portfolio(
        selected,
        target_position=target_position,
        topk=topk,
        single_stock_max=single_stock_max,
        industry_max=industry_max,
        growth_star_max=growth_star_max,
    )


def performance_metrics(daily: pd.DataFrame) -> dict[str, float | int]:
    returns = pd.to_numeric(daily["portfolio_return"], errors="coerce").fillna(0.0)
    benchmark = pd.to_numeric(daily["benchmark_return"], errors="coerce").fillna(0.0)

    def annualized(series: pd.Series) -> float:
        if series.empty:
            return float("nan")
        total = float((1.0 + series).prod())
        return total ** (252.0 / len(series)) - 1.0 if total > 0 else -1.0

    annual_return = annualized(returns)
    benchmark_annual = annualized(benchmark)
    excess = returns - benchmark
    excess_std = float(excess.std(ddof=1))
    cumulative = (1.0 + returns).cumprod()
    return {
        "days": int(len(daily)),
        "annual_return": annual_return,
        "benchmark_annual_return": benchmark_annual,
        "annual_excess_return": annual_return - benchmark_annual,
        "information_ratio": float(excess.mean()) / excess_std * math.sqrt(252)
        if excess_std > 0
        else float("nan"),
        "max_drawdown": float((cumulative / cumulative.cummax() - 1.0).min())
        if not cumulative.empty
        else float("nan"),
        "total_transaction_cost": float(daily["transaction_cost"].sum()),
        "average_turnover": float(daily["turnover"].mean()),
    }


def _portfolio_within_caps(
    *,
    holdings: Mapping[str, int],
    cash: float,
    prices: Mapping[str, float],
    industries: Mapping[str, str],
    boards: Mapping[str, str],
    gross_position_max: float,
    single_stock_max: float,
    industry_max: float,
    growth_star_max: float,
) -> bool:
    values: dict[str, float] = {}
    for symbol, quantity in holdings.items():
        if quantity <= 0:
            continue
        price = float(prices.get(symbol, float("nan")))
        if (
            not math.isfinite(price)
            or price <= 0
            or not industries.get(symbol)
            or not boards.get(symbol)
        ):
            return False
        values[symbol] = int(quantity) * price
    portfolio_value = float(cash) + sum(values.values())
    if not math.isfinite(portfolio_value) or portfolio_value <= 0:
        return False
    tolerance = 1e-12
    if sum(values.values()) / portfolio_value > gross_position_max + tolerance:
        return False
    if any(
        value / portfolio_value > single_stock_max + tolerance
        for value in values.values()
    ):
        return False
    industry_values: dict[str, float] = {}
    growth_star_value = 0.0
    for symbol, value in values.items():
        industry = industries[symbol]
        industry_values[industry] = industry_values.get(industry, 0.0) + value
        if boards[symbol] in {"growth", "star"}:
            growth_star_value += value
    if any(
        value / portfolio_value > industry_max + tolerance
        for value in industry_values.values()
    ):
        return False
    return growth_star_value / portfolio_value <= growth_star_max + tolerance


def _normalize_inputs(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = predictions.rename(
        columns={"datetime": "date", "instrument": "symbol"}
    ).copy()
    if not {"date", "symbol", "score"}.issubset(pred.columns):
        raise ValueError(
            "predictions must contain date/symbol/score or datetime/instrument/score"
        )
    required_market = {
        "date",
        "symbol",
        "open",
        "previous_close",
        "list_date",
        "listing_trading_day",
        "industry",
        "board",
        "is_st",
        "suspended",
    }
    if not required_market.issubset(market.columns):
        raise ValueError(
            f"market is missing columns: {', '.join(sorted(required_market - set(market.columns)))}"
        )
    if not {"date", "open"}.issubset(benchmark.columns):
        raise ValueError("benchmark must contain date and open")
    for frame in (pred, market, benchmark):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    pred["symbol"] = pred["symbol"].astype(str).str.upper()
    market["symbol"] = market["symbol"].astype(str).str.upper()
    market["open"] = pd.to_numeric(market["open"], errors="coerce")
    market["previous_close"] = pd.to_numeric(market["previous_close"], errors="coerce")
    market["list_date"] = pd.to_datetime(
        market["list_date"], errors="coerce"
    ).dt.normalize()
    market["listing_trading_day"] = pd.to_numeric(
        market["listing_trading_day"], errors="coerce"
    ).astype("Int64")
    benchmark["open"] = pd.to_numeric(benchmark["open"], errors="coerce")
    if (
        market.duplicated(["date", "symbol"]).any()
        or benchmark.duplicated(["date"]).any()
    ):
        raise ValueError("market and benchmark keys must be unique")
    missing_execution_metadata = market["list_date"].isna() | (
        market["open"].notna()
        & ~market["suspended"].fillna(True).astype(bool)
        & market["listing_trading_day"].isna()
    )
    if missing_execution_metadata.any():
        raise ValueError("market listing metadata is incomplete")
    return pred, market, benchmark.sort_values("date")


def _price_limit_for_row(
    symbol: str, day: pd.Timestamp, row: Mapping[str, Any], *, side: str
) -> float | None:
    return price_limit_ratio(
        symbol,
        day,
        is_st=bool(row["is_st"]),
        list_date=row.get("list_date"),
        listing_trading_day=row.get("listing_trading_day"),
        side=side,
    )


def simulate_strategy(
    *,
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    benchmark: pd.DataFrame,
    strategy_id: str,
    initial_cash: float = 10_000_000.0,
    topk: int = 50,
    fixed_position: float = 0.9,
    regimes: pd.Series | None = None,
    regime_positions: Mapping[str, float] | None = None,
    single_stock_max: float = 0.03,
    industry_max: float = 0.25,
    growth_star_max: float = 0.30,
    dropout_buffer: int = 10,
    commission_rate: float = 0.0003,
    minimum_commission: float = 5.0,
    slippage_bps: float = 0.0,
    all_in_round_trip_bps: float | None = None,
    additional_execution_delay_days: int = 0,
    defer_unfilled_open_orders: bool = True,
) -> dict[str, Any]:
    if strategy_id not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy_id}")
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    pred, prices, benchmark_frame = _normalize_inputs(predictions, market, benchmark)
    calendar = pd.DatetimeIndex(benchmark_frame["date"].dropna().unique()).sort_values()
    schedule = {
        execution: signal
        for signal, execution in weekly_signal_execution_pairs(
            calendar, additional_execution_delay_days=additional_execution_delay_days
        )
    }
    calendar_positions = {day: index for index, day in enumerate(calendar)}
    benchmark_open = benchmark_frame.set_index("date")["open"]
    price_by_day = {
        day: frame.set_index("symbol")
        for day, frame in prices.groupby("date", sort=True)
    }
    prediction_by_day = {day: frame for day, frame in pred.groupby("date", sort=True)}
    normalized_regimes = None
    if regimes is not None:
        normalized_regimes = regimes.copy()
        normalized_regimes.index = pd.to_datetime(normalized_regimes.index).normalize()
    positions = regime_positions or {
        "risk_on": 0.9,
        "neutral": 0.7,
        "defensive": 0.5,
        "risk_off": 0.2,
    }

    cash = float(initial_cash)
    holdings: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    last_industries: dict[str, str] = {}
    last_boards: dict[str, str] = {}
    previous_value = float(initial_cash)
    previous_benchmark: float | None = None
    records: list[dict[str, Any]] = []
    pending_desired: dict[str, int] = {}
    pending_weights: dict[str, float] = {}

    def order_costs(
        side: str, notional: float, trade_day: pd.Timestamp
    ) -> dict[str, float]:
        order = pd.DataFrame({"side": [side], "notional": [notional]})
        if all_in_round_trip_bps is not None:
            return calculate_all_in_order_costs(
                order,
                trade_date=trade_day,
                commission_rate=commission_rate,
                minimum_commission=minimum_commission,
                round_trip_total_bps=all_in_round_trip_bps,
            )
        return calculate_order_costs(
            order,
            trade_date=trade_day,
            commission_rate=commission_rate,
            minimum_commission=minimum_commission,
            slippage_bps=slippage_bps,
        )

    for day in calendar:
        day_prices = price_by_day.get(day, pd.DataFrame())
        carried_holdings = dict(holdings)
        carried_prices = dict(last_prices)
        carried_boards = dict(last_boards)
        for symbol, row in day_prices.iterrows():
            industry = str(row.get("industry") or "")
            if industry:
                last_industries[str(symbol)] = industry
            board = str(row.get("board") or "")
            if board:
                last_boards[str(symbol)] = board
            if math.isfinite(float(row["open"])) and float(row["open"]) > 0:
                last_prices[str(symbol)] = float(row["open"])
        board_start_values = {"main": 0.0, "growth": 0.0, "star": 0.0}
        board_price_pnl = {"main": 0.0, "growth": 0.0, "star": 0.0}
        for symbol, quantity in carried_holdings.items():
            previous_price = float(carried_prices.get(symbol, float("nan")))
            current_price = float(last_prices.get(symbol, previous_price))
            board = carried_boards.get(symbol) or last_boards.get(symbol)
            if (
                board not in board_start_values
                or not math.isfinite(previous_price)
                or previous_price <= 0
                or not math.isfinite(current_price)
                or current_price <= 0
            ):
                continue
            board_start_values[board] += quantity * previous_price
            board_price_pnl[board] += quantity * (current_price - previous_price)
        value_before = cash + sum(
            quantity * last_prices.get(symbol, 0.0)
            for symbol, quantity in holdings.items()
        )
        cash_start_exposure = cash / previous_value if previous_value > 0 else 0.0
        trade_count = 0
        transaction_cost = 0.0
        board_transaction_cost = {"main": 0.0, "growth": 0.0, "star": 0.0}
        turnover_notional = 0.0
        signal_day = schedule.get(day)
        daily_position_limit = fixed_position
        if (
            strategy_id in {"regime_scaled", "full_risk_control"}
            and normalized_regimes is not None
            and calendar_positions[day] > 0
        ):
            regime_signal_day = calendar[calendar_positions[day] - 1]
            daily_position_limit = float(
                positions.get(
                    str(normalized_regimes.get(regime_signal_day)), fixed_position
                )
            )
        if (
            signal_day is None
            and strategy_id in {"regime_scaled", "full_risk_control"}
            and normalized_regimes is not None
            and calendar_positions[day] > 0
            and holdings
            and not day_prices.empty
        ):
            target_position = daily_position_limit
            invested_value = sum(
                quantity * last_prices.get(symbol, 0.0)
                for symbol, quantity in holdings.items()
            )
            gross_position = invested_value / value_before if value_before > 0 else 0.0
            if target_position + 1e-12 < gross_position:
                scale = target_position / gross_position if gross_position > 0 else 0.0
                for symbol in sorted(holdings):
                    if symbol not in day_prices.index:
                        continue
                    current = holdings[symbol]
                    target_quantity = int((current * scale) // 100) * 100
                    if target_quantity >= current:
                        continue
                    row = day_prices.loc[symbol]
                    limit = _price_limit_for_row(symbol, day, row, side="sell")
                    if not can_execute_open(
                        side="sell",
                        previous_close=float(row["previous_close"]),
                        open_price=float(row["open"]),
                        limit_ratio=limit,
                        suspended=bool(row["suspended"]),
                    ):
                        continue
                    quantity = current - target_quantity
                    notional = quantity * float(row["open"])
                    costs = order_costs("sell", notional, day)
                    cash += notional - costs["total"]
                    holdings[symbol] = target_quantity
                    transaction_cost += costs["total"]
                    board = last_boards.get(symbol)
                    if board in board_transaction_cost:
                        board_transaction_cost[board] += costs["total"]
                    turnover_notional += notional
                    trade_count += 1
                for symbol in list(pending_desired):
                    pending_desired[symbol] = min(
                        pending_desired[symbol], holdings.get(symbol, 0)
                    )
        if (
            signal_day is not None
            and signal_day in prediction_by_day
            and not day_prices.empty
        ):
            candidates = (
                prediction_by_day[signal_day]
                .loc[:, ["symbol", "score"]]
                .merge(
                    day_prices.reset_index().loc[
                        :, ["symbol", "industry", "board", "suspended", "open"]
                    ],
                    on="symbol",
                    how="left",
                )
            )
            candidates["tradable"] = candidates["open"].notna() & ~candidates[
                "suspended"
            ].fillna(True).astype(bool)
            target_position = fixed_position
            if (
                strategy_id in {"regime_scaled", "full_risk_control"}
                and normalized_regimes is not None
            ):
                target_position = daily_position_limit
            target = build_strategy_target(
                candidates,
                strategy_id=strategy_id,
                current_symbols={
                    symbol for symbol, quantity in holdings.items() if quantity > 0
                },
                topk=topk,
                target_position=target_position,
                single_stock_max=single_stock_max,
                industry_max=industry_max,
                growth_star_max=growth_star_max,
                dropout_buffer=dropout_buffer,
            )
            target_weights = dict(
                zip(target["symbol"].astype(str), target["weight"].astype(float))
            )
            desired: dict[str, int] = {}
            for symbol, weight in target_weights.items():
                if symbol not in day_prices.index:
                    continue
                open_price = float(day_prices.loc[symbol, "open"])
                if not math.isfinite(open_price) or open_price <= 0:
                    continue
                desired[symbol] = (
                    int(max(0.0, value_before * weight) // (open_price * 100)) * 100
                )
            pending_desired = {
                symbol: desired.get(symbol, 0)
                for symbol in set(holdings) | set(desired)
            }
            pending_weights = target_weights

        if pending_desired and not day_prices.empty:
            completed_orders: set[str] = set()
            for symbol in sorted(pending_desired):
                current = holdings.get(symbol, 0)
                target_quantity = pending_desired[symbol]
                if target_quantity == current:
                    completed_orders.add(symbol)
                    continue
                if target_quantity > current or symbol not in day_prices.index:
                    continue
                row = day_prices.loc[symbol]
                limit = _price_limit_for_row(symbol, day, row, side="sell")
                if not can_execute_open(
                    side="sell",
                    previous_close=float(row["previous_close"]),
                    open_price=float(row["open"]),
                    limit_ratio=limit,
                    suspended=bool(row["suspended"]),
                ):
                    continue
                quantity = current - target_quantity
                notional = quantity * float(row["open"])
                costs = order_costs("sell", notional, day)
                cash += notional - costs["total"]
                holdings[symbol] = target_quantity
                transaction_cost += costs["total"]
                board = last_boards.get(symbol)
                if board in board_transaction_cost:
                    board_transaction_cost[board] += costs["total"]
                turnover_notional += notional
                trade_count += 1
                completed_orders.add(symbol)

            for symbol, target_quantity in sorted(
                pending_desired.items(),
                key=lambda item: pending_weights.get(item[0], 0.0),
                reverse=True,
            ):
                current = holdings.get(symbol, 0)
                if target_quantity == current:
                    completed_orders.add(symbol)
                    continue
                if target_quantity < current or symbol not in day_prices.index:
                    continue
                row = day_prices.loc[symbol]
                limit = _price_limit_for_row(symbol, day, row, side="buy")
                if not can_execute_open(
                    side="buy",
                    previous_close=float(row["previous_close"]),
                    open_price=float(row["open"]),
                    limit_ratio=limit,
                    suspended=bool(row["suspended"]),
                ):
                    continue
                quantity = target_quantity - current
                open_price = float(row["open"])
                requested_lots = quantity // 100

                def feasible_buy(lots: int) -> bool:
                    proposed_quantity = int(lots) * 100
                    proposed_notional = proposed_quantity * open_price
                    proposed_costs = order_costs("buy", proposed_notional, day)
                    if proposed_notional + proposed_costs["total"] > cash + 1e-9:
                        return False
                    proposed_holdings = dict(holdings)
                    proposed_holdings[symbol] = current + proposed_quantity
                    return _portfolio_within_caps(
                        holdings=proposed_holdings,
                        cash=cash - proposed_notional - proposed_costs["total"],
                        prices=last_prices,
                        industries=last_industries,
                        boards=last_boards,
                        gross_position_max=daily_position_limit,
                        single_stock_max=single_stock_max,
                        industry_max=industry_max,
                        growth_star_max=growth_star_max,
                    )

                low, high = 0, requested_lots
                while low < high:
                    middle = (low + high + 1) // 2
                    if feasible_buy(middle):
                        low = middle
                    else:
                        high = middle - 1
                quantity = low * 100
                if quantity < 100:
                    continue
                notional = quantity * open_price
                costs = order_costs("buy", notional, day)
                cash -= notional + costs["total"]
                holdings[symbol] = current + quantity
                transaction_cost += costs["total"]
                board = last_boards.get(symbol)
                if board in board_transaction_cost:
                    board_transaction_cost[board] += costs["total"]
                turnover_notional += notional
                trade_count += 1
                if holdings[symbol] >= target_quantity:
                    completed_orders.add(symbol)

            for symbol in completed_orders:
                pending_desired.pop(symbol, None)
                pending_weights.pop(symbol, None)
            if not defer_unfilled_open_orders:
                pending_desired.clear()
                pending_weights.clear()

        holdings = {
            symbol: quantity for symbol, quantity in holdings.items() if quantity > 0
        }
        portfolio_value = cash + sum(
            quantity * last_prices.get(symbol, 0.0)
            for symbol, quantity in holdings.items()
        )
        gross_position = (
            sum(
                quantity * last_prices.get(symbol, 0.0)
                for symbol, quantity in holdings.items()
            )
            / portfolio_value
            if portfolio_value > 0
            else 0.0
        )
        board_values = {"main": 0.0, "growth": 0.0, "star": 0.0}
        for symbol, quantity in holdings.items():
            board = last_boards.get(symbol)
            if board in board_values:
                board_values[board] += quantity * last_prices.get(symbol, 0.0)
        portfolio_return = (
            portfolio_value / previous_value - 1.0 if previous_value > 0 else 0.0
        )
        benchmark_value = float(benchmark_open.loc[day])
        benchmark_return = (
            benchmark_value / previous_benchmark - 1.0 if previous_benchmark else 0.0
        )
        board_return_contribution = {
            board: (board_price_pnl[board] - board_transaction_cost[board])
            / previous_value
            if previous_value > 0
            else 0.0
            for board in board_values
        }
        board_start_exposure = {
            board: board_start_values[board] / previous_value
            if previous_value > 0
            else 0.0
            for board in board_values
        }
        records.append(
            {
                "date": day,
                "portfolio_value": portfolio_value,
                "portfolio_return": portfolio_return,
                "benchmark_return": benchmark_return,
                "cash": cash,
                "holding_count": len(holdings),
                "gross_position": gross_position,
                **{
                    f"board_exposure_{board}": value / portfolio_value
                    if portfolio_value > 0
                    else 0.0
                    for board, value in board_values.items()
                },
                **{
                    f"board_return_contribution_{board}": board_return_contribution[
                        board
                    ]
                    for board in board_values
                },
                **{
                    f"board_start_exposure_{board}": board_start_exposure[board]
                    for board in board_values
                },
                **{
                    f"board_excess_contribution_{board}": board_return_contribution[
                        board
                    ]
                    - board_start_exposure[board] * benchmark_return
                    for board in board_values
                },
                "cash_start_exposure": cash_start_exposure,
                "cash_excess_contribution": -cash_start_exposure * benchmark_return,
                "trade_count": trade_count,
                "transaction_cost": transaction_cost,
                "turnover": turnover_notional / value_before
                if value_before > 0
                else 0.0,
                "signal_date": signal_day,
                "pending_order_count": len(pending_desired),
            }
        )
        previous_value = portfolio_value
        previous_benchmark = benchmark_value
    daily = pd.DataFrame(records)
    return {
        "strategy_id": strategy_id,
        "daily": daily,
        "metrics": performance_metrics(daily),
    }
