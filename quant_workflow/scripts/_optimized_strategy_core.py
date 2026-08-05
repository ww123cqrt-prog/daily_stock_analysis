"""Reusable TopK portfolio strategy engine shared by CLI and web backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _equal_weights(holdings: set[object], position_pct: float) -> dict[object, float]:
    if not holdings or position_pct <= 0:
        return {}
    weight = float(position_pct) / len(holdings)
    return {symbol: weight for symbol in holdings}


def _weight_turnover(
    previous: dict[object, float], target: dict[object, float]
) -> float:
    symbols = previous.keys() | target.keys()
    return (
        sum(
            abs(target.get(symbol, 0.0) - previous.get(symbol, 0.0))
            for symbol in symbols
        )
        / 2
    )


class OptimizedStrategy:
    """TopK strategy with turnover, market-state and drawdown controls."""

    def __init__(
        self,
        topk: int = 50,
        n_drop: int = 5,
        cost_bps: float = 15,
        rebalance_freq: int = 1,
        turnover_cap: float = 0.20,
        trend_filter: bool = False,
        trend_ma: str = "ma200",
        trend_reduce_pct: float = 0.5,
        vol_sizing: bool = False,
        vol_reduce_threshold: float = 0.8,
        max_drawdown_stop: float = -0.08,
        max_dd_cooldown_days: int = 20,
        max_single_weight: float = 0.05,
        max_sector_weight: float = 0.30,
        sector_map: dict | None = None,
    ):
        self.topk = topk
        self.n_drop = n_drop
        self.cost_bps = cost_bps
        self.rebalance_freq = rebalance_freq
        self.turnover_cap = turnover_cap
        self.trend_filter = trend_filter
        self.trend_ma = trend_ma
        self.trend_reduce_pct = trend_reduce_pct
        self.vol_sizing = vol_sizing
        self.vol_reduce_threshold = vol_reduce_threshold
        self.max_drawdown_stop = max_drawdown_stop
        self.max_dd_cooldown_days = max_dd_cooldown_days
        self.max_single_weight = max_single_weight
        self.max_sector_weight = max_sector_weight
        self.sector_map = sector_map or {}

    def run(
        self,
        pred: pd.DataFrame,
        label: pd.DataFrame,
        market_indicators: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Run the strategy against saved model scores and realized labels."""
        pred_label = pd.concat([pred, label], axis=1).dropna()
        pred_label.columns = ["score", "label"]

        daily_results = []
        prev_holdings = set()
        prev_weights: dict[object, float] = {}
        cum_return = 1.0
        peak_return = 1.0
        in_cooldown = False
        cooldown_counter = 0
        rebalance_day = 0

        dates = sorted(pred_label.index.get_level_values("datetime").unique())

        for dt in dates:
            day_data = pred_label.loc[dt].copy()
            if len(day_data) < self.topk:
                continue

            position_pct = 1.0

            if market_indicators is not None and dt in market_indicators.index:
                mi = market_indicators.loc[dt]
                if self.trend_filter and not pd.isna(mi.get(self.trend_ma, np.nan)):
                    trend_signal = mi.get(
                        f"trend_{self.trend_ma.replace('ma', '')}", 1.0
                    )
                    if trend_signal < 0.5:
                        position_pct *= 1.0 - self.trend_reduce_pct

                if self.vol_sizing and not pd.isna(mi.get("vol_percentile", np.nan)):
                    vol_pct = mi["vol_percentile"]
                    if vol_pct > self.vol_reduce_threshold:
                        vol_factor = 1.0 - (
                            (vol_pct - self.vol_reduce_threshold)
                            / (1.0 - self.vol_reduce_threshold)
                            * 0.5
                        )
                        position_pct *= vol_factor

            if self.max_drawdown_stop < 0:
                current_dd = cum_return / peak_return - 1
                if current_dd < self.max_drawdown_stop and not in_cooldown:
                    in_cooldown = True
                    cooldown_counter = 0
                    position_pct = 0.0

                if in_cooldown:
                    cooldown_counter += 1
                    if (
                        current_dd > self.max_drawdown_stop * 0.5
                        or cooldown_counter > self.max_dd_cooldown_days
                    ):
                        in_cooldown = False
                    else:
                        position_pct = 0.0

            rebalance_day += 1
            if position_pct < 0.01:
                turnover = _weight_turnover(prev_weights, {})
                net_ret = -(turnover * (self.cost_bps / 10000))
                cum_return *= 1 + net_ret
                peak_return = max(peak_return, cum_return)
                daily_results.append(
                    {
                        "date": dt,
                        "return": net_ret,
                        "gross_return": 0.0,
                        "turnover": turnover,
                        "position_pct": 0.0,
                        "n_stocks": 0,
                    }
                )
                prev_holdings = set()
                prev_weights = {}
                continue

            if (
                self.rebalance_freq > 1
                and rebalance_day % self.rebalance_freq != 0
                and prev_holdings
            ):
                port_data = day_data.loc[day_data.index.isin(prev_holdings)]
                if len(port_data) > 0:
                    target_weights = _equal_weights(prev_holdings, position_pct)
                    turnover = _weight_turnover(prev_weights, target_weights)
                    port_ret = port_data["label"].mean() * position_pct
                    net_ret = port_ret - turnover * (self.cost_bps / 10000)
                    cum_return *= 1 + net_ret
                    peak_return = max(peak_return, cum_return)
                    daily_results.append(
                        {
                            "date": dt,
                            "return": net_ret,
                            "gross_return": port_data["label"].mean(),
                            "turnover": turnover,
                            "position_pct": position_pct,
                            "n_stocks": len(prev_holdings),
                        }
                    )
                    prev_weights = target_weights
                continue

            top_stocks = set(day_data.nlargest(self.topk, "score").index)
            new_stocks = top_stocks - prev_holdings
            removed_stocks = prev_holdings - top_stocks
            composition_turnover = (len(new_stocks) + len(removed_stocks)) / (
                2 * max(self.topk, 1)
            )

            if (
                self.turnover_cap
                and composition_turnover > self.turnover_cap
                and prev_holdings
            ):
                n_keep = int(self.topk * (1 - self.turnover_cap))
                keep = {symbol for symbol in prev_holdings if symbol in day_data.index}
                if len(keep) > n_keep:
                    keep_scores = day_data.loc[list(keep), "score"].nlargest(n_keep)
                    keep = set(keep_scores.index)
                n_new = self.topk - len(keep)
                candidates = day_data.drop(index=list(keep), errors="ignore").nlargest(
                    n_new, "score"
                )
                top_stocks = keep | set(candidates.index)

            port_data = day_data.loc[day_data.index.isin(top_stocks)]
            target_weights = _equal_weights(top_stocks, position_pct)
            turnover = _weight_turnover(prev_weights, target_weights)
            port_ret = port_data["label"].mean() * position_pct
            cost = turnover * (self.cost_bps / 10000)
            net_ret = port_ret - cost

            cum_return *= 1 + net_ret
            peak_return = max(peak_return, cum_return)
            daily_results.append(
                {
                    "date": dt,
                    "return": net_ret,
                    "gross_return": port_data["label"].mean(),
                    "turnover": turnover,
                    "position_pct": position_pct,
                    "n_stocks": len(top_stocks),
                }
            )
            prev_holdings = top_stocks
            prev_weights = target_weights

        if not daily_results:
            return pd.DataFrame(
                columns=[
                    "return",
                    "gross_return",
                    "turnover",
                    "position_pct",
                    "n_stocks",
                ]
            ).rename_axis("date")
        return pd.DataFrame(daily_results).set_index("date")
