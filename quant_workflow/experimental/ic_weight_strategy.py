#!/usr/bin/env python3
"""
IC-weighted portfolio optimization strategy for Qlib.

Instead of TopK dropout, uses model predictions to construct
a portfolio via mean-variance or simple IC-weighting.
"""
import numpy as np
import pandas as pd
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import BaseTradeDecision, TradeDecisionWO
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy


class ICWeightStrategy(TopkDropoutStrategy):
    """
    IC-weighted portfolio strategy.

    Instead of equal-weight TopK, weights stocks by their prediction
    score rank (higher prediction = higher weight).
    """

    def __init__(
        self,
        signal=None,
        topk=50,
        n_drop=5,
        **kwargs,
    ):
        super().__init__(
            signal=signal,
            topk=topk,
            n_drop=n_drop,
            **kwargs,
        )

    def generate_trade_decision(self, execute_result=None):
        """Generate trade decision with IC-weighted positions."""
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        pred_scores = self.signal.get_signal(start_time=trade_start_time, end_time=trade_end_time)

        if pred_scores is None or len(pred_scores) == 0:
            return TradeDecisionWO(order_list=[], strategy=self)

        # Get top-k stocks
        pred_sorted = pred_scores.sort_values(ascending=False)
        top_stocks = pred_sorted.head(self.topk)

        # IC weighting: rank-based weights (higher rank = higher weight)
        ranks = np.arange(len(top_stocks), 0, -1)
        weights = ranks / ranks.sum()

        # Create orders
        from qlib.backtest.exchange import Exchange
        from qlib.backtest.order import Order

        order_list = []
        for stock, weight in zip(top_stocks.index, weights):
            # Create a buy order with weight-based amount
            target_amount = weight * self._get_total_value()
            current_amount = self._get_position(stock)

            diff = target_amount - current_amount
            if abs(diff) > 0:
                direction = "buy" if diff > 0 else "sell"
                order = Order(
                    stock_id=stock,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=direction,
                    amount=abs(diff),
                )
                order_list.append(order)

        return TradeDecisionWO(order_list=order_list, strategy=self)

    def _get_total_value(self):
        """Get total portfolio value."""
        account = self.trade_account
        if account is None:
            return 10000000  # default 10M
        return account.total_value

    def _get_position(self, stock):
        """Get current position value for a stock."""
        account = self.trade_account
        if account is None:
            return 0
        pos = account.positions.get(stock)
        if pos is None:
            return 0
        return pos.amount * pos.price
