"""MiniQMT market-only provider for the existing DSA source manager."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS, normalize_stock_code, _is_etf_code
from .miniqmt_bridge import MiniQMTBridgeClient, MiniQMTError
from .realtime_types import RealtimeSource, UnifiedRealtimeQuote


def market_symbol(code: str) -> str:
    value = normalize_stock_code(code)
    if len(value) != 6 or not value.isdigit():
        raise MiniQMTError("MiniQMT currently supports Shanghai/Shenzhen stocks and ETFs")
    if value[0] in "56":
        return value + ".SH"
    if value[0] in "0123":
        return value + ".SZ"
    raise MiniQMTError("unsupported MiniQMT market; use an existing fallback provider")


def checked_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    required = {"date", "open", "high", "low", "close", "volume", "amount", "suspendFlag"}
    if not required.issubset(frame):
        raise MiniQMTError("MiniQMT daily fields are incomplete")
    if not pd.to_numeric(frame["suspendFlag"], errors="coerce").isin([0, 1]).all():
        raise MiniQMTError("invalid MiniQMT suspension flag")
    frame = frame.loc[pd.to_numeric(frame["suspendFlag"]) == 0].copy()
    if frame.empty:
        return frame
    fields = ["open", "high", "low", "close", "volume", "amount"]
    values = frame[fields].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(values).all().all() or (values[["open", "high", "low", "close"]] <= 0).any().any()
            or (values[["volume", "amount"]] < 0).any().any()
            or (values["high"] < values[["open", "close", "low"]].max(axis=1)).any()
            or (values["low"] > values[["open", "close", "high"]].min(axis=1)).any()):
        raise MiniQMTError("invalid MiniQMT OHLCVA")
    frame[fields] = values
    if pd.to_datetime(frame["date"], errors="coerce").isna().any() or frame["date"].duplicated().any():
        raise MiniQMTError("invalid MiniQMT daily dates")
    return frame.sort_values("date").reset_index(drop=True)


class MiniQMTFetcher(BaseFetcher):
    name = "MiniQMTFetcher"
    priority = -1

    def __init__(self, *, client=None):
        self.client = client or MiniQMTBridgeClient()

    def _fetch_raw_data(self, stock_code, start_date, end_date):
        symbol = market_symbol(stock_code)
        try:
            frames, _ = self.client.daily([symbol], start_date, end_date, adjustment="front")
            return checked_bars(frames[symbol])
        except (MiniQMTError, OSError, ValueError) as exc:
            raise DataFetchError(str(exc)) from exc

    def _normalize_data(self, df, stock_code):
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result["code"] = normalize_stock_code(stock_code)
        # SDK ``volume`` is lots for CN cash instruments; DSA's manager uses shares.
        result["volume"] = pd.to_numeric(result["volume"]) * 100
        result["pct_chg"] = pd.to_numeric(result["close"]).pct_change(fill_method=None) * 100
        result.loc[result.index[0], "pct_chg"] = 0.0
        return result[["code", *STANDARD_COLUMNS]]

    def get_realtime_quote(self, stock_code):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        symbol = market_symbol(stock_code)
        result = self.client.request("quotes", [symbol], today, today)["data"][symbol]

        def number(key):
            value = result.get(key)
            return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None
        price, last = number("lastPrice"), number("lastClose")
        if price is None or price <= 0:
            return None
        stamp = number("time")
        volume = number("volume")
        return UnifiedRealtimeQuote(
            code=normalize_stock_code(stock_code), source=RealtimeSource.MINIQMT,
            price=price, change_pct=(price / last - 1) * 100 if last and last > 0 else None,
            change_amount=price - last if last and last > 0 else None,
            volume=int(volume * 100) if volume is not None else None, amount=number("amount"),
            open_price=number("open"), high=number("high"), low=number("low"), pre_close=last,
            provider_timestamp=datetime.fromtimestamp(stamp / 1000, timezone.utc).isoformat() if stamp else None,
            market="cn", currency="CNY",
        )


class QuantQMTDataFetcher(BaseFetcher):
    """Ordinary DSA historical bars from the same committed local snapshots.

    This adapter does no acquisition. Updates remain an explicit separate
    operation, so a read cannot silently mix unpublished supplier responses.
    """
    name = "QuantQMTDataFetcher"
    priority = -2

    def __init__(self, *, quant_data_root=None):
        from pathlib import Path
        from src.services.quant_workflow_paths import quant_data_root as default_root
        self.root = Path(quant_data_root or default_root())

    def _fetch_raw_data(self, stock_code, start_date, end_date):
        from quant_qmt_data import DataContractError, bar_snapshot, read_bars
        dataset = "cn_etf_daily" if _is_etf_code(stock_code) else "ashare_daily_qfq"
        try:
            snapshot = bar_snapshot(dataset, quant_data_root=self.root, price_basis="qfq")
            frame = read_bars(dataset, stock_code, quant_data_root=self.root, price_basis="qfq",
                              expected_identity=snapshot["identity"], start=start_date, end=end_date)
        except (DataContractError, OSError, ValueError) as exc:
            raise DataFetchError("本机快照无法满足该证券请求；请显式更新或核对数据范围。") from exc
        frame.attrs.update(dataset_identity=snapshot["identity"], backend="quant_qmt_data",
                           volume_unit="units" if dataset == "cn_etf_daily" else "shares")
        return frame

    def _normalize_data(self, df, stock_code):
        result = df.copy()
        result["date"] = pd.to_datetime(result["date"])
        result["code"] = normalize_stock_code(stock_code)
        # The package already returns shares / ETF units; never multiply again.
        if "pctChg" in result:
            result["pct_chg"] = pd.to_numeric(result["pctChg"], errors="coerce")
        else:
            result["pct_chg"] = pd.to_numeric(result["close"]).pct_change(fill_method=None) * 100
            result.loc[result.index[0], "pct_chg"] = 0.0
        return result[["code", *STANDARD_COLUMNS]]
