#!/usr/bin/env python3
"""Build a governed, resumable mainland-listed ETF raw/qfq dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _file_lock import lock_file, unlock_file  # noqa: E402
from _paths import quant_data_root  # noqa: E402
from download_etf_momentum_data import (  # noqa: E402
    KNOWN_CORPORATE_ACTIONS,
    TENCENT_ADJUSTED_SOURCE_NAME,
    TENCENT_KLINE_URL,
    TENCENT_RAW_KLINE_URL,
    TENCENT_RAW_SOURCE_NAME,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    finalize_paired_price_snapshot,
    publish_snapshot_group,
    resolve_dataset,
    sha256_file,
    validate_dataset_snapshot,
)


DATASET_ID = "cn_etf_daily"
DEFAULT_START_DATE = "2004-01-01"
DEFAULT_PAGE_SIZE = 640
DEFAULT_LIMIT = 51
DEFAULT_REQUEST_INTERVAL = 0.3
DEFAULT_REQUEST_BUDGET = 5_000
DEFAULT_INCREMENTAL_REQUEST_BUDGET = 5_000
DEFAULT_INCREMENTAL_PAGE_SIZE = 64
DEFAULT_REQUIRED_CODES = ("512890", "513010", "510720", "159915", "513100")
DEFAULT_SNAPSHOT_RETENTION = 3
SNAPSHOT_RETENTION_POLICY = "immutable_history"
DEFAULT_INCREMENTAL_OVERLAP_DAYS = 14
DEFAULT_REPRESENTATIVE_TARGET_SIZE = 200
DEFAULT_MINIMUM_HISTORY_ROWS = 60
MIN_REQUEST_INTERVAL = 0.2
MIN_EXPANSION_REQUEST_INTERVAL = 0.3
MAX_CONSECUTIVE_FAILURES = 3
PRODUCER_VERSION = "2"
AFFINE_RESIDUAL_TOLERANCE = 0.0015
AFFINE_NEAR_TIE_TOLERANCE = 0.00015
AFFINE_FIT_WINDOWS = (10, 7, 5, 3, 2, 1)
REPRESENTATIVE_CATEGORY_TARGETS = {
    "core_broad": 32,
    "sector_theme": 73,
    "dividend_factor": 22,
    "hong_kong": 24,
    "overseas": 16,
    "bond": 21,
    "commodity": 7,
    "cash": 5,
}
ETF_MANAGER_TOKENS = (
    "华泰柏瑞",
    "易方达",
    "汇添富",
    "海富通",
    "博时",
    "南方",
    "华夏",
    "嘉实",
    "国泰",
    "广发",
    "招商",
    "富国",
    "鹏华",
    "银华",
    "华宝",
    "中银",
    "兴业",
    "摩根",
    "平安",
    "天弘",
    "工银瑞信",
    "华安",
    "建信",
    "景顺长城",
    "国联安",
    "国寿安保",
    "大成",
    "交银施罗德",
    "融通",
    "万家",
    "申万菱信",
    "长城",
    "长盛",
    "中欧",
    "中金",
    "中信保诚",
    "中信建投",
    "中邮",
    "东财",
    "西部利得",
    "永赢",
    "浦银安盛",
    "泰康",
    "国投瑞银",
    "华富",
    "诺安",
    "创金合信",
    "前海开源",
    "鹏扬",
)


class CnEtfDownloadError(RuntimeError):
    """The common ETF dataset could not satisfy its publication contract."""


class HistoricalRevisionDetected(CnEtfDownloadError):
    """A bounded overlap cannot safely reuse the frozen history."""


@dataclass
class RequestGate:
    interval: float
    budget: int
    calls: int = 0
    last_call: float = 0.0

    def call(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        if self.calls >= self.budget:
            raise CnEtfDownloadError("request budget exhausted")
        wait = self.interval - (time.monotonic() - self.last_call)
        if wait > 0:
            time.sleep(wait)
        self.calls += 1
        try:
            return function(*args, **kwargs)
        finally:
            self.last_call = time.monotonic()


def _as_date(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value).normalize()
    if pd.isna(result):
        raise ValueError(f"invalid date: {value}")
    return result


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_sina_symbol(value: object) -> tuple[str, str, str] | None:
    text = str(value or "").strip().lower()
    if len(text) != 8 or text[:2] not in {"sh", "sz"} or not text[2:].isdigit():
        return None
    market = text[:2]
    code = text[2:]
    return f"{market.upper()}{code}", code, market


def _market_for_code(code: str) -> str:
    if code.startswith("1"):
        return "sz"
    if code.startswith("5"):
        return "sh"
    raise CnEtfDownloadError(f"cannot infer ETF market for code {code}")


def build_etf_universe(
    sina: pd.DataFrame,
    ths: pd.DataFrame,
    *,
    through: str,
    recent_listing_days: int = 10,
) -> pd.DataFrame:
    """Reconcile current Sina ETFs with recently traded THS-only listings."""

    if recent_listing_days < 0:
        raise ValueError("recent_listing_days must be nonnegative")
    required_sina = {"代码", "名称"}
    required_ths = {"基金代码", "基金名称", "最新-交易日"}
    if not required_sina.issubset(sina.columns):
        raise CnEtfDownloadError("Sina ETF catalog columns are incomplete")
    if not required_ths.issubset(ths.columns):
        raise CnEtfDownloadError("THS ETF catalog columns are incomplete")

    records: dict[str, dict[str, Any]] = {}
    for row in sina.to_dict(orient="records"):
        parsed = _canonical_sina_symbol(row.get("代码"))
        if parsed is None:
            continue
        symbol, code, market = parsed
        amount = pd.to_numeric(row.get("成交额"), errors="coerce")
        records[symbol] = {
            "symbol": symbol,
            "code": code,
            "market": market,
            "name": str(row.get("名称") or code).strip(),
            "sina_code": f"{market}{code}",
            "in_sina": True,
            "in_ths": False,
            "ths_latest_trade_date": None,
            "sina_amount": float(amount) if pd.notna(amount) else 0.0,
            "fund_type": None,
        }

    through_date = _as_date(through)
    recent_cutoff = through_date - pd.Timedelta(days=recent_listing_days)
    for row in ths.to_dict(orient="records"):
        code = str(row.get("基金代码") or "").strip().zfill(6)
        if len(code) != 6 or not code.isdigit():
            continue
        latest = pd.to_datetime(row.get("最新-交易日"), errors="coerce")
        market = _market_for_code(code)
        symbol = f"{market.upper()}{code}"
        if symbol in records:
            records[symbol]["in_ths"] = True
            records[symbol]["ths_latest_trade_date"] = (
                latest.date().isoformat() if pd.notna(latest) else None
            )
            records[symbol]["fund_type"] = str(row.get("基金类型") or "").strip() or None
            continue
        if pd.isna(latest) or latest < recent_cutoff or latest > through_date:
            continue
        records[symbol] = {
            "symbol": symbol,
            "code": code,
            "market": market,
            "name": str(row.get("基金名称") or code).strip(),
            "sina_code": None,
            "in_sina": False,
            "in_ths": True,
            "ths_latest_trade_date": latest.date().isoformat(),
            "sina_amount": 0.0,
            "fund_type": str(row.get("基金类型") or "").strip() or None,
        }

    if not records:
        raise CnEtfDownloadError("ETF catalog reconciliation returned no symbols")
    return pd.DataFrame(records.values()).sort_values("symbol").reset_index(drop=True)


def select_trial_universe(
    universe: pd.DataFrame,
    *,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
) -> pd.DataFrame:
    """Select a liquid trial set while retaining user-required ETF codes."""

    if limit < 1 or limit > 2_000:
        raise ValueError("ETF universe limit must be between 1 and 2000")
    required = tuple(dict.fromkeys(str(code).zfill(6) for code in required_codes))
    if len(required) > limit:
        raise ValueError("required ETF count exceeds the universe limit")
    by_code = universe.drop_duplicates("code", keep="first").set_index("code")
    missing = [code for code in required if code not in by_code.index]
    if missing:
        raise CnEtfDownloadError(
            "required ETFs are missing from the current catalog: " + ", ".join(missing)
        )

    ranked = universe.copy()
    ranked["sina_amount"] = pd.to_numeric(
        ranked.get("sina_amount"), errors="coerce"
    ).fillna(0.0)
    ranked = ranked.sort_values(
        ["sina_amount", "symbol"], ascending=[False, True], kind="stable"
    )
    ordinary = ranked.loc[~ranked["code"].isin(required)].head(limit - len(required))
    mandatory = ranked.loc[ranked["code"].isin(required)]
    selected = pd.concat([ordinary, mandatory], ignore_index=True)
    if len(selected) != limit or selected["symbol"].duplicated().any():
        raise CnEtfDownloadError("ETF trial universe could not reach the requested size")
    selected["selection_reason"] = "top_turnover"
    selected.loc[selected["code"].isin(required), "selection_reason"] = "required"
    selected = selected.sort_values(
        ["sina_amount", "symbol"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    return selected


def classify_representative_category(name: object, fund_type: object = None) -> str:
    """Assign one mutually exclusive research category from public metadata."""

    text = unicodedata.normalize("NFKC", str(name or "")).upper().replace(" ", "")
    type_text = unicodedata.normalize("NFKC", str(fund_type or "")).upper()
    if any(
        keyword in text
        for keyword in (
            "货币",
            "现金",
            "日利",
            "添益",
            "保证金",
            "快线",
            "金利",
            "理财金",
        )
    ):
        return "cash"
    if "债" in text or "债券" in type_text or "短融" in text:
        return "bond"
    if any(
        keyword in text
        for keyword in (
            "黄金",
            "白银",
            "豆粕",
            "商品",
            "原油",
            "能源化工",
            "有色金属期货",
        )
    ):
        return "commodity"
    if any(
        keyword in text
        for keyword in ("恒生", "港股", "港股通", "香港", "H股", "中概")
    ):
        return "hong_kong"
    if any(
        keyword in text
        for keyword in (
            "纳指",
            "纳斯达克",
            "标普",
            "日经",
            "东证",
            "德国",
            "法国",
            "美国",
            "美股",
            "沙特",
            "印度",
            "东南亚",
            "全球",
            "海外",
            "欧洲",
            "韩国",
        )
    ):
        return "overseas"
    if any(
        keyword in text
        for keyword in (
            "红利",
            "低波",
            "价值",
            "质量",
            "成长",
            "自由现金流",
            "现金流",
            "股息",
            "基本面",
        )
    ):
        return "dividend_factor"
    if any(
        keyword in text
        for keyword in (
            "沪深300",
            "中证500",
            "中证800",
            "中证1000",
            "中证2000",
            "中证A500",
            "A500",
            "中证A50",
            "A50",
            "上证50",
            "上证180",
            "上证380",
            "深证100",
            "深证50",
            "创业板",
            "科创50",
            "科创100",
            "科创200",
            "国证2000",
            "北证50",
            "中小100",
            "深证成指",
            "上证指数",
        )
    ):
        return "core_broad"
    return "sector_theme"


def representative_tracking_key(name: object, code: object = None) -> str:
    """Normalize a display name enough to avoid adding another same-index ETF."""

    text = unicodedata.normalize("NFKC", str(name or "")).upper()
    text = text.replace("交易型开放式指数证券投资基金", "")
    for token in ETF_MANAGER_TOKENS:
        text = text.replace(token.upper(), "")
    text = text.replace("ETF", "").replace("基金", "")
    normalized = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", text)
    return normalized or str(code or "").zfill(6)


def select_representative_universe(
    universe: pd.DataFrame,
    current_universe: pd.DataFrame,
    *,
    target_size: int = DEFAULT_REPRESENTATIVE_TARGET_SIZE,
    category_targets: dict[str, int] | None = None,
    minimum_turnover: float = 1_000_000.0,
    excluded_symbols: tuple[str, ...] = (),
    required_symbols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Retain the frozen pool and fill deterministic category quotas by liquidity."""

    if category_targets is None and target_size != DEFAULT_REPRESENTATIVE_TARGET_SIZE:
        raise ValueError(
            "default representative quotas support target_size=200; "
            "provide explicit category_targets for another registered contract"
        )
    targets = dict(category_targets or REPRESENTATIVE_CATEGORY_TARGETS)
    if target_size < 1 or target_size > 500:
        raise ValueError("representative ETF target_size must be between 1 and 500")
    if sum(targets.values()) != target_size or any(value < 0 for value in targets.values()):
        raise ValueError("representative ETF category targets must sum to target_size")
    unknown = sorted(set(targets) - set(REPRESENTATIVE_CATEGORY_TARGETS))
    if unknown:
        raise ValueError("unknown representative ETF categories: " + ", ".join(unknown))
    required = {"symbol", "code", "market", "name"}
    if universe.empty or not required.issubset(universe.columns):
        raise CnEtfDownloadError("ETF catalog is incomplete for representative selection")
    if current_universe.empty or not required.issubset(current_universe.columns):
        raise CnEtfDownloadError("current ETF universe is incomplete")
    current = current_universe.drop_duplicates("symbol", keep="first").copy()
    normalized_exclusions = {
        str(symbol).strip().upper() for symbol in excluded_symbols if str(symbol).strip()
    }
    normalized_required = {
        str(symbol).strip().upper() for symbol in required_symbols if str(symbol).strip()
    }
    if normalized_required & normalized_exclusions:
        raise CnEtfDownloadError(
            "required ETF symbols cannot also be excluded"
        )
    current_symbol_set = set(current["symbol"].astype(str))
    if normalized_exclusions & current_symbol_set:
        raise CnEtfDownloadError(
            "representative exclusions cannot remove a current ETF symbol"
        )
    if len(current) > target_size:
        raise CnEtfDownloadError("representative expansion cannot shrink the current universe")

    ranked = universe.drop_duplicates("symbol", keep="first").copy()
    optional_defaults = {
        "sina_amount": 0.0,
        "in_sina": False,
        "in_ths": False,
        "fund_type": None,
        "sina_code": None,
        "ths_latest_trade_date": None,
    }
    for column, default in optional_defaults.items():
        if column not in ranked.columns:
            ranked[column] = pd.Series(default, index=ranked.index)
    ranked["sina_amount"] = pd.to_numeric(
        ranked["sina_amount"], errors="coerce"
    ).fillna(0.0)
    ranked["in_sina"] = ranked["in_sina"].fillna(False).astype(bool)
    ranked["in_ths"] = ranked["in_ths"].fillna(False).astype(bool)
    ranked["representative_category"] = [
        classify_representative_category(row.name, getattr(row, "fund_type", None))
        for row in ranked.itertuples(index=False)
    ]
    ranked["tracking_key"] = [
        representative_tracking_key(row.name, row.code)
        for row in ranked.itertuples(index=False)
    ]
    ranked["catalog_pair"] = ranked["in_sina"] & ranked["in_ths"]
    ranked = ranked.sort_values(
        ["catalog_pair", "sina_amount", "symbol"],
        ascending=[False, False, True],
        kind="stable",
    )

    catalog_by_symbol = ranked.set_index("symbol", drop=False)
    selected_records: list[dict[str, Any]] = []
    selected_symbols: set[str] = set()
    used_keys: set[str] = set()
    counts = {category: 0 for category in targets}
    for row in current.itertuples(index=False):
        symbol = str(row.symbol)
        source = (
            catalog_by_symbol.loc[symbol].to_dict()
            if symbol in catalog_by_symbol.index
            else row._asdict()
        )
        category = classify_representative_category(
            source.get("name"), source.get("fund_type")
        )
        if category not in targets:
            raise CnEtfDownloadError(
                f"current ETF category is not covered by the selection contract: {symbol}"
            )
        key = representative_tracking_key(source.get("name"), source.get("code"))
        source.update(
            {
                "representative_category": category,
                "tracking_key": key,
                "selection_reason": "existing_snapshot",
            }
        )
        selected_records.append(source)
        selected_symbols.add(symbol)
        used_keys.add(key)
        counts[category] += 1

    missing_required = sorted(normalized_required - selected_symbols)
    for symbol in missing_required:
        if symbol not in catalog_by_symbol.index:
            raise CnEtfDownloadError(
                f"required ETF is missing from the current catalog: {symbol}"
            )
        source = catalog_by_symbol.loc[symbol].to_dict()
        if not bool(source.get("in_sina")) or float(source.get("sina_amount") or 0.0) < float(
            minimum_turnover
        ):
            raise CnEtfDownloadError(
                f"required ETF does not satisfy current trading evidence: {symbol}"
            )
        category = str(source.get("representative_category") or "")
        if category not in targets or counts[category] >= targets[category]:
            raise CnEtfDownloadError(
                f"required ETF category quota is unavailable: {symbol}"
            )
        key = str(source.get("tracking_key") or "")
        source["selection_reason"] = "user_required"
        selected_records.append(source)
        selected_symbols.add(symbol)
        used_keys.add(key)
        counts[category] += 1

    overfilled = {
        category: counts[category] - target
        for category, target in targets.items()
        if counts[category] > target
    }
    if overfilled:
        details = ", ".join(f"{key}=+{value}" for key, value in overfilled.items())
        raise CnEtfDownloadError(
            "current ETF universe already exceeds representative quotas: " + details
        )

    eligible = ranked.loc[
        ranked["in_sina"]
        & (ranked["sina_amount"] >= float(minimum_turnover))
        & ~ranked["symbol"].astype(str).isin(selected_symbols)
        & ~ranked["symbol"].astype(str).isin(normalized_exclusions)
    ]
    for category, target in targets.items():
        needed = target - counts[category]
        if needed <= 0:
            continue
        added = 0
        for row in eligible.loc[
            eligible["representative_category"] == category
        ].itertuples(index=False):
            symbol = str(row.symbol)
            key = str(row.tracking_key)
            if symbol in selected_symbols or key in used_keys:
                continue
            record = row._asdict()
            record["selection_reason"] = f"representative_{category}"
            selected_records.append(record)
            selected_symbols.add(symbol)
            used_keys.add(key)
            counts[category] += 1
            added += 1
            if added == needed:
                break
        if added != needed:
            raise CnEtfDownloadError(
                f"representative ETF category {category} needs {needed} additions but only {added} passed selection"
            )

    selected = pd.DataFrame(selected_records)
    if len(selected) != target_size or selected["symbol"].astype(str).duplicated().any():
        raise CnEtfDownloadError("representative ETF universe did not reach the exact target")
    final_counts = selected["representative_category"].value_counts().to_dict()
    if any(int(final_counts.get(category, 0)) != target for category, target in targets.items()):
        raise CnEtfDownloadError("representative ETF category quotas are inconsistent")
    selected = selected.drop(
        columns=["catalog_pair", "selection_rank"], errors="ignore"
    )
    selected.insert(0, "selection_rank", range(1, len(selected) + 1))
    return selected.reset_index(drop=True)


def fetch_etf_catalog(
    *,
    through: str,
    request_gate: RequestGate,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch only the two public ETF metadata catalogs used for selection."""

    import akshare as ak

    sina = request_gate.call(ak.fund_etf_category_sina, symbol="ETF基金")
    ths = request_gate.call(ak.fund_etf_category_ths, symbol="ETF")
    if sina is None or sina.empty or ths is None or ths.empty:
        raise CnEtfDownloadError("ETF catalog source returned no rows")
    reconciled = build_etf_universe(sina, ths, through=through)
    return reconciled, {
        "sina_count": int(len(sina)),
        "ths_count": int(len(ths)),
        "reconciled_count": int(len(reconciled)),
    }


def fetch_etf_universe(
    *,
    through: str,
    request_gate: RequestGate,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reconciled, summary = fetch_etf_catalog(
        through=through,
        request_gate=request_gate,
    )
    selected = select_trial_universe(
        reconciled,
        limit=limit,
        required_codes=required_codes,
    )
    return selected, {
        **summary,
        "selected_count": int(len(selected)),
        "selection_method": "sina_turnover_top_n_plus_required_codes",
        "required_codes": list(required_codes),
    }


def _request_tencent_page(
    *,
    session: Any,
    request_gate: RequestGate,
    url: str,
    ticker: str,
    response_key: str,
    param: str,
    timeout: float,
    retries: int,
) -> tuple[list[list[object]], str]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = request_gate.call(
                session.get,
                url,
                params={"param": param},
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            ticker_data = data.get(ticker) if isinstance(data, dict) else None
            response_keys = (
                ("qfqday", "day") if response_key == "qfqday" else ("day",)
            )
            selected_key = next(
                (
                    key
                    for key in response_keys
                    if isinstance(ticker_data, dict)
                    and isinstance(ticker_data.get(key), list)
                ),
                None,
            )
            rows = ticker_data.get(selected_key) if selected_key else None
            if payload.get("code") != 0 or not isinstance(rows, list) or not selected_key:
                raise CnEtfDownloadError(
                    f"Tencent response is invalid: {payload.get('msg') if isinstance(payload, dict) else 'not an object'}"
                )
            return rows, selected_key
        except Exception as exc:
            last_error = exc
            if getattr(getattr(exc, 'response', None), 'status_code', None) in {400, 401, 403, 404, 501}:
                break
            if attempt + 1 < retries:
                time.sleep(float(attempt + 1))
    raise CnEtfDownloadError(
        f"Tencent request failed: {type(last_error).__name__}: {last_error}"
    ) from last_error


def fetch_tencent_history(
    symbol: str,
    metadata: dict[str, str],
    *,
    adjustment: str,
    start_date: str,
    end_date: str,
    session: Any,
    request_gate: RequestGate,
    page_size: int = DEFAULT_PAGE_SIZE,
    retries: int = 2,
    timeout: float = 20.0,
) -> pd.DataFrame:
    if adjustment not in {"raw", "qfq"}:
        raise ValueError("Tencent adjustment must be raw or qfq")
    if page_size < 1 or page_size > DEFAULT_PAGE_SIZE:
        raise ValueError(f"Tencent page_size must be between 1 and {DEFAULT_PAGE_SIZE}")
    if retries < 1 or retries > 2:
        raise ValueError("Tencent retries must be between 1 and 2")
    start = _as_date(start_date)
    end = _as_date(end_date)
    if start > end:
        raise ValueError("start_date must not be after end_date")

    ticker = f"{metadata['market']}{metadata['code']}"
    response_key = "day" if adjustment == "raw" else "qfqday"
    suffix = "" if adjustment == "raw" else ",qfq"
    url = TENCENT_RAW_KLINE_URL if adjustment == "raw" else TENCENT_KLINE_URL
    page_end = end
    collected: list[list[object]] = []
    response_keys_seen: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for _page in range(32):
        param = (
            f"{ticker},day,{start.date().isoformat()},"
            f"{page_end.date().isoformat()},{page_size}{suffix}"
        )
        rows, response_key_used = _request_tencent_page(
            session=session,
            request_gate=request_gate,
            url=url,
            ticker=ticker,
            response_key=response_key,
            param=param,
            timeout=timeout,
            retries=retries,
        )
        response_keys_seen.add(response_key_used)
        normalized_rows = [row[:6] for row in rows if isinstance(row, list) and len(row) >= 6]
        if not normalized_rows:
            if collected:
                break
            raise CnEtfDownloadError(f"Tencent returned no {adjustment} rows for {symbol}")
        dates = pd.to_datetime([row[0] for row in normalized_rows], errors="coerce")
        if dates.isna().any():
            raise CnEtfDownloadError(f"Tencent returned invalid dates for {symbol}")
        oldest = pd.Timestamp(dates.min()).normalize()
        if previous_oldest is not None and oldest >= previous_oldest:
            raise CnEtfDownloadError(f"Tencent pagination did not move backwards for {symbol}")
        collected.extend(normalized_rows)
        if oldest <= start or len(normalized_rows) < page_size:
            break
        previous_oldest = oldest
        page_end = oldest - pd.Timedelta(days=1)
    else:
        raise CnEtfDownloadError(f"Tencent pagination limit exceeded for {symbol}")

    frame = pd.DataFrame(
        collected,
        columns=["date", "open", "close", "high", "low", "volume"],
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    frame = frame.loc[(frame["date"] >= start) & (frame["date"] <= end)].copy()
    frame = frame.loc[(frame["open"] > 0) & (frame["close"] > 0)]
    frame = frame.sort_values("date", kind="stable").drop_duplicates(
        "date", keep="last"
    ).reset_index(drop=True)
    if frame.empty:
        raise CnEtfDownloadError(f"no Tencent {adjustment} rows remain for {symbol}")
    frame.insert(0, "symbol", symbol)
    frame.insert(1, "name", metadata["name"])
    frame["source"] = (
        TENCENT_RAW_SOURCE_NAME
        if adjustment == "raw"
        else TENCENT_ADJUSTED_SOURCE_NAME
    )
    if adjustment == "qfq":
        frame["adjustment"] = "qfq"
        frame["provider_response_key"] = ",".join(sorted(response_keys_seen))
    return frame


def fetch_tencent_pair(
    symbol: str,
    metadata: dict[str, str],
    *,
    start_date: str,
    end_date: str,
    session: Any,
    request_gate: RequestGate,
    page_size: int = DEFAULT_PAGE_SIZE,
    retries: int = 2,
    timeout: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = fetch_tencent_history(
        symbol,
        metadata,
        adjustment="raw",
        start_date=start_date,
        end_date=end_date,
        session=session,
        request_gate=request_gate,
        page_size=page_size,
        retries=retries,
        timeout=timeout,
    )
    qfq = fetch_tencent_history(
        symbol,
        metadata,
        adjustment="qfq",
        start_date=start_date,
        end_date=end_date,
        session=session,
        request_gate=request_gate,
        page_size=page_size,
        retries=retries,
        timeout=timeout,
    )
    build_paired_adjustments(symbol, raw, qfq)
    return raw, qfq


def fetch_eastmoney_pair(
    symbol: str, metadata: dict[str, str], *, start_date: str, end_date: str,
    session: Any, request_gate: RequestGate,
    page_size: int = DEFAULT_INCREMENTAL_PAGE_SIZE,
    retries: int = 2, timeout: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a bounded raw/QFQ window from the legacy EastMoney endpoint."""
    start, end = _as_date(start_date), _as_date(end_date)
    limit = max(page_size, len(pd.bdate_range(start, end)))
    if start > end or limit > 640:
        raise CnEtfDownloadError("EastMoney incremental window exceeds its bounded limit")
    frames = []
    for fqt, kind in (("0", "raw"), ("1", "qfq")):
        params = {
            "secid": ("1." if symbol.startswith("SH") else "0.") + metadata["code"],
            "klt": "101", "fqt": fqt, "beg": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"), "lmt": str(limit),
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        }
        last_error = None
        for attempt in range(retries):
            response = None
            try:
                try:
                    response = request_gate.call(
                        session.get, "https://push2his.eastmoney.com/api/qt/stock/kline/get",
                        params=params, timeout=timeout,
                    )
                    response.raise_for_status()
                    payload = response.json()
                finally:
                    if response is not None:
                        primary = sys.exc_info()[1]
                        try:
                            response.close()
                        except Exception as cleanup:
                            if primary is None:
                                raise
                            primary.add_note(f"response cleanup failed: {cleanup}")
                data = payload.get("data") or {}
                rows = data.get("klines")
                if payload.get("rc") != 0 or not rows:
                    raise CnEtfDownloadError(f"no EastMoney {kind} rows for {symbol}")
                if data.get("code", metadata["code"]) != metadata["code"]:
                    raise CnEtfDownloadError(f"EastMoney symbol differs for {symbol}")
                values = [row.split(",") for row in rows]
                frame = pd.DataFrame(values, columns=[
                    "date", "open", "close", "high", "low", "volume", "amount",
                    "amplitude", "pctChg", "change", "turnover",
                ])[['date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
                if (frame["date"].duplicated().any()
                        or not frame["date"].between(start, end).all()):
                    raise CnEtfDownloadError(f"EastMoney dates are invalid for {symbol}")
                for field in ("open", "high", "low", "close", "volume", "amount"):
                    frame[field] = pd.to_numeric(frame[field], errors="raise")
                frame.insert(0, "symbol", symbol)
                frame.insert(1, "name", metadata["name"])
                frame["source"] = f"eastmoney.etf_daily_{kind}"
                if kind == "qfq":
                    frame["adjustment"] = "qfq"
                    frame["provider_response_key"] = "eastmoney_fqt_1"
                frames.append(frame.sort_values("date", kind="stable").reset_index(drop=True))
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < retries:
                    time.sleep(float(attempt + 1))
        else:
            raise CnEtfDownloadError(f"EastMoney {kind} request failed for {symbol}: {last_error}") from last_error
    raw, qfq = frames
    build_paired_adjustments(symbol, raw, qfq)
    if not np.array_equal(raw["volume"].to_numpy(), qfq["volume"].to_numpy()):
        raise CnEtfDownloadError(f"EastMoney raw/qfq volume differs for {symbol}")
    return raw, qfq


_BAOSTOCK_WINDOW_SCRIPT = r'''
import json, sys
import baostock as bs
request = json.loads(sys.argv[1])
try:
    login = bs.login()
    if str(login.error_code) != "0":
        raise RuntimeError("Baostock login failed: " + str(login.error_code))
    result = bs.query_history_k_data_plus(**request)
    rows = []
    while str(result.error_code) == "0" and result.next():
        rows.append(result.get_row_data())
        if len(rows) > 640:
            raise RuntimeError("Baostock response exceeded bounded row limit")
    payload = dict(request=request, error_code=str(result.error_code),
                   error_msg=result.error_msg, fields=result.fields, rows=rows)
finally:
    primary = sys.exc_info()[1]
    try:
        bs.logout()
    except Exception as cleanup:
        if primary is None:
            raise
        primary.add_note("Baostock logout failed: " + str(cleanup))
print("ETF_RESULT:" + json.dumps(payload, ensure_ascii=False))
'''


def _read_baostock_window(request: dict[str, str], *, timeout: float) -> dict[str, Any]:
    """Bound login, all SDK pages, logout and EOF loops in one owned child."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", _BAOSTOCK_WINDOW_SCRIPT, json.dumps(request)],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
        records = [line.removeprefix("ETF_RESULT:") for line in result.stdout.splitlines()
                   if line.startswith("ETF_RESULT:")]
        if len(records) != 1 or len(records[0]) > 1_000_000:
            raise ValueError("missing or oversized SDK result")
        return json.loads(records[0])
    except (subprocess.SubprocessError, ValueError) as exc:
        raise CnEtfDownloadError(f"Baostock bounded request failed: {exc}") from exc


def fetch_baostock_pair(
    symbol: str, metadata: dict[str, str], *, start_date: str, end_date: str,
    session: Any, request_gate: RequestGate,
    page_size: int = DEFAULT_INCREMENTAL_PAGE_SIZE,
    retries: int = 2, timeout: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only a bounded raw/QFQ tail; preserve the SDK's native values."""
    start, end = _as_date(start_date), _as_date(end_date)
    if start > end or len(pd.bdate_range(start, end)) > 640:
        raise CnEtfDownloadError("Baostock incremental window exceeds its bounded limit")
    code = f"{metadata['market']}.{metadata['code']}"
    if code != symbol[:2].lower() + "." + symbol[2:]:
        raise CnEtfDownloadError(f"Baostock metadata differs for {symbol}")
    fields = "date,code,open,high,low,close,volume,amount,tradestatus"
    frames, responses = [], []
    for flag, kind in (("3", "raw"), ("2", "qfq")):
        request = dict(code=code, fields=fields, start_date=start.date().isoformat(),
                       end_date=end.date().isoformat(), frequency="d", adjustflag=flag)
        for attempt in range(retries):
            try:
                payload = request_gate.call(_read_baostock_window, request, timeout=timeout)
                break
            except CnEtfDownloadError:
                if attempt + 1 == retries:
                    raise
        if (payload.get("request") != request or payload.get("error_code") != "0"
                or payload.get("fields") != fields.split(",")):
            raise CnEtfDownloadError(f"Baostock response identity differs for {symbol}")
        responses.append(payload)
        try:
            frame = pd.DataFrame(payload["rows"], columns=payload["fields"])
            if frame.empty or set(frame["code"]) != {code}:
                raise ValueError("missing rows or wrong code")
            frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
            if frame["date"].duplicated().any() or not frame["date"].between(start, end).all():
                raise ValueError("invalid dates")
            if not set(frame["tradestatus"]).issubset({"0", "1"}):
                raise ValueError("unknown trade status")
            # SDK suspension placeholders are evidence, never newly manufactured bars.
            frame = frame.loc[frame["tradestatus"] == "1"].copy()
            if frame.empty:
                raise ValueError("no actual trading rows")
            for field in ("open", "high", "low", "close", "volume", "amount"):
                frame[field] = pd.to_numeric(frame[field], errors="raise")
            if not np.isfinite(frame[["open", "high", "low", "close", "volume", "amount"]]).all().all():
                raise ValueError("nonfinite bar values")
            if (frame[["volume", "amount"]] < 0).any().any():
                raise ValueError("negative volume or amount")
            frame["native_volume"] = frame["volume"]
            frame["volume"] = np.floor(frame["volume"] / 100.0 + 0.5)
            frame["volume_unit"] = "lot_100_shares_rounded"
            frame["symbol"], frame["name"] = symbol, metadata["name"]
            frame["source"] = f"baostock.etf_daily_{kind}"
            if kind == "qfq":
                frame["provider_response_key"] = "baostock_adjustflag_2"
            frames.append(frame.drop(columns=["code"]).sort_values("date").reset_index(drop=True))
        except (KeyError, TypeError, ValueError) as exc:
            raise CnEtfDownloadError(f"invalid Baostock {kind} response for {symbol}: {exc}") from exc
    raw, qfq = frames
    if raw["date"].tolist() != qfq["date"].tolist() or not np.array_equal(raw["volume"], qfq["volume"]):
        raise CnEtfDownloadError(f"Baostock raw/qfq keys or volume differ for {symbol}")
    build_paired_adjustments(symbol, raw, qfq)
    raw.attrs["source_receipt"] = {
        "provider": "baostock", "responses": responses,
        "volume_rule": "floor(shares / 100 + 0.5) to legacy whole lots",
    }
    return raw, qfq


def build_paired_adjustments(
    symbol: str,
    raw: pd.DataFrame,
    adjusted: pd.DataFrame,
) -> pd.DataFrame:
    """Derive and validate one affine OHLC transform for every trade date.

    Tencent represents share splits with a multiplicative scale and some cash
    distributions with an additive offset. Both are needed to reproduce all
    four adjusted OHLC fields from the paired raw response.
    """

    raw_frame = raw.copy()
    qfq_frame = adjusted.copy()
    raw_frame["date"] = pd.to_datetime(raw_frame["date"], errors="coerce").dt.normalize()
    qfq_frame["date"] = pd.to_datetime(qfq_frame["date"], errors="coerce").dt.normalize()
    if (
        raw_frame["date"].isna().any()
        or qfq_frame["date"].isna().any()
        or raw_frame["date"].duplicated().any()
        or qfq_frame["date"].duplicated().any()
        or set(raw_frame["date"]) != set(qfq_frame["date"])
    ):
        raise DataContractError(f"raw/qfq dates differ for {symbol}")
    fields = ("open", "high", "low", "close")
    for field in fields:
        if field not in raw_frame or field not in qfq_frame:
            raise DataContractError(f"raw/qfq {field} field is missing for {symbol}")
    paired = raw_frame[["date", *fields]].merge(
        qfq_frame[["date", *fields]],
        on="date",
        how="inner",
        validate="one_to_one",
        suffixes=("_raw", "_qfq"),
    ).sort_values("date", kind="stable")

    raw_values = np.column_stack(
        [pd.to_numeric(paired[f"{field}_raw"], errors="coerce") for field in fields]
    )
    qfq_values = np.column_stack(
        [pd.to_numeric(paired[f"{field}_qfq"], errors="coerce") for field in fields]
    )
    if (
        not np.isfinite(raw_values).all()
        or not np.isfinite(qfq_values).all()
        or (raw_values <= 0).any()
        or (qfq_values <= 0).any()
    ):
        raise DataContractError(f"raw/qfq OHLC values are invalid for {symbol}")

    def residual(values: np.ndarray, targets: np.ndarray, scale: float, offset: float) -> float:
        return float(np.max(np.abs(targets - (values * scale + offset))))

    def fit_window(start: int, stop: int) -> tuple[float, float, str] | None:
        values = raw_values[start:stop].reshape(-1)
        targets = qfq_values[start:stop].reshape(-1)
        candidates: list[tuple[float, int, float, float, str]] = []

        def add_candidate(scale: float, offset: float, priority: int, method: str) -> None:
            if not np.isfinite(scale) or not np.isfinite(offset) or scale <= 0:
                return
            candidates.append(
                (
                    residual(values, targets, float(scale), float(offset)),
                    priority,
                    float(scale),
                    float(offset),
                    method,
                )
            )

        add_candidate(1.0, 0.0, 0, "identity")
        add_candidate(1.0, float(np.median(targets - values)), 1, "offset")
        add_candidate(float(np.median(targets / values)), 0.0, 2, "scale")
        simple = [
            candidate
            for candidate in candidates
            if candidate[0] <= AFFINE_RESIDUAL_TOLERANCE
        ]
        if simple:
            best_residual = min(candidate[0] for candidate in simple)
            near_best = [
                candidate
                for candidate in simple
                if candidate[0] <= best_residual + AFFINE_NEAR_TIE_TOLERANCE
            ]
            _error, _priority, scale, offset, method = min(
                near_best, key=lambda candidate: candidate[1]
            )
            return scale, offset, method

        if float(np.ptp(values)) <= 1e-9:
            return None
        design = np.column_stack([values, np.ones(len(values))])
        scale, offset = np.linalg.lstsq(design, targets, rcond=None)[0]
        affine_residual = residual(values, targets, float(scale), float(offset))
        if (
            not np.isfinite(scale)
            or not np.isfinite(offset)
            or scale <= 0
            or affine_residual > AFFINE_RESIDUAL_TOLERANCE
        ):
            return None
        return float(scale), float(offset), "affine"

    latest_residual = residual(raw_values[-1], qfq_values[-1], 1.0, 0.0)
    if latest_residual > AFFINE_RESIDUAL_TOLERANCE:
        raise DataContractError(f"raw/qfq latest transform is not normalized for {symbol}")

    row_count = len(paired)
    assigned_scales = np.empty(row_count, dtype=float)
    assigned_offsets = np.empty(row_count, dtype=float)
    assigned_residuals = np.empty(row_count, dtype=float)
    reverse_segment_ids = np.empty(row_count, dtype=int)
    assigned_methods: list[str] = [""] * row_count
    scale = 1.0
    offset = 0.0
    method = "identity"
    reverse_segment_id = 0

    for position in range(row_count - 1, -1, -1):
        row_residual = residual(raw_values[position], qfq_values[position], scale, offset)
        if row_residual > AFFINE_RESIDUAL_TOLERANCE:
            fitted: tuple[float, float, str] | None = None
            attempted_starts: set[int] = set()
            for window in AFFINE_FIT_WINDOWS:
                start = max(0, position - window + 1)
                if start in attempted_starts:
                    continue
                attempted_starts.add(start)
                fitted = fit_window(start, position + 1)
                if fitted is not None:
                    break
            if fitted is None:
                raise DataContractError(
                    f"raw/qfq affine OHLC relationship is invalid for {symbol}"
                )
            scale, offset, method = fitted
            reverse_segment_id += 1
            row_residual = residual(
                raw_values[position], qfq_values[position], scale, offset
            )
        assigned_scales[position] = scale
        assigned_offsets[position] = offset
        assigned_residuals[position] = row_residual
        reverse_segment_ids[position] = reverse_segment_id
        assigned_methods[position] = method

    if float(np.max(assigned_residuals)) > AFFINE_RESIDUAL_TOLERANCE:
        raise DataContractError(f"raw/qfq affine OHLC relationship is invalid for {symbol}")
    segment_ids = int(reverse_segment_ids.max()) - reverse_segment_ids
    result = pd.DataFrame(
        {
            "date": paired["date"].to_numpy(),
            "factor": assigned_scales,
            "offset": assigned_offsets,
            "segment_id": segment_ids,
            "segment_method": assigned_methods,
            "max_ohlc_residual": assigned_residuals,
            "factor_source": "derived_same_provider_raw_qfq_piecewise_affine_ohlc",
            "adjustment_formula": "qfq_ohlc=raw_ohlc*factor+offset",
        }
    )
    return result


def _corporate_actions_from_adjustments(
    adjustments_by_symbol: dict[str, pd.DataFrame],
    *,
    provider: str,
) -> pd.DataFrame:
    known_dates = {
        (item["symbol"], pd.Timestamp(item["resumption_date"]))
        for item in KNOWN_CORPORATE_ACTIONS
    }
    rows: list[dict[str, Any]] = []
    for symbol, adjustments in adjustments_by_symbol.items():
        ordered = adjustments.sort_values("date", kind="stable").reset_index(drop=True)
        scales = pd.to_numeric(ordered["factor"], errors="coerce")
        offsets = pd.to_numeric(ordered["offset"], errors="coerce")
        scale_ratio = scales / scales.shift(1)
        offset_change = offsets - offsets.shift(1)
        if "segment_id" in ordered.columns:
            segments = pd.to_numeric(ordered["segment_id"], errors="coerce")
            changed = segments.ne(segments.shift(1)) & segments.shift(1).notna()
        else:
            changed = ((scale_ratio - 1.0).abs() >= 0.005) | (
                offset_change.abs() >= 0.005
            )
        for position in ordered.index[changed.fillna(False)]:
            effective = pd.Timestamp(ordered.loc[position, "date"])
            if (symbol, effective) in known_dates:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "record_date": pd.NaT,
                    "ex_date": effective,
                    "effective_trade_date": effective,
                    "resumption_date": effective,
                    "action_type": "provider_adjustment_event",
                    "ratio": float(scale_ratio.loc[position]),
                    "offset_change": float(offset_change.loc[position]),
                    "factor_before": float(scales.loc[position - 1]),
                    "factor_after": float(scales.loc[position]),
                    "offset_before": float(offsets.loc[position - 1]),
                    "offset_after": float(offsets.loc[position]),
                    "evidence_method": "derived_from_same_provider_raw_qfq_affine_pair",
                    "source": f"{provider}.paired_raw_qfq",
                    "source_url": None,
                }
            )
    for item in KNOWN_CORPORATE_ACTIONS:
        symbol = str(item["symbol"])
        if symbol not in adjustments_by_symbol:
            continue
        adjustments = adjustments_by_symbol[symbol].sort_values("date", kind="stable")
        effective = pd.Timestamp(item["resumption_date"])
        before = adjustments.loc[adjustments["date"] < effective]
        after = adjustments.loc[adjustments["date"] >= effective]
        rows.append(
            {
                "symbol": symbol,
                "record_date": pd.Timestamp(item["record_date"]),
                "ex_date": pd.Timestamp(item["split_date"]),
                "effective_trade_date": effective,
                "resumption_date": effective,
                "action_type": item["type"],
                "ratio": 5.0,
                "offset_change": (
                    float(after.iloc[0]["offset"] - before.iloc[-1]["offset"])
                    if not before.empty and not after.empty
                    else None
                ),
                "factor_before": (
                    float(before.iloc[-1]["factor"]) if not before.empty else None
                ),
                "factor_after": (
                    float(after.iloc[0]["factor"]) if not after.empty else None
                ),
                "offset_before": (
                    float(before.iloc[-1]["offset"]) if not before.empty else None
                ),
                "offset_after": (
                    float(after.iloc[0]["offset"]) if not after.empty else None
                ),
                "evidence_method": "announcement_and_same_provider_raw_qfq_affine_pair",
                "source": item["source"],
                "source_url": item["source_url"],
            }
        )
    columns = [
        "symbol",
        "record_date",
        "ex_date",
        "effective_trade_date",
        "resumption_date",
        "action_type",
        "ratio",
        "offset_change",
        "factor_before",
        "factor_after",
        "offset_before",
        "offset_after",
        "evidence_method",
        "source",
        "source_url",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    for field in ("record_date", "ex_date", "effective_trade_date", "resumption_date"):
        frame[field] = pd.to_datetime(frame[field], errors="coerce")
    return frame.sort_values(
        ["symbol", "effective_trade_date"], kind="stable"
    ).reset_index(drop=True)


def _universe_hash(universe: pd.DataFrame) -> str:
    stable = universe.sort_values("symbol").fillna("").to_dict(orient="records")
    encoded = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        try:
            lock_file(handle, exclusive=True, blocking=False)
            acquired = True
        except BlockingIOError as exc:
            raise CnEtfDownloadError("another common ETF update is already running") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            if acquired:
                unlock_file(handle)
        finally:
            handle.close()


def _cache_symbol_complete(cache: Path, symbol: str) -> bool:
    paths = {
        key: cache / key / f"{symbol}.parquet"
        for key in ("raw", "qfq", "factors")
    }
    if not all(path.is_file() for path in paths.values()):
        return False
    try:
        raw = pd.read_parquet(paths["raw"])
        qfq = pd.read_parquet(paths["qfq"])
        recorded = pd.read_parquet(paths["factors"])
        if "provider_response_key" not in qfq.columns:
            return False
        required_factor_columns = {
            "date",
            "factor",
            "offset",
            "segment_id",
            "segment_method",
            "factor_source",
            "adjustment_formula",
        }
        if not required_factor_columns.issubset(recorded.columns):
            return False
        rebuilt = build_paired_adjustments(symbol, raw, qfq)
    except Exception:
        return False
    return (
        not raw.empty
        and len(recorded) == len(rebuilt)
        and recorded["date"].tolist() == rebuilt["date"].tolist()
        and pd.Series(recorded["factor"]).round(10).tolist()
        == pd.Series(rebuilt["factor"]).round(10).tolist()
        and pd.Series(recorded["offset"]).round(10).tolist()
        == pd.Series(rebuilt["offset"]).round(10).tolist()
        and recorded["segment_id"].tolist() == rebuilt["segment_id"].tolist()
        and recorded["segment_method"].tolist()
        == rebuilt["segment_method"].tolist()
        and recorded["factor_source"].tolist()
        == rebuilt["factor_source"].tolist()
        and recorded["adjustment_formula"].tolist()
        == rebuilt["adjustment_formula"].tolist()
    )


def _upgrade_cached_adjustment_evidence(cache: Path, symbol: str) -> bool:
    """Rebuild obsolete factor evidence from an already downloaded raw/qfq pair."""

    raw_path = cache / "raw" / f"{symbol}.parquet"
    qfq_path = cache / "qfq" / f"{symbol}.parquet"
    if not raw_path.is_file() or not qfq_path.is_file():
        return False
    try:
        raw = pd.read_parquet(raw_path)
        qfq = pd.read_parquet(qfq_path)
        if "provider_response_key" not in qfq.columns:
            return False
        rebuilt = build_paired_adjustments(symbol, raw, qfq)
        _atomic_parquet(rebuilt, cache / "factors" / f"{symbol}.parquet")
    except Exception:
        return False
    return True


def _normalize_component_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    component: str,
) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    if frame.empty or not required.issubset(frame.columns):
        raise DataContractError(f"{component} history is incomplete for {symbol}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(
        normalized["date"], errors="coerce"
    ).dt.normalize()
    if normalized["date"].isna().any() or normalized["date"].duplicated().any():
        raise DataContractError(f"{component} dates are invalid for {symbol}")
    if set(normalized["symbol"].astype(str)) != {symbol}:
        raise DataContractError(f"{component} symbol is inconsistent for {symbol}")
    return normalized.sort_values("date", kind="stable").reset_index(drop=True)


def _overlap_values_match(
    current: pd.DataFrame,
    incoming: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> bool:
    fields = ("open", "high", "low", "close", "volume")
    left = current.set_index("date").loc[dates, fields]
    right = incoming.set_index("date").loc[dates, fields]
    left_values = left.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right_values = right.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(
        np.allclose(
            left_values,
            right_values,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=True,
        )
    )


def merge_incremental_pair(
    symbol: str,
    current_raw: pd.DataFrame,
    current_qfq: pd.DataFrame,
    incoming_raw: pd.DataFrame,
    incoming_qfq: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Keep raw history and align an affine QFQ basis using a bounded overlap."""

    old_raw = _normalize_component_frame(
        current_raw, symbol=symbol, component="current raw"
    )
    old_qfq = _normalize_component_frame(
        current_qfq, symbol=symbol, component="current qfq"
    )
    new_raw = _normalize_component_frame(
        incoming_raw, symbol=symbol, component="incremental raw"
    )
    new_qfq = _normalize_component_frame(
        incoming_qfq, symbol=symbol, component="incremental qfq"
    )
    if old_raw["date"].tolist() != old_qfq["date"].tolist():
        raise DataContractError(f"current raw/qfq dates differ for {symbol}")
    if new_raw["date"].tolist() != new_qfq["date"].tolist():
        raise DataContractError(f"incremental raw/qfq dates differ for {symbol}")

    old_dates = pd.DatetimeIndex(old_raw["date"])
    incoming_dates = pd.DatetimeIndex(new_raw["date"])
    overlap = old_dates.intersection(incoming_dates)
    if overlap.empty:
        raise HistoricalRevisionDetected(
            f"incremental response has no overlap for {symbol}"
        )
    if not _overlap_values_match(old_raw, new_raw, overlap):
        raise HistoricalRevisionDetected(
            f"provider revised raw history for {symbol}"
        )
    rebase = None
    if not _overlap_values_match(old_qfq, new_qfq, overlap):
        if len(overlap) < 2 or old_dates.max() not in overlap:
            raise HistoricalRevisionDetected(f"insufficient qfq overlap for {symbol}")
        fields = ("open", "high", "low", "close")
        values = old_qfq.set_index("date").loc[overlap, fields].to_numpy(dtype=float).ravel()
        targets = new_qfq.set_index("date").loc[overlap, fields].to_numpy(dtype=float).ravel()
        candidates = [(1.0, float(np.median(targets - values)), "offset"),
                      (float(np.median(targets / values)), 0.0, "scale")]
        if np.isfinite(values).all() and float(np.ptp(values)) > 1e-9:
            scale, offset = np.linalg.lstsq(np.column_stack([values, np.ones(len(values))]), targets, rcond=None)[0]
            candidates.append((float(scale), float(offset), "affine"))
        for scale, offset, method in candidates:
            error = float(np.max(np.abs(targets - (values * scale + offset))))
            if (np.isfinite(scale) and np.isfinite(offset) and scale > 0
                    and np.isfinite(error) and error <= AFFINE_RESIDUAL_TOLERANCE):
                rebase = {"scale": scale, "offset": offset, "method": method,
                          "overlap_start": overlap.min().date().isoformat(),
                          "overlap_end": overlap.max().date().isoformat(),
                          "overlap_sessions": len(overlap), "max_ohlc_residual": error}
                break
        if rebase is None:
            raise HistoricalRevisionDetected(f"provider revised qfq history nonuniformly for {symbol}")
        old_qfq = old_qfq.copy()
        for field in fields:
            old_qfq[field] = pd.to_numeric(old_qfq[field]) * rebase["scale"] + rebase["offset"]
        rebased = old_qfq[list(fields)].to_numpy(dtype=float)
        if not np.isfinite(rebased).all() or (rebased <= 0).any():
            raise HistoricalRevisionDetected(f"qfq rebase produces invalid historical prices for {symbol}")

    old_end = old_dates.max()
    append_raw = new_raw.loc[new_raw["date"] > old_end]
    append_qfq = new_qfq.loc[new_qfq["date"] > old_end]
    if append_raw["date"].tolist() != append_qfq["date"].tolist():
        raise DataContractError(f"incremental raw/qfq additions differ for {symbol}")
    if append_raw.empty:
        if rebase:
            old_qfq.attrs["historical_rebase"] = rebase
        return old_raw, old_qfq, 0
    combined_raw = pd.concat([old_raw, append_raw], ignore_index=True)
    combined_qfq = pd.concat([old_qfq, append_qfq], ignore_index=True)
    if rebase:
        combined_qfq.attrs["historical_rebase"] = rebase
    return combined_raw, combined_qfq, int(len(append_raw))


def _incremental_identity(symbol, metadata, base_manifest_sha256, start_date, end_date, provider):
    identity = {"base_manifest_sha256": base_manifest_sha256, "symbol": symbol,
                "metadata": metadata, "start": start_date, "end": end_date,
                "provider": provider, "rules": "affine-bounded-increment-v1"}
    if provider == "baostock":
        identity["volume_rule"] = "native-shares-to-half-up-100-share-lots-v2"
    if provider == "miniqmt":
        identity["source_rule"] = "bounded-cash-share-front-sdk-lots-v1"
    return identity


def _cached_incremental_pair(
    cache: Path, symbol: str, metadata: dict[str, str], *,
    base_manifest_sha256: str, start_date: str, end_date: str,
    session: Any, request_gate: RequestGate, retries: int, timeout: float,
    provider: str = "tencent",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retain a verified bounded acquisition across preparation and publication."""
    identity = _incremental_identity(symbol, metadata, base_manifest_sha256, start_date, end_date, provider)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = cache / key
    marker = folder / "COMPLETE.json"
    receipt_path = folder / "SOURCE_RESPONSE.json"
    paths = {kind: folder / kind / f"{symbol}.parquet" for kind in ("raw", "qfq", "factors")}

    def attributed(raw: pd.DataFrame, qfq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        for kind, frame in (("raw", raw), ("qfq", qfq)):
            source_id = f"{provider}:normalized-{kind}:{sha256_file(paths[kind])}"
            if "source_id" not in frame:
                frame["source_id"] = source_id
            else:
                frame["source_id"] = frame["source_id"].fillna(source_id)
        return raw, qfq
    if marker.is_file():
        try:
            complete = json.loads(marker.read_text())
            if complete["identity"] != identity or any(
                not path.is_file() or sha256_file(path) != complete["sha256"][kind]
                for kind, path in paths.items()
            ) or not _cache_symbol_complete(folder, symbol):
                raise DataContractError(f"incremental cache changed for {symbol}")
            if provider in {"baostock", "miniqmt"} and (
                not receipt_path.is_file()
                or sha256_file(receipt_path) != complete.get("source_receipt_sha256")
            ):
                raise DataContractError(f"incremental source response changed for {symbol}")
            raw, qfq = attributed(pd.read_parquet(paths["raw"]), pd.read_parquet(paths["qfq"]))
            if provider == "miniqmt":
                from miniqmt_etf_source import verified_response
                receipt = json.loads(receipt_path.read_text())
                _, response = verified_response(cache, receipt)
                raw.attrs["source_receipt"] = receipt
                from data_provider.miniqmt_fetcher import market_symbol
                request = response["request"]
                if (request.get("operation") != "update_pair" or request.get("adjustment") != "none"
                        or request.get("start") != start_date.replace("-", "")
                        or request.get("end") != end_date.replace("-", "")
                        or market_symbol(symbol) not in request.get("symbols", [])):
                    raise DataContractError("MiniQMT source response request changed")
                observed = response["data"][market_symbol(symbol)]
                raw.attrs["miniqmt_actions"] = observed["actions"]
                raw.attrs["miniqmt_action_through"] = observed.get("action_history_through", end_date)
            return raw, qfq
        except (OSError, ValueError, KeyError) as exc:
            raise DataContractError(f"incremental cache is invalid for {symbol}") from exc
    try:
        fetch_pair = session.fetch_pair if provider == "miniqmt" else {
            "tencent": fetch_tencent_pair, "eastmoney": fetch_eastmoney_pair,
            "baostock": fetch_baostock_pair}[provider]
        raw, qfq = fetch_pair(
            symbol, metadata, start_date=start_date, end_date=end_date, session=session,
            request_gate=request_gate, page_size=DEFAULT_INCREMENTAL_PAGE_SIZE,
            retries=retries, timeout=timeout,
        )
        if _as_date(raw["date"].max()) < _as_date(end_date):
            raise CnEtfDownloadError(f"{provider} response for {symbol} ends at {raw['date'].max()}, before requested {end_date}")
        factors = build_paired_adjustments(symbol, raw, qfq)
    except DataContractError as exc:
        raise CnEtfDownloadError(f"incoming raw/qfq pair is invalid for {symbol}: {exc}") from exc
    except Exception as exc:
        from data_provider.miniqmt_bridge import MiniQMTError
        if provider != "miniqmt" or not isinstance(exc, MiniQMTError):
            raise
        raise CnEtfDownloadError(f"MiniQMT input unavailable for {symbol}: {exc}") from exc
    for kind, frame in (("raw", raw), ("qfq", qfq), ("factors", factors)):
        _atomic_parquet(frame, paths[kind])
    complete = {"identity": identity, "sha256": {kind: sha256_file(path) for kind, path in paths.items()}}
    if provider in {"baostock", "miniqmt"}:
        atomic_write_json(receipt_path, raw.attrs["source_receipt"])
        complete["source_receipt_sha256"] = sha256_file(receipt_path)
    atomic_write_json(marker, complete)
    return attributed(raw, qfq)


def _link_or_copy(source: Path, destination: Path) -> None:
    """Copy a snapshot file without sharing its inode with another snapshot.

    Snapshot retention relies on the previous snapshot being an independent
    rollback point. Hard links make a later accidental in-place write mutate
    both the current snapshot and its backup, so ordinary updates always use a
    real copy here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_incremental_status(root: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(
        root / "runtime" / "status" / "cn_etf_daily_update_latest.json",
        payload,
    )


def _sync_root_manifest(root: Path) -> None:
    from sync_data_manifest import sync_data_manifest

    sync_data_manifest(quant_data_root=root)


def prune_snapshot_history(
    dataset_root: Path,
    *,
    keep: int = DEFAULT_SNAPSHOT_RETENTION,
    preferred_backup_snapshot_id: str | None = None,
) -> list[str]:
    """Keep the current complete snapshot and a configurable immutable history."""

    if keep < 1:
        raise ValueError("ETF snapshot retention must be at least 1")
    dataset_root = Path(dataset_root).expanduser()
    if dataset_root.is_symlink():
        raise CnEtfDownloadError("ETF dataset root cannot be a symlink")
    dataset_root = dataset_root.resolve()
    pointer_path = dataset_root / "LATEST.json"
    snapshots_root = dataset_root / "snapshots"
    if pointer_path.is_symlink() or snapshots_root.is_symlink():
        raise CnEtfDownloadError("ETF snapshot retention paths cannot be symlinks")
    if not pointer_path.is_file() or not snapshots_root.is_dir():
        raise CnEtfDownloadError("ETF snapshot retention state is incomplete")
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CnEtfDownloadError("ETF latest pointer is unreadable") from exc
    current = str(pointer.get("snapshot_id") or "")
    records: list[tuple[str, str, Path]] = []
    for path in sorted(snapshots_root.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise CnEtfDownloadError("ETF snapshots directory contains an unsafe entry")
        manifest_path = path / "DATASET_MANIFEST.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CnEtfDownloadError(
                f"ETF snapshot manifest is unreadable: {path.name}"
            ) from exc
        if str(manifest.get("snapshot_id") or "") != path.name:
            raise CnEtfDownloadError(
                f"ETF snapshot identity is inconsistent: {path.name}"
            )
        records.append(
            (str(manifest.get("source_retrieved_at") or ""), path.name, path)
        )
    if current not in {snapshot_id for _, snapshot_id, _ in records}:
        raise CnEtfDownloadError("ETF latest pointer target is missing")

    ordered = sorted(records, key=lambda item: (item[0], item[1]), reverse=True)
    retained = {current}
    if preferred_backup_snapshot_id:
        preferred = str(preferred_backup_snapshot_id)
        available = {snapshot_id for _, snapshot_id, _ in records}
        if preferred == current or preferred not in available:
            raise CnEtfDownloadError("preferred ETF backup snapshot is unavailable")
        retained.add(preferred)
    for _, snapshot_id, _ in ordered:
        if len(retained) >= keep:
            break
        if snapshot_id != current:
            retained.add(snapshot_id)
    removed: list[str] = []
    for _, snapshot_id, path in records:
        if snapshot_id in retained:
            continue
        if path.is_symlink():
            raise CnEtfDownloadError("ETF snapshot retention path became a symlink")
        try:
            path.resolve().relative_to(snapshots_root.resolve())
        except ValueError as exc:
            raise CnEtfDownloadError(
                "ETF snapshot retention path escapes the dataset"
            ) from exc
        shutil.rmtree(path)
        removed.append(snapshot_id)
    return sorted(removed)


def _post_publish_housekeeping(
    root: Path,
    dataset_root: Path,
    *,
    preferred_backup_snapshot_id: str | None = None,
    sync_catalog: bool = True,
) -> tuple[list[str], list[str]]:
    """Run non-transactional housekeeping without hiding a committed publish."""

    removed: list[str] = []
    warnings: list[str] = []
    try:
        removed = prune_snapshot_history(
            dataset_root,
            keep=DEFAULT_SNAPSHOT_RETENTION,
            preferred_backup_snapshot_id=preferred_backup_snapshot_id,
        )
    except Exception as exc:
        warnings.append(f"retention cleanup failed: {type(exc).__name__}: {exc}")
    if sync_catalog:
        try:
            _sync_root_manifest(root)
        except Exception as exc:
            warnings.append(f"manifest sync failed: {type(exc).__name__}: {exc}")
    for warning in warnings:
        print(f"ETF snapshot was published; {warning}", file=sys.stderr)
    return removed, warnings


def update_cn_etf_snapshot(
    *,
    quant_data_dir: Path | None = None,
    end_date: str | None = None,
    overlap_days: int = DEFAULT_INCREMENTAL_OVERLAP_DAYS,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_INCREMENTAL_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    generated_at: str | None = None,
    base_snapshot: Path | None = None,
    symbols: list[str] | None = None,
    publish: bool = True,
    keep_failed_history: bool | None = None,
    provider: str = "miniqmt",
) -> Path:
    """Reuse the frozen ETF snapshot and fetch only its missing date window."""

    if provider not in {"miniqmt", "tencent", "eastmoney", "baostock"}:
        raise ValueError("unsupported incremental ETF provider")
    if keep_failed_history is None:
        keep_failed_history = provider == "miniqmt"
    if overlap_days < 1 or overlap_days > 60:
        raise ValueError("overlap_days must be between 1 and 60")
    if request_interval < MIN_REQUEST_INTERVAL:
        raise ValueError(f"request_interval must be at least {MIN_REQUEST_INTERVAL}")
    if request_budget < 1 or request_budget > 10_000:
        raise ValueError("incremental request_budget must be between 1 and 10000")
    if retries < 1 or retries > 2:
        raise ValueError("retries must be between 1 and 2")

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    requested_through = _as_date(
        end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat()
    )
    # The updater is commonly invoked by a daily job on Saturday or Sunday.
    # Those dates cannot have new A-share ETF bars; normalize them before
    # calculating missing symbols so a weekend does not trigger 51 requests.
    through = requested_through
    if through.weekday() >= 5:
        through = through - pd.Timedelta(days=through.weekday() - 4)
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    started = time.monotonic()

    with _run_lock(lock_path):
        if base_snapshot is None:
            resolved = resolve_dataset(DATASET_ID, quant_data_root=root, require_basis="qfq")
            current, manifest = resolved.snapshot_path, resolved.manifest
        else:
            current = Path(base_snapshot).expanduser().resolve()
            try:
                current.relative_to(root / "parquet/etf_1d/cn_etf_daily/snapshots")
            except ValueError as exc:
                raise DataContractError("ETF base snapshot is outside its dataset") from exc
            manifest = validate_dataset_snapshot(current, verify_hashes=True)
            if manifest.get("dataset_id") != DATASET_ID or manifest.get("snapshot_id") != current.name:
                raise DataContractError("ETF base snapshot identity differs")
        if manifest.get("data_kind") != "paired_daily_bars":
            raise DataContractError("cn_etf_daily is not a paired raw/qfq snapshot")
        history_provider = (manifest.get("source") or {}).get("provider")
        if history_provider not in {"miniqmt", "tencent", "eastmoney", "baostock", "tickflow", "mixed"}:
            raise DataContractError("incremental update requires an identified frozen history provider")
        base_manifest_sha256 = sha256_file(current / "DATASET_MANIFEST.json")

        universe_path = current / "ETF_UNIVERSE.parquet"
        if not universe_path.is_file():
            raise DataContractError("current ETF universe is unavailable")
        universe = pd.read_parquet(universe_path)
        required_universe_columns = {"symbol", "code", "market", "name"}
        if universe.empty or not required_universe_columns.issubset(universe.columns):
            raise DataContractError("current ETF universe is invalid")
        if universe["symbol"].astype(str).duplicated().any():
            raise DataContractError("current ETF universe contains duplicate symbols")

        qfq_index = pd.read_parquet(current / str(manifest["file_index_path"]))
        indexed_symbols = set(qfq_index["symbol"].astype(str))
        universe_symbols = set(universe["symbol"].astype(str))
        metadata_only_symbols: list[str] = []
        if indexed_symbols != universe_symbols:
            if base_snapshot is not None and history_provider == "tickflow" and indexed_symbols.issubset(universe_symbols):
                metadata_only_symbols = sorted(universe_symbols - indexed_symbols)
                universe = universe.loc[universe["symbol"].astype(str).isin(indexed_symbols)].reset_index(drop=True)
                universe_symbols = indexed_symbols
            else:
                raise DataContractError("current ETF universe and snapshot symbols differ")
        selected = universe_symbols if symbols is None else set(symbols)
        if not selected or not selected.issubset(universe_symbols):
            raise ValueError("requested ETF symbols must belong to the frozen universe")
        if publish and selected != universe_symbols:
            raise ValueError("a symbol-scoped preparation cannot publish the whole ETF dataset")
        last_by_symbol = {
            str(row.symbol): _as_date(str(row.last_date))
            for row in qfq_index.itertuples(index=False)
        }
        metadata_by_symbol = {str(row.symbol): row for row in universe.itertuples(index=False)}
        first_by_symbol = {str(row.symbol): _as_date(str(row.first_date)) for row in qfq_index.itertuples(index=False)} if provider == "miniqmt" else {}
        calendar_bridge_calls = 0
        if provider == "miniqmt" and through > max(last_by_symbol.values()):
            frozen = False
            for s in sorted(selected):
                if last_by_symbol[s] >= through:
                    continue
                row = metadata_by_symbol[s]
                first = first_by_symbol[s]
                start = max(first, last_by_symbol[s] - pd.Timedelta(days=overlap_days)).date().isoformat()
                metadata = {"code": str(row.code).zfill(6), "market": str(row.market).lower(), "name": str(row.name)}
                identity = _incremental_identity(s, metadata, base_manifest_sha256, start, through.date().isoformat(), provider)
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                if (root / "cache/cn-etf-incremental" / key / "COMPLETE.json").is_file():
                    frozen = True
                    break
            # This only avoids a redundant calendar RPC. Every cache and its
            # frozen source request is still validated before any adoption.
            if not frozen:
                from miniqmt_etf_source import latest_session
                through = latest_session(through)
                calendar_bridge_calls = 1
        missing_symbols = sorted(
            symbol for symbol, last_date in last_by_symbol.items() if symbol in selected and last_date < through
        )
        if not missing_symbols:
            _write_incremental_status(
                root,
                {
                    "status": "already_current",
                    "dataset_id": DATASET_ID,
                    "snapshot_id": current.name,
                    "requested_through": requested_through.date().isoformat(),
                    "through": through.date().isoformat(),
                    "requests_used": calendar_bridge_calls,
                    "calendar_bridge_calls": calendar_bridge_calls,
                    "symbols_requested": 0,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "generated_at": timestamp,
                },
            )
            return current

        gate = RequestGate(interval=request_interval, budget=request_budget)
        import requests

        if provider == "miniqmt":
            from miniqmt_etf_source import MiniQMTEtfSource
            windows = {
                s: (max(first_by_symbol[s],
                        last_by_symbol[s] - pd.Timedelta(days=overlap_days)).date().isoformat(), through.date().isoformat())
                for s in missing_symbols
            }
            session = MiniQMTEtfSource(root / "cache/cn-etf-incremental", windows)
        else:
            session = requests.Session()
            session.trust_env = False
        run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        snapshot_id = f"cn-etf-{provider}-incremental-{run_id}"
        staging_run = root / "staging" / f"cn-etf-incremental-{run_id}"
        stage = staging_run / snapshot_id
        changed_symbols: list[str] = []
        targeted_full_refresh_symbols: list[str] = []
        historical_qfq_rebases: dict[str, dict[str, Any]] = {}
        symbol_update_failures: dict[str, dict[str, str]] = {}
        rows_added_by_symbol: dict[str, int] = {}
        factors_by_symbol: dict[str, pd.DataFrame] = {}
        miniqmt_receipts: dict[str, dict] = {}
        miniqmt_action_history = dict((manifest.get("source") or {}).get("miniqmt_action_history") or {})
        keep_prepared = False
        try:
            for symbol in sorted(universe_symbols):
                current_raw_path = current / "raw" / f"{symbol}.parquet"
                current_qfq_path = current / "qfq" / f"{symbol}.parquet"
                current_factor_path = current / "factors" / f"{symbol}.parquet"
                if not all(
                    path.is_file()
                    for path in (current_raw_path, current_qfq_path, current_factor_path)
                ):
                    raise DataContractError(
                        f"current ETF component is missing for {symbol}"
                    )

                if symbol not in missing_symbols:
                    _link_or_copy(current_raw_path, stage / "raw" / current_raw_path.name)
                    _link_or_copy(current_qfq_path, stage / "qfq" / current_qfq_path.name)
                    _link_or_copy(
                        current_factor_path,
                        stage / "factors" / current_factor_path.name,
                    )
                    factors_by_symbol[symbol] = pd.read_parquet(current_factor_path)
                    continue

                current_raw = pd.read_parquet(current_raw_path)
                current_qfq = pd.read_parquet(current_qfq_path)
                row = metadata_by_symbol[symbol]
                metadata = {"code": str(row.code).zfill(6), "market": str(row.market).lower(), "name": str(row.name)}
                for kind, frame in (("raw", current_raw), ("qfq", current_qfq)):
                    if "symbol" not in frame:
                        frame.insert(0, "symbol", symbol)
                    if "name" not in frame:
                        frame.insert(1, "name", metadata["name"])
                    if "source" not in frame:
                        frame["source"] = history_provider
                    if "source_id" not in frame:
                        frame["source_id"] = f"{history_provider}:{base_manifest_sha256}"
                    else:
                        frame["source_id"] = frame["source_id"].fillna(f"{history_provider}:{base_manifest_sha256}")
                    if kind == "qfq" and "provider_response_key" not in frame:
                        frame["provider_response_key"] = f"frozen_{history_provider}_snapshot"
                for frame in (current_raw, current_qfq):
                    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
                old_start = _as_date(current_raw["date"].min())
                old_end = _as_date(current_raw["date"].max())
                overlap_start = max(
                    old_start,
                    old_end - pd.Timedelta(days=overlap_days),
                )
                try:
                    incoming_raw, incoming_qfq = _cached_incremental_pair(
                        root / "cache" / "cn-etf-incremental", symbol, metadata,
                        base_manifest_sha256=base_manifest_sha256,
                        start_date=overlap_start.date().isoformat(), end_date=through.date().isoformat(),
                        session=session, request_gate=gate, retries=retries, timeout=timeout,
                        provider=provider,
                    )
                    if provider == "miniqmt":
                        from data_provider.miniqmt_bridge import action_history_digest
                        previous = miniqmt_action_history.get(symbol)
                        if previous and action_history_digest(incoming_raw.attrs["miniqmt_actions"], previous["through"], previous.get("since")) != previous["sha256"]:
                            raise HistoricalRevisionDetected(f"MiniQMT revised historical cash/share events for {symbol}; retain old files for separate repair")
                    updated_raw, updated_qfq, rows_added = merge_incremental_pair(
                        symbol, current_raw, current_qfq, incoming_raw, incoming_qfq,
                    )
                    if provider == "miniqmt":
                        from miniqmt_etf_source import verified_response
                        receipt = incoming_raw.attrs["source_receipt"]
                        response_path, _ = verified_response(root / "cache/cn-etf-incremental", receipt)
                        destination = stage / "source_receipts/miniqmt" / response_path.name
                        if not destination.exists():
                            _link_or_copy(response_path, destination)
                        miniqmt_receipts[receipt["sha256"]] = {
                            "relative_path": destination.relative_to(stage).as_posix(), "sha256": receipt["sha256"]}
                        action_through = incoming_raw.attrs["miniqmt_action_through"]
                        miniqmt_action_history[symbol] = {
                            "through": action_through,
                            "since": first_by_symbol[symbol].date().isoformat(),
                            "sha256": action_history_digest(incoming_raw.attrs["miniqmt_actions"], action_through, first_by_symbol[symbol].date().isoformat()),
                        }
                except CnEtfDownloadError as exc:
                    if not keep_failed_history:
                        raise
                    symbol_update_failures[symbol] = {"error_type": type(exc).__name__, "error": str(exc),
                                                     "last_verified_date": old_end.date().isoformat(),
                                                     "requested_through": through.date().isoformat()}
                    for kind in ("raw", "qfq", "factors"):
                        _link_or_copy(current / kind / f"{symbol}.parquet", stage / kind / f"{symbol}.parquet")
                    factors_by_symbol[symbol] = pd.read_parquet(current_factor_path)
                    continue
                rebase = updated_qfq.attrs.get("historical_rebase")
                if rebase:
                    historical_qfq_rebases[symbol] = dict(rebase)

                if "source_id" not in updated_raw.columns:
                    updated_raw["source_id"] = updated_raw["source"].astype(str)
                if "source_id" not in updated_qfq.columns:
                    updated_qfq["source_id"] = updated_qfq["source"].astype(str)
                factors = build_paired_adjustments(symbol, updated_raw, updated_qfq)
                if rows_added == 0 and not rebase:
                    _link_or_copy(current_raw_path, stage / "raw" / current_raw_path.name)
                    _link_or_copy(current_qfq_path, stage / "qfq" / current_qfq_path.name)
                    _link_or_copy(
                        current_factor_path,
                        stage / "factors" / current_factor_path.name,
                    )
                    factors_by_symbol[symbol] = pd.read_parquet(current_factor_path)
                    continue

                _atomic_parquet(updated_raw, stage / "raw" / f"{symbol}.parquet")
                _atomic_parquet(updated_qfq, stage / "qfq" / f"{symbol}.parquet")
                _atomic_parquet(factors, stage / "factors" / f"{symbol}.parquet")
                factors_by_symbol[symbol] = factors
                changed_symbols.append(symbol)
                rows_added_by_symbol[symbol] = rows_added

            if not changed_symbols:
                _write_incremental_status(
                    root,
                    {
                        "status": "blocked" if symbol_update_failures else "no_new_trade_rows",
                        "dataset_id": DATASET_ID,
                        "snapshot_id": current.name,
                        "through": through.date().isoformat(),
                        "requests_used": gate.calls,
                        "symbols_requested": len(missing_symbols),
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "generated_at": timestamp,
                        "symbol_update_failures": symbol_update_failures,
                    },
                )
                return current

            combined_provider = provider if history_provider == provider else "mixed"
            actions = _corporate_actions_from_adjustments(
                factors_by_symbol, provider=combined_provider,
            )
            if combined_provider == "mixed":
                actions["evidence_method"] = actions["evidence_method"].str.replace(
                    "same_provider", "frozen_history_and_verified_increment", regex=False,
                )
            _atomic_parquet(actions, stage / "corporate_actions.parquet")
            if metadata_only_symbols:
                _atomic_parquet(universe, stage / "ETF_UNIVERSE.parquet")
            else:
                _link_or_copy(universe_path, stage / "ETF_UNIVERSE.parquet")
            current_universe_manifest_path = current / "UNIVERSE_MANIFEST.json"
            if current_universe_manifest_path.is_file():
                universe_manifest = json.loads(
                    current_universe_manifest_path.read_text(encoding="utf-8")
                )
            else:
                universe_manifest = {}
            universe_manifest.update(
                {
                    "inherited_from_snapshot_id": current.name,
                    "used_for_update_through": through.date().isoformat(),
                    "universe_file_sha256": sha256_file(stage / "ETF_UNIVERSE.parquet"),
                }
            )
            atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)

            import pyarrow.parquet as pq
            qfq_response_keys = set()
            for path in sorted((stage / "qfq").glob("*.parquet")):
                if "provider_response_key" in pq.read_schema(path).names:
                    qfq_response_keys.update(str(v) for v in pd.read_parquet(path, columns=["provider_response_key"])["provider_response_key"].dropna().unique())
            qfq_response_keys = sorted(qfq_response_keys)
            source = dict(manifest.get("source") or {})
            source.update(
                {
                    "provider": combined_provider,
                    "increment_provider": provider,
                    "retained_history_provider": history_provider,
                    "interface": ("ifzq_day_and_qfqday_bounded_increment" if provider == "tencent"
                                  else f"{provider}_raw_qfq_bounded_increment"),
                    "update_mode": "missing_dates_only",
                    "parent_snapshot_id": current.name,
                    "selection_method": "frozen_previous_snapshot",
                    "universe_path": "ETF_UNIVERSE.parquet",
                    "universe_manifest_path": "UNIVERSE_MANIFEST.json",
                    "requested_symbol_count": len(missing_symbols),
                    "reused_symbol_count": int(len(universe) - len(changed_symbols)),
                    "overlap_calendar_days": int(overlap_days),
                    "rows_added": int(sum(rows_added_by_symbol.values())),
                    "provider_request_calls": gate.calls,
                    "calendar_bridge_calls": calendar_bridge_calls,
                    "network_requests": getattr(session, "network_calls", gate.calls),
                    "targeted_full_refresh_symbols": sorted(
                        targeted_full_refresh_symbols
                    ),
                    "historical_qfq_rebases": historical_qfq_rebases,
                    "parent_manifest_sha256": base_manifest_sha256,
                    "prepared_only": not publish,
                    "symbol_update_failures": symbol_update_failures,
                    "metadata_only_symbols_without_history": metadata_only_symbols,
                    "qfq_response_keys": qfq_response_keys,
                    "snapshot_retention": {
                        "policy": SNAPSHOT_RETENTION_POLICY,
                        "max_complete_snapshots": DEFAULT_SNAPSHOT_RETENTION,
                    },
                }
            )
            if provider == "miniqmt":
                source["miniqmt_receipts"] = list(miniqmt_receipts.values())
                source["bridge_calls"] = session.bridge_calls
                source["volume_unit"] = "lots"
                source["qfq_event_rule"] = "cash-share-affine-front-v1"
                source["miniqmt_action_history"] = miniqmt_action_history
            latest_date = max(
                _as_date(pd.read_parquet(path, columns=["date"])["date"].max())
                for path in (stage / "qfq").glob("*.parquet")
            ).date().isoformat()
            finalize_paired_price_snapshot(
                stage,
                dataset_id=DATASET_ID,
                snapshot_id=snapshot_id,
                role="canonical",
                status="ready",
                source=source,
                adjustment={
                    "method": "provider_native_paired",
                    "adjusted_at": timestamp,
                    "adjustment_as_of": latest_date,
                    "adjusted_fields": ["open", "high", "low", "close"],
                    "normalization_rule": (
                        "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
                    ),
                    "algorithm_version": "provider-native-affine-paired-v1",
                    "paired_raw_component_path": "raw",
                    "volume_adjusted": False,
                },
                producer={
                    "script": "download_cn_etf_data.py",
                    "version": PRODUCER_VERSION,
                },
                source_retrieved_at=timestamp,
            )
            dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
            if not publish:
                validate_dataset_snapshot(stage, verify_hashes=True)
                keep_prepared = True
                return stage
            published = publish_snapshot_group(
                [(stage, dataset_root)],
                quant_data_root=root,
                update_root_catalog=True,
            )[0]
            removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
                root,
                dataset_root,
                preferred_backup_snapshot_id=current.name,
                # publish_snapshot_group already updated the exact catalog entry.
                sync_catalog=False,
            )
            published_status = {
                "status": (
                    "published_with_warnings"
                    if post_publish_warnings
                    else ("published_partial" if symbol_update_failures else "published")
                ),
                "dataset_id": DATASET_ID,
                "snapshot_id": published.name,
                "parent_snapshot_id": current.name,
                "requested_through": requested_through.date().isoformat(),
                "through": through.date().isoformat(),
                "requests_used": gate.calls,
                "symbols_requested": len(missing_symbols),
                "symbols_changed": len(changed_symbols),
                "rows_added": int(sum(rows_added_by_symbol.values())),
                "targeted_full_refresh_symbols": sorted(
                    targeted_full_refresh_symbols
                ),
                "symbol_update_failures": symbol_update_failures,
                "removed_snapshots": removed_snapshots,
                "post_publish_warnings": post_publish_warnings,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "generated_at": timestamp,
            }
            try:
                _write_incremental_status(root, published_status)
            except Exception as exc:
                print(
                    "ETF snapshot was published; status write failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return published
        finally:
            session.close()
            if not keep_prepared:
                shutil.rmtree(staging_run, ignore_errors=True)


def _expansion_cache_symbol_complete(
    cache: Path,
    symbol: str,
    *,
    minimum_history_rows: int,
    through: pd.Timestamp,
) -> bool:
    if not _cache_symbol_complete(cache, symbol):
        return False
    try:
        raw = pd.read_parquet(cache / "raw" / f"{symbol}.parquet")
        qfq = pd.read_parquet(cache / "qfq" / f"{symbol}.parquet")
    except Exception:
        return False
    for frame in (raw, qfq):
        values = frame[["open", "high", "low", "close"]].apply(
            pd.to_numeric, errors="coerce"
        )
        if (
            values.isna().any().any()
            or (values <= 0).any().any()
            or (
                values["high"]
                < values[["open", "low", "close"]].max(axis=1)
            ).any()
            or (
                values["low"]
                > values[["open", "high", "close"]].min(axis=1)
            ).any()
        ):
            return False
    raw_dates = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
    qfq_dates = pd.to_datetime(qfq["date"], errors="coerce").dt.normalize()
    return bool(
        len(raw_dates) >= minimum_history_rows
        and raw_dates.notna().all()
        and qfq_dates.notna().all()
        and raw_dates.tolist() == qfq_dates.tolist()
        and raw_dates.max() == through
    )


def _reuse_expansion_symbol_from_prior_contract(
    cache: Path,
    symbol: str,
    *,
    selection_config: dict[str, Any],
    minimum_history_rows: int,
    through: pd.Timestamp,
) -> str | None:
    """Copy a verified symbol cache from a compatible failed selection contract."""

    compatible_fields = (
        "parent_identity",
        "start_date",
        "through",
        "provider",
        "producer_version",
    )
    parent = cache.parent
    if parent.is_symlink() or not parent.is_dir():
        return None
    for sibling in sorted(parent.iterdir()):
        if sibling == cache or sibling.is_symlink() or not sibling.is_dir():
            continue
        contract_path = sibling / "RUN_CONTRACT.json"
        if contract_path.is_symlink() or not contract_path.is_file():
            continue
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            sibling_config = contract["selection_config"]
        except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
            continue
        if not isinstance(sibling_config, dict) or any(
            sibling_config.get(field) != selection_config.get(field)
            for field in compatible_fields
        ):
            continue
        if not _expansion_cache_symbol_complete(
            sibling,
            symbol,
            minimum_history_rows=minimum_history_rows,
            through=through,
        ):
            continue
        for component in ("raw", "qfq", "factors"):
            _link_or_copy(
                sibling / component / f"{symbol}.parquet",
                cache / component / f"{symbol}.parquet",
            )
        if not _expansion_cache_symbol_complete(
            cache,
            symbol,
            minimum_history_rows=minimum_history_rows,
            through=through,
        ):
            for component in ("raw", "qfq", "factors"):
                (cache / component / f"{symbol}.parquet").unlink(missing_ok=True)
            continue
        return sibling.name
    return None


def expand_cn_etf_universe(
    *,
    quant_data_dir: Path | None = None,
    target_size: int = DEFAULT_REPRESENTATIVE_TARGET_SIZE,
    category_targets: dict[str, int] | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    minimum_turnover: float = 1_000_000.0,
    minimum_history_rows: int = DEFAULT_MINIMUM_HISTORY_ROWS,
    excluded_symbols: tuple[str, ...] = (),
    required_symbols: tuple[str, ...] = (),
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    generated_at: str | None = None,
    resume: bool = True,
) -> Path:
    """Expand the frozen ETF universe without re-downloading existing symbols."""

    if category_targets is None and target_size != DEFAULT_REPRESENTATIVE_TARGET_SIZE:
        raise ValueError(
            "default representative quotas support target_size=200; "
            "provide explicit category_targets for another registered contract"
        )
    targets = dict(category_targets or REPRESENTATIVE_CATEGORY_TARGETS)
    if target_size < 1 or target_size > 500:
        raise ValueError("representative ETF target_size must be between 1 and 500")
    if sum(targets.values()) != target_size:
        raise ValueError("representative ETF category targets must sum to target_size")
    if minimum_turnover < 0:
        raise ValueError("minimum_turnover must be nonnegative")
    if minimum_history_rows < 1:
        raise ValueError("minimum_history_rows must be positive")
    if request_interval < MIN_EXPANSION_REQUEST_INTERVAL:
        raise ValueError(
            f"request_interval must be at least {MIN_EXPANSION_REQUEST_INTERVAL}"
        )
    if request_budget < 1 or request_budget > 10_000:
        raise ValueError("expansion request_budget must be between 1 and 10000")
    if retries < 1 or retries > 2:
        raise ValueError("retries must be between 1 and 2")
    if max_consecutive_failures < 1 or max_consecutive_failures > 3:
        raise ValueError("max_consecutive_failures must be between 1 and 3")

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    through = _as_date(
        end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat()
    )
    if through.weekday() >= 5:
        through = through - pd.Timedelta(days=through.weekday() - 4)
    if _as_date(start_date) > through:
        raise ValueError("start_date must not be after end_date")
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    normalized_exclusions = tuple(
        sorted(
            {
                str(symbol).strip().upper()
                for symbol in excluded_symbols
                if str(symbol).strip()
            }
        )
    )
    normalized_required = tuple(
        sorted(
            {
                str(symbol).strip().upper()
                for symbol in required_symbols
                if str(symbol).strip()
            }
        )
    )
    if set(normalized_required) & set(normalized_exclusions):
        raise ValueError("required ETF symbols cannot also be excluded")
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    started = time.monotonic()

    with _run_lock(lock_path):
        resolved = resolve_dataset(
            DATASET_ID,
            quant_data_root=root,
            require_basis="qfq",
        )
        current = resolved.snapshot_path
        manifest = resolved.manifest
        if manifest.get("data_kind") != "paired_daily_bars":
            raise DataContractError("cn_etf_daily is not a paired raw/qfq snapshot")
        current_universe_path = current / "ETF_UNIVERSE.parquet"
        if not current_universe_path.is_file():
            raise DataContractError("current ETF universe is unavailable")
        current_universe = pd.read_parquet(current_universe_path)
        current_symbols = set(current_universe["symbol"].astype(str))
        if len(current_symbols) != len(current_universe):
            raise DataContractError("current ETF universe contains duplicate symbols")
        missing_required = set(normalized_required) - current_symbols
        if len(current_symbols) + len(missing_required) > target_size:
            raise CnEtfDownloadError(
                "target_size has no capacity for every missing required ETF symbol"
            )
        if len(current_symbols) == target_size:
            return current
        if len(current_symbols) > target_size:
            raise CnEtfDownloadError(
                "ETF universe expansion cannot remove current symbols"
            )

        parent_identity = {
            "snapshot_id": current.name,
            "manifest_sha256": str(
                resolved.catalog_entry.get("manifest_sha256") or ""
            ),
            "content_sha256": str(manifest.get("content_sha256") or ""),
        }
        selection_config = {
            "schema_version": 1,
            "parent_identity": parent_identity,
            "start_date": _as_date(start_date).date().isoformat(),
            "through": through.date().isoformat(),
            "target_size": int(target_size),
            "category_targets": targets,
            "minimum_turnover": float(minimum_turnover),
            "minimum_history_rows": int(minimum_history_rows),
            "excluded_symbols": list(normalized_exclusions),
            "required_symbols": list(normalized_required),
            "provider": "tencent",
            "producer_version": PRODUCER_VERSION,
        }
        config_sha256 = hashlib.sha256(
            json.dumps(selection_config, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache = (
            root
            / "cache"
            / DATASET_ID
            / "expansion"
            / "v1"
            / current.name
            / through.date().isoformat()
            / f"target-{target_size}-{config_sha256[:12]}"
        )
        cache.mkdir(parents=True, exist_ok=True)
        contract_path = cache / "RUN_CONTRACT.json"
        universe_path = cache / "ETF_UNIVERSE.parquet"
        summary_path = cache / "UNIVERSE_SUMMARY.json"
        gate = RequestGate(interval=request_interval, budget=request_budget)

        if resume and contract_path.is_file() and universe_path.is_file() and summary_path.is_file():
            try:
                contract = json.loads(contract_path.read_text(encoding="utf-8"))
                selected = pd.read_parquet(universe_path)
                catalog_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise CnEtfDownloadError("ETF expansion resume cache is unreadable") from exc
            if contract.get("selection_config") != selection_config:
                raise CnEtfDownloadError("ETF expansion resume cache has another contract")
            if contract.get("universe_sha256") != _universe_hash(selected):
                raise CnEtfDownloadError("ETF expansion universe hash is inconsistent")
        else:
            catalog, catalog_summary = fetch_etf_catalog(
                through=through.date().isoformat(),
                request_gate=gate,
            )
            selected = select_representative_universe(
                catalog,
                current_universe,
                target_size=target_size,
                category_targets=targets,
                minimum_turnover=minimum_turnover,
                excluded_symbols=normalized_exclusions,
                required_symbols=normalized_required,
            )
            universe_sha256 = _universe_hash(selected)
            contract = {
                "selection_config": selection_config,
                "universe_sha256": universe_sha256,
                "symbol_count": int(len(selected)),
                "added_symbols": sorted(
                    set(selected["symbol"].astype(str)) - current_symbols
                ),
            }
            atomic_write_json(contract_path, contract)
            _atomic_parquet(selected, universe_path)
            atomic_write_json(summary_path, catalog_summary)

        selected_symbols = set(selected["symbol"].astype(str))
        if not current_symbols.issubset(selected_symbols):
            raise CnEtfDownloadError("ETF expansion dropped a current symbol")
        added_symbols = sorted(selected_symbols - current_symbols)
        if len(selected_symbols) != target_size:
            raise CnEtfDownloadError("ETF expansion cache has the wrong symbol count")

        import requests

        session = requests.Session()
        session.trust_env = False
        failures: dict[str, str] = {}
        failures_path = cache / "failures.json"
        if failures_path.is_file():
            try:
                payload = json.loads(failures_path.read_text(encoding="utf-8"))
                failures = {
                    str(item["symbol"]): str(item["error"])
                    for item in payload.get("failures", [])
                    if isinstance(item, dict) and item.get("symbol")
                }
            except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
                failures = {}

        metadata_by_symbol = {
            str(row.symbol): row for row in selected.itertuples(index=False)
        }
        completed = 0
        downloaded = 0
        consecutive_failures = 0
        reused_prior_contract_symbols: list[str] = []
        for position, symbol in enumerate(added_symbols, 1):
            atomic_write_json(
                cache / "progress.json",
                {
                    "status": "running",
                    "parent_snapshot_id": current.name,
                    "current": position,
                    "total": len(added_symbols),
                    "symbol": symbol,
                    "completed": completed,
                    "downloaded_this_run": downloaded,
                    "requests_used": gate.calls,
                    "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                },
            )
            if resume:
                cache_complete = _expansion_cache_symbol_complete(
                    cache,
                    symbol,
                    minimum_history_rows=minimum_history_rows,
                    through=through,
                )
                if not cache_complete:
                    reused_from = _reuse_expansion_symbol_from_prior_contract(
                        cache,
                        symbol,
                        selection_config=selection_config,
                        minimum_history_rows=minimum_history_rows,
                        through=through,
                    )
                    if reused_from is not None:
                        reused_prior_contract_symbols.append(symbol)
                        cache_complete = True
                if cache_complete:
                    completed += 1
                    failures.pop(symbol, None)
                    continue
            row = metadata_by_symbol[symbol]
            metadata = {
                "code": str(row.code).zfill(6),
                "market": str(row.market).lower(),
                "name": str(row.name),
            }
            try:
                raw, qfq = fetch_tencent_pair(
                    symbol,
                    metadata,
                    start_date=_as_date(start_date).date().isoformat(),
                    end_date=through.date().isoformat(),
                    session=session,
                    request_gate=gate,
                    retries=retries,
                    timeout=timeout,
                )
                raw = _normalize_component_frame(
                    raw, symbol=symbol, component="expansion raw"
                )
                qfq = _normalize_component_frame(
                    qfq, symbol=symbol, component="expansion qfq"
                )
                if len(raw) < minimum_history_rows:
                    raise DataContractError(
                        f"expansion history is shorter than {minimum_history_rows} rows for {symbol}"
                    )
                if raw["date"].tolist() != qfq["date"].tolist():
                    raise DataContractError(f"expansion raw/qfq dates differ for {symbol}")
                if _as_date(raw["date"].max()) != through:
                    raise DataContractError(
                        f"expansion history does not reach {through.date().isoformat()} for {symbol}"
                    )
                raw["source_id"] = raw["source"].astype(str)
                qfq["source_id"] = qfq["source"].astype(str)
                factors = build_paired_adjustments(symbol, raw, qfq)
                _atomic_parquet(raw, cache / "raw" / f"{symbol}.parquet")
                _atomic_parquet(qfq, cache / "qfq" / f"{symbol}.parquet")
                _atomic_parquet(factors, cache / "factors" / f"{symbol}.parquet")
                failures.pop(symbol, None)
                completed += 1
                downloaded += 1
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                failures[symbol] = f"{type(exc).__name__}: {str(exc)[:300]}"
                atomic_write_json(
                    failures_path,
                    {
                        "failures": [
                            {"symbol": key, "error": value}
                            for key, value in sorted(failures.items())
                        ]
                    },
                )
                if consecutive_failures >= max_consecutive_failures:
                    break

        atomic_write_json(
            failures_path,
            {
                "failures": [
                    {"symbol": key, "error": value}
                    for key, value in sorted(failures.items())
                ]
            },
        )
        incomplete = [
            symbol
            for symbol in added_symbols
            if not _expansion_cache_symbol_complete(
                cache,
                symbol,
                minimum_history_rows=minimum_history_rows,
                through=through,
            )
        ]
        if incomplete:
            atomic_write_json(
                cache / "progress.json",
                {
                    "status": "incomplete",
                    "parent_snapshot_id": current.name,
                    "completed": len(added_symbols) - len(incomplete),
                    "total": len(added_symbols),
                    "incomplete": incomplete,
                    "requests_used": gate.calls,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
            )
            raise CnEtfDownloadError(
                f"ETF universe expansion is incomplete: {len(incomplete)} of {len(added_symbols)} new symbols failed; cache retained for resume"
            )

        run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        snapshot_id = f"cn-etf-tencent-expansion-{run_id}"
        staging_run = root / "staging" / f"cn-etf-expansion-{run_id}"
        stage = staging_run / snapshot_id
        old_hashes: dict[str, dict[str, str]] = {
            component: {} for component in ("raw", "qfq", "factors")
        }
        try:
            factors_by_symbol: dict[str, pd.DataFrame] = {}
            for symbol in sorted(current_symbols):
                for component in ("raw", "qfq", "factors"):
                    source_path = current / component / f"{symbol}.parquet"
                    if not source_path.is_file():
                        raise DataContractError(
                            f"current ETF {component} file is missing for {symbol}"
                        )
                    old_hashes[component][symbol] = sha256_file(source_path)
                    _link_or_copy(
                        source_path,
                        stage / component / source_path.name,
                    )
                factors_by_symbol[symbol] = pd.read_parquet(
                    current / "factors" / f"{symbol}.parquet"
                )
            for symbol in added_symbols:
                for component in ("raw", "qfq", "factors"):
                    _link_or_copy(
                        cache / component / f"{symbol}.parquet",
                        stage / component / f"{symbol}.parquet",
                    )
                factors_by_symbol[symbol] = pd.read_parquet(
                    cache / "factors" / f"{symbol}.parquet"
                )
            for component, hashes in old_hashes.items():
                for symbol, digest in hashes.items():
                    if sha256_file(stage / component / f"{symbol}.parquet") != digest:
                        raise DataContractError(
                            f"ETF expansion changed inherited {component} content for {symbol}"
                        )

            actions = _corporate_actions_from_adjustments(
                factors_by_symbol,
                provider="tencent",
            )
            _atomic_parquet(actions, stage / "corporate_actions.parquet")
            _atomic_parquet(selected, stage / "ETF_UNIVERSE.parquet")
            universe_sha256 = _universe_hash(selected)
            category_counts = {
                key: int(value)
                for key, value in selected["representative_category"]
                .value_counts()
                .to_dict()
                .items()
            }
            universe_manifest = {
                **catalog_summary,
                "selected_count": int(len(selected)),
                "symbol_count": int(len(selected)),
                "retrieved_at": timestamp,
                "through": through.date().isoformat(),
                "parent_snapshot_id": current.name,
                "parent_content_sha256": parent_identity["content_sha256"],
                "selection_method": "frozen_existing_plus_category_quota_liquidity",
                "selection_config_sha256": config_sha256,
                "category_targets": targets,
                "required_symbols": list(normalized_required),
                "category_counts": category_counts,
                "universe_sha256": universe_sha256,
                "universe_file_sha256": sha256_file(stage / "ETF_UNIVERSE.parquet"),
            }
            atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)
            qfq_response_keys = sorted(
                {
                    str(value)
                    for path in sorted((stage / "qfq").glob("*.parquet"))
                    for value in pd.read_parquet(
                        path, columns=["provider_response_key"]
                    )["provider_response_key"].dropna().unique()
                }
            )
            source = dict(manifest.get("source") or {})
            source.update(
                {
                    "provider": "tencent",
                    "interface": "ifzq_day_and_qfqday_full_history_for_new_symbols",
                    "update_mode": "universe_expansion",
                    "parent_snapshot_id": current.name,
                    "parent_content_sha256": parent_identity["content_sha256"],
                    "selection_method": "frozen_existing_plus_category_quota_liquidity",
                    "selection_config_sha256": config_sha256,
                    "universe_path": "ETF_UNIVERSE.parquet",
                    "universe_manifest_path": "UNIVERSE_MANIFEST.json",
                    "universe_sha256": universe_sha256,
                    "retained_symbol_count": len(current_symbols),
                    "added_symbol_count": len(added_symbols),
                    "added_symbols": added_symbols,
                    "excluded_symbols": list(normalized_exclusions),
                    "required_symbols": list(normalized_required),
                    "reused_prior_contract_symbol_count": len(
                        reused_prior_contract_symbols
                    ),
                    "reused_prior_contract_symbols": sorted(
                        reused_prior_contract_symbols
                    ),
                    "category_targets": targets,
                    "category_counts": category_counts,
                    "qfq_response_keys": qfq_response_keys,
                    "snapshot_retention": {
                        "policy": SNAPSHOT_RETENTION_POLICY,
                        "max_complete_snapshots": DEFAULT_SNAPSHOT_RETENTION,
                    },
                }
            )
            finalize_paired_price_snapshot(
                stage,
                dataset_id=DATASET_ID,
                snapshot_id=snapshot_id,
                role="canonical",
                status="ready",
                source=source,
                adjustment={
                    "method": "provider_native_paired",
                    "adjusted_at": timestamp,
                    "adjustment_as_of": through.date().isoformat(),
                    "adjusted_fields": ["open", "high", "low", "close"],
                    "normalization_rule": (
                        "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
                    ),
                    "algorithm_version": "provider-native-affine-paired-v1",
                    "paired_raw_component_path": "raw",
                    "volume_adjusted": False,
                },
                producer={
                    "script": "download_cn_etf_data.py",
                    "version": PRODUCER_VERSION,
                },
                source_retrieved_at=timestamp,
            )
            for component, hashes in old_hashes.items():
                for symbol, digest in hashes.items():
                    if sha256_file(stage / component / f"{symbol}.parquet") != digest:
                        raise DataContractError(
                            f"ETF expansion altered inherited {component} after finalization for {symbol}"
                        )
            dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
            published = publish_snapshot_group(
                [(stage, dataset_root)],
                quant_data_root=root,
                update_root_catalog=True,
                published_at=timestamp,
            )[0]
            removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
                root,
                dataset_root,
            )
            status = {
                "status": (
                    "published_with_warnings"
                    if post_publish_warnings
                    else "published"
                ),
                "dataset_id": DATASET_ID,
                "snapshot_id": published.name,
                "parent_snapshot_id": current.name,
                "through": through.date().isoformat(),
                "target_size": target_size,
                "retained_symbol_count": len(current_symbols),
                "added_symbol_count": len(added_symbols),
                "downloaded_this_run": downloaded,
                "reused_prior_contract_symbol_count": len(
                    reused_prior_contract_symbols
                ),
                "requests_used": gate.calls,
                "category_counts": category_counts,
                "removed_snapshots": removed_snapshots,
                "post_publish_warnings": post_publish_warnings,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "generated_at": timestamp,
            }
            try:
                atomic_write_json(
                    root / "runtime" / "status" / "cn_etf_daily_expand_latest.json",
                    status,
                )
                atomic_write_json(cache / "progress.json", status)
            except Exception as exc:
                print(
                    "ETF expansion was published; status write failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return published
        finally:
            shutil.rmtree(staging_run, ignore_errors=True)


def reissue_cn_etf_expansion_metadata(
    *,
    quant_data_dir: Path | None = None,
    generated_at: str | None = None,
) -> Path:
    """Publish corrected expansion lineage without changing or downloading bars."""

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    with _run_lock(lock_path):
        resolved = resolve_dataset(
            DATASET_ID,
            quant_data_root=root,
            require_basis="qfq",
        )
        current = resolved.snapshot_path
        manifest = resolved.manifest
        source = dict(manifest.get("source") or {})
        if source.get("update_mode") != "universe_expansion":
            raise CnEtfDownloadError(
                "current ETF snapshot is not an expansion snapshot"
            )
        parent_snapshot_id = str(source.get("parent_snapshot_id") or "")
        dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
        parent_snapshot = dataset_root / "snapshots" / parent_snapshot_id
        if (
            not parent_snapshot_id
            or parent_snapshot.is_symlink()
            or not parent_snapshot.is_dir()
        ):
            raise CnEtfDownloadError(
                "ETF expansion parent snapshot is unavailable for metadata correction"
            )
        universe_path = current / "ETF_UNIVERSE.parquet"
        universe_manifest_path = current / "UNIVERSE_MANIFEST.json"
        if not universe_path.is_file() or not universe_manifest_path.is_file():
            raise DataContractError("ETF expansion universe evidence is unavailable")
        universe = pd.read_parquet(universe_path)
        universe_manifest = json.loads(
            universe_manifest_path.read_text(encoding="utf-8")
        )
        universe_sha256 = _universe_hash(universe)
        category_counts = (
            {
                key: int(value)
                for key, value in universe["representative_category"]
                .value_counts()
                .to_dict()
                .items()
            }
            if "representative_category" in universe.columns
            else dict(source.get("category_counts") or {})
        )

        run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
        snapshot_id = f"cn-etf-tencent-expansion-metadata-{run_id}"
        staging_run = root / "staging" / f"cn-etf-expansion-metadata-{run_id}"
        stage = staging_run / snapshot_id
        try:
            for component in ("raw", "qfq", "factors"):
                component_root = current / component
                if component_root.is_symlink() or not component_root.is_dir():
                    raise DataContractError(
                        f"ETF expansion {component} component is unavailable"
                    )
                for path in sorted(component_root.glob("*.parquet")):
                    _link_or_copy(path, stage / component / path.name)
            for name in ("corporate_actions.parquet", "ETF_UNIVERSE.parquet"):
                path = current / name
                if path.is_symlink() or not path.is_file():
                    raise DataContractError(
                        f"ETF expansion metadata correction source is missing: {name}"
                    )
                _link_or_copy(path, stage / name)

            universe_manifest.update(
                {
                    "selected_count": int(len(universe)),
                    "symbol_count": int(len(universe)),
                    "metadata_corrected_at": timestamp,
                    "metadata_correction_of_snapshot_id": current.name,
                    "universe_sha256": universe_sha256,
                    "universe_file_sha256": sha256_file(
                        stage / "ETF_UNIVERSE.parquet"
                    ),
                }
            )
            atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)
            source.update(
                {
                    "universe_sha256": universe_sha256,
                    "category_counts": category_counts,
                    "metadata_corrected_at": timestamp,
                    "metadata_correction_of_snapshot_id": current.name,
                }
            )
            adjustment = dict(manifest.get("adjustment") or {})
            producer = dict(manifest.get("producer") or {})
            producer.update(
                {
                    "script": "download_cn_etf_data.py",
                    "version": PRODUCER_VERSION,
                    "operation": "expansion_metadata_correction",
                }
            )
            finalize_paired_price_snapshot(
                stage,
                dataset_id=DATASET_ID,
                snapshot_id=snapshot_id,
                role="canonical",
                status="ready",
                source=source,
                adjustment=adjustment,
                producer=producer,
                source_retrieved_at=str(
                    manifest.get("source_retrieved_at") or timestamp
                ),
            )
            published = publish_snapshot_group(
                [(stage, dataset_root)],
                quant_data_root=root,
                update_root_catalog=True,
                published_at=timestamp,
            )[0]
            removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
                root,
                dataset_root,
                preferred_backup_snapshot_id=parent_snapshot_id,
            )
            try:
                atomic_write_json(
                    root / "runtime" / "status" / "cn_etf_daily_expand_latest.json",
                    {
                        "status": (
                            "published_with_warnings"
                            if post_publish_warnings
                            else "published"
                        ),
                        "dataset_id": DATASET_ID,
                        "snapshot_id": published.name,
                        "parent_snapshot_id": parent_snapshot_id,
                        "metadata_correction_of_snapshot_id": current.name,
                        "symbol_count": int(len(universe)),
                        "removed_snapshots": removed_snapshots,
                        "post_publish_warnings": post_publish_warnings,
                        "generated_at": timestamp,
                    },
                )
            except Exception as exc:
                print(
                    "ETF metadata correction was published; status write failed: "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
            return published
        finally:
            shutil.rmtree(staging_run, ignore_errors=True)


def _publish_cn_etf_snapshot_locked(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    generated_at: str | None = None,
    resume: bool = True,
    confirm_severe_current_corruption: bool = False,
    confirm_backup_unrecoverable: bool = False,
    confirm_user_approved: bool = False,
) -> Path:
    """Fully rebuild ETF history after the separate approval gate is satisfied."""

    if not all(
        (
            confirm_severe_current_corruption,
            confirm_backup_unrecoverable,
            confirm_user_approved,
        )
    ):
        raise CnEtfDownloadError(
            "full ETF rebuild requires proven severe current corruption, "
            "an unrecoverable backup, and explicit user approval"
        )
    if request_interval < MIN_REQUEST_INTERVAL:
        raise ValueError(f"request_interval must be at least {MIN_REQUEST_INTERVAL}")
    if request_budget < 1:
        raise ValueError("request_budget must be positive")
    if retries < 1 or retries > 2:
        raise ValueError("retries must be between 1 and 2")
    if max_consecutive_failures < 1 or max_consecutive_failures > 3:
        raise ValueError("max_consecutive_failures must be between 1 and 3")

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    through = (end_date or pd.Timestamp.now(tz="Asia/Shanghai").date().isoformat())
    if _as_date(start_date) > _as_date(through):
        raise ValueError("start_date must not be after end_date")
    timestamp = generated_at or datetime.now().astimezone().isoformat()
    gate = RequestGate(interval=request_interval, budget=request_budget)
    normalized_required_codes = tuple(
        dict.fromkeys(str(code).zfill(6) for code in required_codes)
    )
    selection_contract = {
        "start_date": _as_date(start_date).date().isoformat(),
        "through": _as_date(through).date().isoformat(),
        "limit": int(limit),
        "required_codes": list(normalized_required_codes),
        "provider": "tencent",
        "producer_version": PRODUCER_VERSION,
    }
    selection_digest = hashlib.sha256(
        json.dumps(selection_contract, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache = (
        root
        / "cache"
        / "cn_etf_daily"
        / "v1"
        / through
        / f"trial-{limit}-{selection_digest[:12]}"
    )
    cache.mkdir(parents=True, exist_ok=True)
    contract_path = cache / "RUN_CONTRACT.json"
    universe_path = cache / "ETF_UNIVERSE.parquet"
    summary_path = cache / "UNIVERSE_SUMMARY.json"
    if resume and contract_path.is_file() and universe_path.is_file() and summary_path.is_file():
        existing_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if any(
            existing_contract.get(key) != value
            for key, value in selection_contract.items()
        ):
            raise CnEtfDownloadError("resume cache belongs to another ETF contract")
        universe = pd.read_parquet(universe_path)
        catalog_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        universe_digest = _universe_hash(universe)
        if (
            existing_contract.get("universe_sha256") != universe_digest
            or int(existing_contract.get("symbol_count") or 0) != len(universe)
        ):
            raise CnEtfDownloadError("resume ETF universe hash does not match its contract")
    else:
        universe, catalog_summary = fetch_etf_universe(
            through=through,
            request_gate=gate,
            limit=limit,
            required_codes=normalized_required_codes,
        )
        universe_digest = _universe_hash(universe)
        contract = {
            "schema_version": 1,
            **selection_contract,
            "universe_sha256": universe_digest,
            "symbol_count": int(len(universe)),
        }
        atomic_write_json(contract_path, contract)
        _atomic_parquet(universe, universe_path)
        atomic_write_json(summary_path, catalog_summary)

    import requests

    session = requests.Session()
    session.trust_env = False
    failures: dict[str, str] = {}
    failures_path = cache / "failures.json"
    if failures_path.is_file():
        try:
            payload = json.loads(failures_path.read_text(encoding="utf-8"))
            failures = {
                str(item["symbol"]): str(item["error"])
                for item in payload.get("failures", [])
                if isinstance(item, dict) and item.get("symbol")
            }
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            failures = {}

    completed = 0
    downloaded = 0
    upgraded_cache = 0
    consecutive_failures = 0
    started = time.monotonic()
    for position, row in enumerate(universe.itertuples(index=False), 1):
        symbol = str(row.symbol)
        progress = {
            "status": "running",
            "current": position,
            "total": int(len(universe)),
            "symbol": symbol,
            "completed": completed,
            "downloaded_this_run": downloaded,
            "upgraded_cached_factor_files": upgraded_cache,
            "requests_used": gate.calls,
            "through": through,
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        atomic_write_json(cache / "progress.json", progress)
        if resume:
            cache_complete = _cache_symbol_complete(cache, symbol)
            if not cache_complete and _upgrade_cached_adjustment_evidence(cache, symbol):
                upgraded_cache += 1
                cache_complete = _cache_symbol_complete(cache, symbol)
            if cache_complete:
                completed += 1
                failures.pop(symbol, None)
                continue
        metadata = {
            "code": str(row.code),
            "market": str(row.market),
            "name": str(row.name),
        }
        try:
            raw, qfq = fetch_tencent_pair(
                symbol,
                metadata,
                start_date=start_date,
                end_date=through,
                session=session,
                request_gate=gate,
                retries=retries,
                timeout=timeout,
            )
            factors = build_paired_adjustments(symbol, raw, qfq)
            raw["source_id"] = raw["source"].astype(str)
            qfq["source_id"] = qfq["source"].astype(str)
            _atomic_parquet(raw, cache / "raw" / f"{symbol}.parquet")
            _atomic_parquet(qfq, cache / "qfq" / f"{symbol}.parquet")
            _atomic_parquet(factors, cache / "factors" / f"{symbol}.parquet")
            failures.pop(symbol, None)
            completed += 1
            downloaded += 1
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            failures[symbol] = f"{type(exc).__name__}: {str(exc)[:300]}"
            atomic_write_json(
                failures_path,
                {
                    "failures": [
                        {"symbol": key, "error": value}
                        for key, value in sorted(failures.items())
                    ]
                },
            )
            if consecutive_failures >= max_consecutive_failures:
                break
    atomic_write_json(
        failures_path,
        {
            "failures": [
                {"symbol": key, "error": value}
                for key, value in sorted(failures.items())
            ]
        },
    )

    incomplete = [
        str(symbol)
        for symbol in universe["symbol"]
        if not _cache_symbol_complete(cache, str(symbol))
    ]
    if incomplete:
        atomic_write_json(
            cache / "progress.json",
            {
                "status": "incomplete",
                "completed": int(len(universe) - len(incomplete)),
                "total": int(len(universe)),
                "incomplete": incomplete,
                "requests_used": gate.calls,
                "upgraded_cached_factor_files": upgraded_cache,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "through": through,
            },
        )
        raise CnEtfDownloadError(
            f"common ETF download is incomplete: {len(incomplete)} of {len(universe)} symbols failed; cache retained for resume"
        )

    run_id = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    snapshot_id = f"cn-etf-tencent-{run_id}"
    staging_run = root / "staging" / f"cn-etf-{run_id}"
    stage = staging_run / snapshot_id
    try:
        factors_by_symbol: dict[str, pd.DataFrame] = {}
        for symbol in universe["symbol"].astype(str):
            raw = pd.read_parquet(cache / "raw" / f"{symbol}.parquet")
            qfq = pd.read_parquet(cache / "qfq" / f"{symbol}.parquet")
            factors = pd.read_parquet(cache / "factors" / f"{symbol}.parquet")
            _atomic_parquet(raw, stage / "raw" / f"{symbol}.parquet")
            _atomic_parquet(qfq, stage / "qfq" / f"{symbol}.parquet")
            _atomic_parquet(factors, stage / "factors" / f"{symbol}.parquet")
            factors_by_symbol[symbol] = factors
        actions = _corporate_actions_from_adjustments(
            factors_by_symbol,
            provider="tencent",
        )
        _atomic_parquet(actions, stage / "corporate_actions.parquet")
        _atomic_parquet(universe, stage / "ETF_UNIVERSE.parquet")
        universe_manifest = {
            **catalog_summary,
            "retrieved_at": timestamp,
            "through": through,
            "universe_sha256": universe_digest,
            "universe_file_sha256": sha256_file(stage / "ETF_UNIVERSE.parquet"),
        }
        atomic_write_json(stage / "UNIVERSE_MANIFEST.json", universe_manifest)
        qfq_response_keys = sorted(
            {
                str(value)
                for symbol in universe["symbol"].astype(str)
                for value in pd.read_parquet(
                    cache / "qfq" / f"{symbol}.parquet",
                    columns=["provider_response_key"],
                )["provider_response_key"].dropna().unique()
            }
        )
        source = {
            "provider": "tencent",
            "interface": "ifzq_day_and_qfqday_backward_pagination",
            "universe_providers": ["sina", "tonghuashun"],
            "universe_path": "ETF_UNIVERSE.parquet",
            "universe_manifest_path": "UNIVERSE_MANIFEST.json",
            "universe_sha256": universe_digest,
            "selection_method": catalog_summary.get("selection_method"),
            "requested_symbol_count": int(len(universe)),
            "qfq_response_keys": qfq_response_keys,
            "snapshot_retention": {
                "policy": SNAPSHOT_RETENTION_POLICY,
                "max_complete_snapshots": DEFAULT_SNAPSHOT_RETENTION,
            },
        }
        latest_date = max(
            pd.read_parquet(cache / "qfq" / f"{symbol}.parquet", columns=["date"])[
                "date"
            ].max().date().isoformat()
            for symbol in universe["symbol"].astype(str)
        )
        finalize_paired_price_snapshot(
            stage,
            dataset_id=DATASET_ID,
            snapshot_id=snapshot_id,
            role="canonical",
            status="ready",
            source=source,
            adjustment={
                "method": "provider_native_paired",
                "adjusted_at": timestamp,
                "adjustment_as_of": latest_date,
                "adjusted_fields": ["open", "high", "low", "close"],
                "normalization_rule": (
                    "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
                ),
                "algorithm_version": "provider-native-affine-paired-v1",
                "paired_raw_component_path": "raw",
                "volume_adjusted": False,
            },
            producer={"script": "download_cn_etf_data.py", "version": PRODUCER_VERSION},
            source_retrieved_at=timestamp,
        )
        dataset_root = root / "parquet" / "etf_1d" / DATASET_ID
        published = publish_snapshot_group(
            [(stage, dataset_root)],
            quant_data_root=root,
            update_root_catalog=True,
        )
        removed_snapshots, post_publish_warnings = _post_publish_housekeeping(
            root,
            dataset_root,
        )
        published_status = {
            "status": (
                "published_with_warnings"
                if post_publish_warnings
                else "published"
            ),
            "completed": int(len(universe)),
            "total": int(len(universe)),
            "downloaded_this_run": downloaded,
            "upgraded_cached_factor_files": upgraded_cache,
            "requests_used": gate.calls,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "snapshot_id": snapshot_id,
            "removed_snapshots": removed_snapshots,
            "post_publish_warnings": post_publish_warnings,
            "through": through,
        }
        try:
            atomic_write_json(cache / "progress.json", published_status)
        except Exception as exc:
            print(
                "ETF snapshot was published; progress write failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        return published[0]
    finally:
        shutil.rmtree(staging_run, ignore_errors=True)


def publish_cn_etf_snapshot(
    *,
    quant_data_dir: Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str | None = None,
    limit: int = DEFAULT_LIMIT,
    required_codes: tuple[str, ...] = DEFAULT_REQUIRED_CODES,
    request_interval: float = DEFAULT_REQUEST_INTERVAL,
    request_budget: int = DEFAULT_REQUEST_BUDGET,
    retries: int = 2,
    timeout: float = 20.0,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
    generated_at: str | None = None,
    resume: bool = True,
    confirm_severe_current_corruption: bool = False,
    confirm_backup_unrecoverable: bool = False,
    confirm_user_approved: bool = False,
) -> Path:
    """Hold the shared ETF lock through full-history publication and cleanup."""

    root = Path(quant_data_dir or quant_data_root()).expanduser().resolve()
    lock_path = root / "runtime" / "locks" / "cn_etf_update.lock"
    with _run_lock(lock_path):
        return _publish_cn_etf_snapshot_locked(
            quant_data_dir=root,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            required_codes=required_codes,
            request_interval=request_interval,
            request_budget=request_budget,
            retries=retries,
            timeout=timeout,
            max_consecutive_failures=max_consecutive_failures,
            generated_at=generated_at,
            resume=resume,
            confirm_severe_current_corruption=confirm_severe_current_corruption,
            confirm_backup_unrecoverable=confirm_backup_unrecoverable,
            confirm_user_approved=confirm_user_approved,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-data-root", type=Path, default=quant_data_root())
    parser.add_argument("--full-rebuild", action="store_true")
    parser.add_argument("--update", action="store_true", help="Update the existing file snapshot without the project service")
    parser.add_argument("--base-snapshot", type=Path, help="Explicit frozen file snapshot")
    parser.add_argument("--provider", choices=("miniqmt", "tencent", "eastmoney", "baostock"), default="miniqmt", help="Provider for bounded updates only")
    parser.add_argument("--symbol", action="append", dest="symbols", help="Prepare selected existing symbols only")
    parser.add_argument("--prepare-only", action="store_true", help="Validate staged files without changing LATEST")
    parser.add_argument("--keep-failed-history", action="store_true", default=None, help="Disclose supplier failures and retain those symbols' exact old files (default for MiniQMT)")
    parser.add_argument("--confirm-severe-current-corruption", action="store_true")
    parser.add_argument("--confirm-backup-unrecoverable", action="store_true")
    parser.add_argument("--confirm-user-approved", action="store_true")
    parser.add_argument("--start", default=DEFAULT_START_DATE)
    parser.add_argument("--through", required=True)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--required-code", action="append", dest="required_codes")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL)
    parser.add_argument("--request-budget", type=int, default=DEFAULT_REQUEST_BUDGET)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.update:
        if args.full_rebuild:
            raise ValueError("--update and --full-rebuild are mutually exclusive")
        updated = update_cn_etf_snapshot(
            quant_data_dir=args.quant_data_root, end_date=args.through,
            base_snapshot=args.base_snapshot, symbols=args.symbols, publish=not args.prepare_only,
            keep_failed_history=args.keep_failed_history, provider=args.provider,
            request_interval=args.request_interval,
            request_budget=args.request_budget, retries=args.retries, timeout=args.timeout,
        )
        print(updated)
        return 0
    confirmations = (
        args.full_rebuild,
        args.confirm_severe_current_corruption,
        args.confirm_backup_unrecoverable,
        args.confirm_user_approved,
    )
    if not all(confirmations):
        print(
            "拒绝全量重建：必须同时确认当前快照严重损坏、备份无法恢复，"
            "并且用户已明确批准本次全量重建。普通更新请使用 "
            "quant_data_cli.py update cn_etf_daily。",
            file=sys.stderr,
        )
        return 2
    required_codes = tuple(args.required_codes or DEFAULT_REQUIRED_CODES)
    published = publish_cn_etf_snapshot(
        quant_data_dir=args.quant_data_root,
        start_date=args.start,
        end_date=args.through,
        limit=args.limit,
        required_codes=required_codes,
        request_interval=args.request_interval,
        request_budget=args.request_budget,
        retries=args.retries,
        timeout=args.timeout,
        resume=not args.no_resume,
        confirm_severe_current_corruption=args.confirm_severe_current_corruption,
        confirm_backup_unrecoverable=args.confirm_backup_unrecoverable,
        confirm_user_approved=args.confirm_user_approved,
    )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
