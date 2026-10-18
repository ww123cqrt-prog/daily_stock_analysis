#!/usr/bin/env python3
"""Read-only ETF data diagnostics; never a publication or reading gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import stat
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

try:
    import resource
except ImportError:  # Windows has no resource module; the checker stays usable.
    resource = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from quant_data_contract import (  # noqa: E402
    BAR_OHLC_ABSOLUTE_TOLERANCE,
    BAR_OHLC_RELATIVE_TOLERANCE,
    PAIRED_PRICE_ABSOLUTE_TOLERANCE,
    DataContractError,
    _json_bytes,
    _validate_bar_frame,
)

FIELDS = ("open", "high", "low", "close", "volume", "amount")
PRICE_FIELDS = FIELDS[:4]
FINDING_FIELDS = (
    "severity",
    "rule",
    "symbol",
    "date",
    "date_end",
    "component",
    "field",
    "affected_rows",
    "observed",
    "expected",
    "detail",
    "evidence",
)
RULE_VERSION = "etf-readonly-healthcheck.v2"
# Rules are copied/adapted, not imported from or executed by the sealed service.
RULE_ORIGIN = {
    "existing": "quant_workflow/scripts/quant_data_contract.py:_validate_bar_frame,_validate_paired_snapshot_components",
    "extracted_from": "quant_data@a1eb38bd86c2c8830bb91a3f7cf5c79e28a2e250",
    "extracted_rules": "contract.py:validate_raw_qfq_bar_pair_v3,validate_canonical_bar_price_auxiliary_v1; production_v2_fragments.py:_assert_qfq_values",
    "adaptation": "Retain declared legacy lots and affine price offsets; missing amount is unknown, not zero; no governance dependencies.",
}


def _value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    return str(value) if not isinstance(value, (str, int, bool)) else value


def _date(value: Any) -> str:
    return "" if pd.isna(value) else pd.Timestamp(value).date().isoformat()


def _finding(
    rule,
    symbol,
    component,
    field,
    *,
    severity="confirmed_error",
    date="",
    date_end="",
    observed=None,
    expected=None,
    detail="",
    affected_rows=1,
    evidence="",
):
    return dict(
        zip(
            FINDING_FIELDS,
            (
                severity,
                rule,
                symbol,
                date,
                date_end or date,
                component,
                field,
                affected_rows,
                _value(observed),
                _value(expected),
                detail,
                evidence,
            ),
            strict=True,
        )
    )


def _mask_findings(
    mask,
    dates,
    *,
    rule,
    symbol,
    component,
    field,
    values=None,
    expected=None,
    severity="confirmed_error",
    detail="",
    compress=False,
):
    positions = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(positions):
        return []
    groups = (
        np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1)
        if compress
        else [[p] for p in positions]
    )
    return [
        _finding(
            rule,
            symbol,
            component,
            field,
            severity=severity,
            date=_date(dates.iloc[group[0]]),
            date_end=_date(dates.iloc[group[-1]]),
            affected_rows=len(group),
            observed=values.iloc[group[0]] if values is not None else None,
            expected=expected,
            detail=detail,
        )
        for group in groups
    ]


def _numeric(frame, field):
    if field not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[field], errors="coerce").astype(float)


def _frame_dates(frame):
    if "date" not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame["date"], errors="coerce").dt.normalize()


def inspect_etf_frames(
    symbol: str,
    raw: pd.DataFrame,
    qfq: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    auxiliary_semantics: str | None = None,
    split_events: list[dict] | None = None,
) -> list[dict]:
    """Collect all local contradictions without cleaning or changing the inputs."""
    raw, qfq, factors = (frame.reset_index(drop=True) for frame in (raw, qfq, factors))
    issues = []
    aligned = True
    dates_by_component = {}
    for component, frame in (("raw", raw), ("qfq", qfq), ("factors", factors)):
        frame = frame.reset_index(drop=True)
        dates = _frame_dates(frame)
        dates_by_component[component] = dates
        if "date" not in frame:
            issues.append(_finding("missing_column", symbol, component, "date"))
        issues.extend(
            _mask_findings(
                dates.isna(),
                dates,
                rule="invalid_date",
                symbol=symbol,
                component=component,
                field="date",
            )
        )
        issues.extend(
            _mask_findings(
                dates.duplicated(keep=False),
                dates,
                rule="duplicate_key",
                symbol=symbol,
                component=component,
                field="date",
            )
        )
        if not dates.dropna().is_monotonic_increasing:
            issues.append(
                _finding(
                    "date_order",
                    symbol,
                    component,
                    "date",
                    date=_date(dates.min()),
                    date_end=_date(dates.max()),
                    detail="日期没有按时间升序排列",
                )
            )
        if "symbol" in frame:
            issues.extend(
                _mask_findings(
                    frame.symbol.astype("string").ne(symbol).fillna(True),
                    dates,
                    rule="symbol_identity",
                    symbol=symbol,
                    component=component,
                    field="symbol",
                    values=frame.symbol,
                    expected=symbol,
                )
            )
        if component == "factors":
            continue
        # Reuse the existing contract. On failure locate every offending field,
        # rather than stopping at the first exception or dropping the bad rows.
        try:
            _validate_bar_frame(frame, Path(f"{symbol}-{component}.parquet"))
        except DataContractError:
            numbers = {field: _numeric(frame, field) for field in FIELDS}
            for field in FIELDS[:-1]:
                if field not in frame:
                    issues.append(_finding("missing_column", symbol, component, field))
                    continue
                invalid = ~np.isfinite(numbers[field]) | (
                    numbers[field] <= 0 if field in PRICE_FIELDS else numbers[field] < 0
                )
                issues.extend(
                    _mask_findings(
                        invalid,
                        dates,
                        rule="invalid_numeric",
                        symbol=symbol,
                        component=component,
                        field=field,
                        values=frame[field],
                    )
                )
            if set(PRICE_FIELDS).issubset(frame):
                upper = pd.concat(
                    [numbers[f] for f in ("open", "low", "close")], axis=1
                ).max(axis=1)
                lower = pd.concat(
                    [numbers[f] for f in ("open", "high", "close")], axis=1
                ).min(axis=1)
                for field, bad in (
                    (
                        "high",
                        (numbers["high"] < upper)
                        & ~np.isclose(
                            numbers["high"],
                            upper,
                            rtol=BAR_OHLC_RELATIVE_TOLERANCE,
                            atol=BAR_OHLC_ABSOLUTE_TOLERANCE,
                        ),
                    ),
                    (
                        "low",
                        (numbers["low"] > lower)
                        & ~np.isclose(
                            numbers["low"],
                            lower,
                            rtol=BAR_OHLC_RELATIVE_TOLERANCE,
                            atol=BAR_OHLC_ABSOLUTE_TOLERANCE,
                        ),
                    ),
                ):
                    issues.extend(
                        _mask_findings(
                            bad,
                            dates,
                            rule="ohlc_relationship",
                            symbol=symbol,
                            component=component,
                            field=field,
                            values=frame[field],
                            detail="最高/最低价与当日开收盘价矛盾",
                        )
                    )
        if "amount" in frame:
            amount = _numeric(frame, "amount")
            invalid = frame.amount.notna() & (~np.isfinite(amount) | (amount < 0))
            issues.extend(
                _mask_findings(
                    invalid,
                    dates,
                    rule="invalid_numeric",
                    symbol=symbol,
                    component=component,
                    field="amount",
                    values=frame.amount,
                )
            )
        if {"native_volume", "volume_unit", "volume"}.issubset(frame):
            native = _numeric(frame, "native_volume")
            declared = frame.volume_unit.eq("lot_100_shares_rounded") & native.notna()
            expected_volume = np.floor(native / 100.0 + 0.5)
            bad = declared & _numeric(frame, "volume").ne(expected_volume)
            for pos in np.flatnonzero(bad):
                issues.append(
                    _finding(
                        "volume_unit_conversion",
                        symbol,
                        component,
                        "volume",
                        date=_date(dates.iloc[pos]),
                        observed=frame.volume.iloc[pos],
                        expected=expected_volume.iloc[pos],
                        detail="与已声明的份数转整数手、半手向上取整规则不一致",
                    )
                )
        for field in ("preclose", "pctChg", "pct_chg"):
            if field not in frame or frame[field].isna().all():
                continue
            if auxiliary_semantics != "same_basis_previous_close":
                issues.append(
                    _finding(
                        "auxiliary_basis_unproven",
                        symbol,
                        component,
                        field,
                        severity="insufficient_evidence",
                        date=_date(dates.min()),
                        date_end=_date(dates.max()),
                        affected_rows=int(frame[field].notna().sum()),
                        detail="未声明该辅助字段是同口径前收盘/涨跌幅，不擅自套用规则",
                    )
                )
                continue
            previous = _numeric(frame, "close").shift(1)
            expected_values = (
                previous
                if field == "preclose"
                else (_numeric(frame, "close") / previous - 1) * 100
            )
            actual = _numeric(frame, field)
            bad = (
                actual.notna()
                & previous.notna()
                & ~np.isclose(actual, expected_values, rtol=2e-6, atol=1e-8)
            )
            for pos in np.flatnonzero(bad):
                issues.append(
                    _finding(
                        "auxiliary_mismatch",
                        symbol,
                        component,
                        field,
                        date=_date(dates.iloc[pos]),
                        observed=actual.iloc[pos],
                        expected=expected_values.iloc[pos],
                        detail="与声明的同口径前收盘序列不一致",
                    )
                )
    raw_dates = dates_by_component["raw"]
    for component in ("qfq", "factors"):
        other = dates_by_component[component]
        if (
            raw_dates.isna().any()
            or other.isna().any()
            or raw_dates.duplicated().any()
            or other.duplicated().any()
            or not raw_dates.equals(other)
        ):
            aligned = False
            for day in sorted(
                set(raw_dates.dropna()).symmetric_difference(set(other.dropna()))
            ):
                issues.append(
                    _finding(
                        "pair_key_mismatch",
                        symbol,
                        component,
                        "date",
                        date=_date(day),
                        detail="raw、QFQ 和因子必须逐日对应",
                    )
                )
            if not any(
                i["rule"] == "pair_key_mismatch" and i["component"] == component
                for i in issues
            ):
                issues.append(
                    _finding(
                        "pair_key_mismatch",
                        symbol,
                        component,
                        "date",
                        detail="日期顺序、重复键或行数不一致",
                    )
                )
    if aligned:
        for field in ("volume", "amount"):
            left, right = _numeric(raw, field), _numeric(qfq, field)
            known = np.isfinite(left) & np.isfinite(right)
            issues.extend(
                _mask_findings(
                    known & left.ne(right),
                    raw_dates,
                    rule="raw_qfq_unadjusted_field",
                    symbol=symbol,
                    component="raw/qfq",
                    field=field,
                    values=right,
                    detail="复权不应改变成交量/成交额",
                )
            )
            if field == "amount":
                issues.extend(
                    _mask_findings(
                        left.isna() | right.isna(),
                        raw_dates,
                        rule="missing_amount",
                        symbol=symbol,
                        component="raw/qfq",
                        field=field,
                        severity="insufficient_evidence",
                        detail="缺少成交额；不以零或价格乘成交量补造",
                        compress=True,
                    )
                )
        scale = _numeric(factors, "factor")
        offset = (
            _numeric(factors, "offset")
            if "offset" in factors
            else pd.Series(0.0, index=factors.index)
        )
        for field, invalid in (
            ("factor", ~np.isfinite(scale) | (scale <= 0)),
            ("offset", ~np.isfinite(offset)),
        ):
            issues.extend(
                _mask_findings(
                    invalid,
                    raw_dates,
                    rule="invalid_factor",
                    symbol=symbol,
                    component="factors",
                    field=field,
                    values=factors[field] if field in factors else None,
                )
            )
        for field in PRICE_FIELDS:
            expected = _numeric(raw, field) * scale + offset
            actual = _numeric(qfq, field)
            bad = (
                np.isfinite(expected)
                & np.isfinite(actual)
                & ~np.isclose(
                    actual,
                    expected,
                    rtol=BAR_OHLC_RELATIVE_TOLERANCE,
                    atol=PAIRED_PRICE_ABSOLUTE_TOLERANCE,
                )
            )
            for pos in np.flatnonzero(bad):
                issues.append(
                    _finding(
                        "qfq_transform",
                        symbol,
                        "qfq",
                        field,
                        date=_date(raw_dates.iloc[pos]),
                        observed=actual.iloc[pos],
                        expected=expected.iloc[pos],
                        detail="与本文件声明的 raw×factor+offset 关系矛盾",
                    )
                )
        if len(scale) and (
            not np.isclose(scale.iloc[-1], 1.0) or not np.isclose(offset.iloc[-1], 0.0)
        ):
            issues.append(
                _finding(
                    "qfq_terminal_basis",
                    symbol,
                    "factors",
                    "factor/offset",
                    date=_date(raw_dates.iloc[-1]),
                    observed=f"{scale.iloc[-1]}/{offset.iloc[-1]}",
                    expected="1/0",
                    detail="末日复权基准未归一",
                )
            )
        for event in split_events or ():
            effective = pd.Timestamp(event["effective_trade_date"]).normalize()
            before = np.flatnonzero(raw_dates < effective)
            after = np.flatnonzero(raw_dates >= effective)
            if (
                not len(before)
                or not len(after)
                or raw_dates.iloc[after[0]] != effective
            ):
                issues.append(
                    _finding(
                        "split_boundary_unavailable",
                        symbol,
                        "qfq",
                        "factor",
                        severity="insufficient_evidence",
                        date=_date(effective),
                        detail="缺少公告拆分边界两侧行情，不能推算是否正确复权",
                        evidence=event.get("source_url", ""),
                    )
                )
                continue
            left, right = before[-1], after[0]
            if effective - raw_dates.iloc[left] > pd.Timedelta(days=10):
                issues.append(
                    _finding(
                        "split_boundary_unavailable",
                        symbol,
                        "qfq",
                        "factor",
                        severity="insufficient_evidence",
                        date=_date(effective),
                        detail="拆分前行情过远，期间其他事件未排除",
                        evidence=event.get("source_url", ""),
                    )
                )
                continue
            # For a pure share split, later adjustments cancel in this ratio.
            # Never fit an event ratio to the observed price jump.
            actual_ratio = scale.iloc[right] / scale.iloc[left]
            if not np.isclose(
                actual_ratio, float(event["ratio"]), rtol=2e-6, atol=1e-8
            ):
                # A fitted affine representation can be nonunique on flat
                # bars; rounding can also perturb its scale. Only convict
                # the explicit unchanged-raw-as-QFQ case, not the fit alone.
                raw_boundary = np.column_stack(
                    [
                        _numeric(raw, field).iloc[[left, right]].to_numpy()
                        for field in PRICE_FIELDS
                    ]
                )
                qfq_boundary = np.column_stack(
                    [
                        _numeric(qfq, field).iloc[[left, right]].to_numpy()
                        for field in PRICE_FIELDS
                    ]
                )
                missing_adjustment = np.array_equal(
                    raw_boundary, qfq_boundary
                ) and np.isclose(actual_ratio, 1.0, rtol=1e-12, atol=1e-12)
                issues.append(
                    _finding(
                        "split_adjustment_missing"
                        if missing_adjustment
                        else "split_ratio_disagreement",
                        symbol,
                        "qfq/factors",
                        "factor_ratio",
                        severity="confirmed_error"
                        if missing_adjustment
                        else "needs_review",
                        date=_date(effective),
                        observed=actual_ratio,
                        expected=float(event["ratio"]),
                        detail=(
                            f"与所提供纯拆分/合并记录矛盾；前一行情日{_date(raw_dates.iloc[left])}，边界两侧QFQ仍完全等于raw"
                            if missing_adjustment
                            else "拟合因子比例与公告不同，可能涉及精度或非唯一的平价仿射表示，不能只凭比例直接判错"
                        ),
                        evidence=event.get("source_url", ""),
                    )
                )
    if len(factors):
        # Algebraic consistency is not independent evidence of corporate actions.
        sources = factors.get(
            "factor_source", pd.Series("unknown", index=factors.index)
        ).astype(str)
        unproven = sources.str.contains(
            "derived_same_provider|derived_close_ratio|unknown", regex=True
        )
        issues.extend(
            _mask_findings(
                unproven,
                dates_by_component["factors"],
                rule="factor_authority_unproven",
                symbol=symbol,
                component="factors",
                field="factor/offset",
                severity="insufficient_evidence",
                detail="因子来自同组价格推导，关系通过不等于分红/拆分历史已经独立证实",
                compress=True,
            )
        )
    if "source_id" not in raw:
        issues.append(
            _finding(
                "source_reference_missing",
                symbol,
                "raw",
                "source_id",
                severity="insufficient_evidence",
                date=_date(raw_dates.min()),
                date_end=_date(raw_dates.max()),
                affected_rows=len(raw),
                detail="缺少逐行来源引用",
            )
        )
    else:
        missing = raw.source_id.isna() | raw.source_id.astype(str).str.strip().eq("")
        issues.extend(
            _mask_findings(
                missing,
                raw_dates,
                rule="source_reference_missing",
                symbol=symbol,
                component="raw",
                field="source_id",
                severity="insufficient_evidence",
                compress=True,
            )
        )
    if "close" in qfq:
        returns = _numeric(qfq, "close").pct_change(fill_method=None)
        resolved_days = {
            item["date"]
            for item in issues
            if item["rule"] == "split_adjustment_missing"
        }
        for position in np.flatnonzero(returns.abs() > 0.30):
            day = _date(dates_by_component["qfq"].iloc[position])
            if day in resolved_days:
                continue
            issues.append(
                _finding(
                    "large_adjusted_return",
                    symbol,
                    "qfq",
                    "close",
                    date=day,
                    observed=qfq.close.iloc[position],
                    severity="needs_review",
                    detail=f"相对前一条记录涨跌{returns.iloc[position] * 100:+.6f}%；仅列核查线索，不据此判错",
                )
            )
    return issues


def _state(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _child(root, relative):
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ValueError("invalid relative input path")
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("input path uses a symlink")
    current.resolve().relative_to(root.resolve())
    return current


class _ReadSet:
    def __init__(self):
        self.states = {}
        self.records = []
        self.bytes = 0
        self.decodes = 0

    def read(self, path, expected=None):
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_size > 128 * 1024 * 1024:
            raise ValueError("input is not a bounded regular file")
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            body = stream.read(128 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        if (
            _state(before) != _state(opened)
            or _state(opened) != _state(after)
            or _state(after) != _state(path.lstat())
        ):
            raise ValueError("input changed while being read")
        self.states[path] = _state(after)
        digest = hashlib.sha256(body).hexdigest()
        self.bytes += len(body)
        self.records.append(
            {
                "path": str(path),
                "sha256": digest,
                "size_bytes": len(body),
                "expected_sha256": expected,
            }
        )
        if expected is not None and digest != expected:
            raise ValueError("file SHA256 differs from the frozen index")
        return body

    def parquet(self, path, expected=None):
        body = self.read(path, expected)
        self.decodes += 1
        return pd.read_parquet(io.BytesIO(body))

    def json(self, path, expected=None):
        payload = json.loads(self.read(path, expected))
        if not isinstance(payload, dict):
            raise ValueError("JSON input is not an object")
        return payload

    def unchanged(self):
        for path, previous in self.states.items():
            try:
                if _state(path.lstat()) != previous:
                    return False
            except OSError:
                return False
        return True

    def content_digest(self, snapshot, manifest):
        """Same existing snapshot hash algorithm, using this pass's file hashes."""
        known = {Path(item["path"]): item["sha256"] for item in self.records}
        digest = hashlib.sha256()
        for path in sorted(snapshot.rglob("*")):
            if path.name == "DATASET_MANIFEST.json" or not path.is_file():
                continue
            if path.is_symlink():
                raise ValueError("snapshot contains a symlink")
            if path not in known:
                self.read(path)
                known[path] = self.records[-1]["sha256"]
            digest.update(path.relative_to(snapshot).as_posix().encode())
            digest.update(b"\0")
            digest.update(known[path].encode("ascii"))
            digest.update(b"\n")
        stable = dict(manifest)
        stable.pop("content_sha256", None)
        digest.update(_json_bytes(stable))
        return digest.hexdigest()


def _write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _announced_absences(snapshot, source, indexes, target, readset):
    """Use operator-inspected announcements only for the exact bound snapshot.

    As with split evidence, hashes prove retained bytes, not document authenticity
    or the operator's interpretation. Empty vendor caches are never announcements.
    """
    descriptor = source.get("evidence_followup")
    if not descriptor:
        return {}, []
    accepted, issues = {}, []
    try:
        bundle = readset.json(
            _child(snapshot, descriptor["path"]), descriptor["sha256"]
        )
        if (
            bundle["dataset_id"] != "cn_etf_daily"
            or pd.isna(pd.Timestamp(bundle["checked_through"]))
            or pd.Timestamp(bundle["checked_through"]) < target
            or not isinstance(bundle["absence_intervals"], list)
        ):
            raise ValueError("absence evidence dataset/date scope differs")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {}, [_finding(
            "absence_evidence_unavailable", "*", "evidence", "absence",
            severity="insufficient_evidence", detail=str(exc),
        )]
    for record in bundle["absence_intervals"]:
        symbol = str(record.get("symbol", "*")) if isinstance(record, dict) else "*"
        try:
            start, end = (pd.Timestamp(record[key]) for key in ("start", "end"))
            if (
                pd.isna(start) or pd.isna(end) or start > end
                or start != start.normalize() or end != end.normalize()
                or end > pd.Timestamp(bundle["checked_through"])
                or record["reason"] not in {
                    "trading_suspended", "fund_liquidation", "delisted"
                }
                or not isinstance(record["sources"], list) or not record["sources"]
            ):
                raise ValueError("invalid announcement interval/reason")
            for component in ("raw", "qfq"):
                if record["component_sha256"][component] != indexes[component][symbol]["sha256"]:
                    raise ValueError("announcement binding differs from current price files")
            for document in record["sources"]:
                if (
                    not isinstance(document["sha256"], str)
                    or len(document["sha256"]) != 64
                    or not str(document["url"]).startswith(("https://", "http://"))
                ):
                    raise ValueError("announcement source reference is incomplete")
                readset.read(_child(snapshot, document["path"]), document["sha256"])
            accepted.setdefault(symbol, []).append({
                "start": start, "end": end, "reason": record["reason"],
                "detail": str(record.get("note", "")),
                "evidence": " | ".join(item["url"] for item in record["sources"]),
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(_finding(
                "absence_evidence_unavailable", symbol, "evidence", "absence",
                severity="insufficient_evidence", detail=str(exc),
            ))
    return accepted, issues


def _reference_comparison(symbol, qfq, body, evidence):
    """Replay exact saved Tencent observations; a disagreement doesn't pick a winner."""
    payload = json.loads(body)
    ticker = symbol[:2].lower() + symbol[2:]
    data = payload.get("data", {}).get(ticker, {})
    rows = data.get("qfqday")
    if not isinstance(rows, list):
        return [], {
            "symbol": symbol,
            "evidence": evidence,
            "status": "not_comparable",
            "reason": "saved body has no explicit QFQ rows",
        }
    indexed = qfq.copy()
    indexed.index = _frame_dates(qfq)
    if indexed.index.duplicated().any():
        return [], {
            "symbol": symbol,
            "evidence": evidence,
            "status": "not_comparable",
            "reason": "local keys are duplicated",
        }
    issues, dates = [], set()
    for row in rows:
        if not isinstance(row, list) or len(row) < 5:
            continue
        day = pd.Timestamp(row[0]).normalize()
        if day not in indexed.index:
            continue
        dates.add(_date(day))
        for field, pos in (("open", 1), ("close", 2), ("high", 3), ("low", 4)):
            actual, reference = float(indexed.loc[day, field]), float(row[pos])
            if not np.isclose(
                actual, reference, rtol=BAR_OHLC_RELATIVE_TOLERANCE, atol=0.00051
            ):
                issues.append(
                    _finding(
                        "external_price_disagreement",
                        symbol,
                        "qfq",
                        field,
                        severity="needs_review",
                        date=_date(day),
                        observed=actual,
                        expected=reference,
                        evidence=evidence,
                        detail="已存腾讯观察与本文件不一致，未据此认定哪一源正确",
                    )
                )
    return issues, {
        "symbol": symbol,
        "evidence": evidence,
        "status": "disagreed" if issues else "matched" if dates else "no_common_dates",
        "dates": len(dates),
        "start": min(dates) if dates else "",
        "end": max(dates) if dates else "",
        "scope": "saved observation only, not certification of full history",
    }


def audit_etf_snapshot(
    root: Path,
    *,
    output_dir: Path,
    calendar_path: Path | None = None,
    split_evidence_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    started = time.monotonic()
    root, output_dir = root.expanduser().resolve(), output_dir.expanduser().resolve()
    if output_dir == root or root in output_dir.parents:
        raise ValueError("report output must be outside the data root")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("report directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    readset, issues, summaries, references = _ReadSet(), [], [], []
    pointer = readset.json(_child(root, "parquet/etf_1d/cn_etf_daily/LATEST.json"))
    snapshot = _child(root, pointer["snapshot_path"])
    snapshot.relative_to(root / "parquet/etf_1d/cn_etf_daily/snapshots")
    manifest = readset.json(
        _child(snapshot, "DATASET_MANIFEST.json"), pointer["manifest_sha256"]
    )
    if (
        manifest.get("snapshot_id") != pointer.get("snapshot_id")
        or manifest.get("dataset_id") != "cn_etf_daily"
    ):
        raise ValueError("snapshot and pointer identity differ")
    if (root / "DATA_MANIFEST.json").exists():
        readset.read(_child(root, "DATA_MANIFEST.json"))
    indexes = {}
    for component, key in (
        ("raw", "raw_file_index_path"),
        ("qfq", "file_index_path"),
        ("factors", "factor_index_path"),
    ):
        index = readset.parquet(_child(snapshot, manifest[key]))
        needed = {
            "symbol",
            "relative_path",
            "sha256",
            "row_count",
            "first_date",
            "last_date",
        }
        if not needed.issubset(index):
            raise ValueError(f"{component} index is missing required columns")
        if index.symbol.duplicated().any():
            issues.append(_finding("index_duplicate_symbol", "*", component, "symbol"))
        indexes[component] = {
            str(row["symbol"]): row for row in index.to_dict("records")
        }
    symbols = sorted(set().union(*(set(v) for v in indexes.values())))
    if manifest.get("symbol_count") != len(symbols):
        issues.append(
            _finding(
                "manifest_symbol_count",
                "*",
                "manifest",
                "symbol_count",
                observed=manifest.get("symbol_count"),
                expected=len(symbols),
            )
        )
    source = manifest.get("source", {})
    split_events = {}
    if split_evidence_path:
        action_file = split_evidence_path.expanduser().resolve()
        actions = pd.read_csv(io.BytesIO(readset.read(action_file)))
        required = {
            "symbol",
            "effective_trade_date",
            "ratio",
            "action_type",
            "source_url",
            "evidence_path",
            "evidence_sha256",
        }
        if (
            not required.issubset(actions)
            or actions.duplicated(["symbol", "effective_trade_date"]).any()
        ):
            raise ValueError("split evidence columns/keys are invalid")
        for event in actions.to_dict("records"):
            if (
                event["action_type"] not in {"split", "reverse_split"}
                or not np.isfinite(float(event["ratio"]))
                or float(event["ratio"]) <= 0
            ):
                raise ValueError(
                    "only explicit positive pure split ratios are supported"
                )
            try:
                readset.read(
                    _child(action_file.parent, event["evidence_path"]),
                    event["evidence_sha256"],
                )
            except (OSError, ValueError) as exc:
                issues.append(
                    _finding(
                        "split_evidence_unavailable",
                        str(event["symbol"]),
                        "evidence",
                        "split_ratio",
                        severity="insufficient_evidence",
                        date=str(event["effective_trade_date"]),
                        detail=str(exc),
                        evidence=str(event["source_url"]),
                    )
                )
                continue
            split_events.setdefault(str(event["symbol"]), []).append(event)
    universe_name = source.get("universe_path")
    if universe_name:
        expected_sha = None
        if source.get("universe_manifest_path"):
            universe_meta = readset.json(
                _child(snapshot, source["universe_manifest_path"])
            )
            expected_sha = universe_meta.get("universe_file_sha256")
        universe = readset.parquet(_child(snapshot, universe_name), expected_sha)
        if (
            "symbol" not in universe
            or universe.symbol.duplicated().any()
            or set(universe.symbol.astype(str)) != set(symbols)
        ):
            issues.append(_finding("universe_mismatch", "*", "universe", "symbol"))
    if manifest.get("corporate_actions_path"):
        readset.parquet(
            _child(snapshot, manifest["corporate_actions_path"]),
            manifest.get("corporate_actions_sha256"),
        )
    calendar = None
    calendar_bounds = None
    if calendar_path:
        path = calendar_path.expanduser().resolve()
        body = readset.read(path)
        if path.suffix == ".parquet":
            readset.decodes += 1
            calendar_frame = pd.read_parquet(io.BytesIO(body))
        else:
            calendar_frame = pd.read_csv(io.BytesIO(body))
        dates = pd.to_datetime(calendar_frame["date"], errors="raise").dt.normalize()
        if dates.duplicated().any():
            raise ValueError("calendar dates are duplicated")
        flags = (
            calendar_frame.get("is_open", pd.Series(True, index=calendar_frame.index))
            .astype(str)
            .str.lower()
        )
        if not flags.isin(["true", "false", "1", "0"]).all():
            raise ValueError("calendar is_open is invalid")
        calendar = pd.DatetimeIndex(dates[flags.isin(["true", "1"])])
        calendar_bounds = (dates.min(), dates.max())
    else:
        issues.append(
            _finding(
                "calendar_evidence_unavailable",
                "*",
                "coverage",
                "date",
                severity="insufficient_evidence",
                detail="未提供独立交易日历；内部缺日只与其他ETF实际日期对比，不把缺行直接判作数据错误",
            )
        )
    target = pd.Timestamp(manifest["latest_trade_date"])
    absence_intervals, absence_issues = _announced_absences(
        snapshot, source, indexes, target, readset
    )
    issues.extend(absence_issues)
    explained_absences = []
    date_arrays = {}
    semantic_started = time.monotonic()
    for count, symbol in enumerate(symbols, 1):
        beginning = len(issues)
        frames = {}
        for component, index in indexes.items():
            row = index.get(symbol)
            if row is None:
                issues.append(
                    _finding("component_missing", symbol, component, "symbol")
                )
                continue
            try:
                if not isinstance(row["sha256"], str) or len(row["sha256"]) != 64:
                    raise ValueError("invalid declared SHA256")
                frame = readset.parquet(
                    _child(snapshot, row["relative_path"]), row["sha256"]
                )
                frames[component] = frame
                dates = _frame_dates(frame)
                if (
                    len(frame) != int(row["row_count"])
                    or _date(dates.min()) != str(row["first_date"])
                    or _date(dates.max()) != str(row["last_date"])
                ):
                    issues.append(
                        _finding(
                            "index_metadata",
                            symbol,
                            component,
                            "row_count/date",
                            detail="索引行数或日期边界与实际文件不同",
                        )
                    )
            except (OSError, ValueError, TypeError, KeyError) as exc:
                issues.append(
                    _finding(
                        "file_identity",
                        symbol,
                        component,
                        "file",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
        if len(frames) == 3:
            issues.extend(
                inspect_etf_frames(
                    symbol,
                    **frames,
                    auxiliary_semantics=source.get("auxiliary_fields_semantics"),
                    split_events=split_events.get(symbol),
                )
            )
            dates = _frame_dates(frames["raw"])
            valid_dates = dates.dropna()
            if len(valid_dates):
                date_arrays[symbol] = valid_dates.to_numpy(dtype="datetime64[ns]")
            metadata = source.get("historical_qfq_rebuilds", {}).get(symbol, {})
            for label, desc in metadata.get("frozen_input_files", {}).items():
                try:
                    evidence = readset.read(
                        _child(snapshot, desc["path"]), desc["sha256"]
                    )
                    if label.startswith("tencent_") and label.endswith(".body"):
                        more, reference = _reference_comparison(
                            symbol, frames["qfq"], evidence, desc["path"]
                        )
                        issues.extend(more)
                        references.append(reference)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    issues.append(
                        _finding(
                            "source_evidence_identity",
                            symbol,
                            "evidence",
                            label,
                            detail=str(exc),
                        )
                    )
        raw = frames.get("raw", pd.DataFrame())
        dates = _frame_dates(raw)
        last = dates.max()
        unit_rows = (
            int(
                (
                    raw.volume_unit.eq("lot_100_shares_rounded")
                    & _numeric(raw, "native_volume").notna()
                ).sum()
            )
            if {"native_volume", "volume_unit"}.issubset(raw)
            else 0
        )
        summaries.append(
            {
                "symbol": symbol,
                "first_date": _date(dates.min()),
                "last_date": _date(last),
                "target_date": _date(target),
                "raw_rows": len(raw),
                "qfq_rows": len(frames.get("qfq", [])),
                "native_volume_checked_rows": unit_rows,
                "at_target": bool(pd.notna(last) and last >= target),
                "confirmed_error": sum(
                    i["severity"] == "confirmed_error" for i in issues[beginning:]
                ),
                "insufficient_evidence": sum(
                    i["severity"] == "insufficient_evidence" for i in issues[beginning:]
                ),
                "needs_review": sum(
                    i["severity"] == "needs_review" for i in issues[beginning:]
                ),
            }
        )
        if progress and (count % 100 == 0 or count == len(symbols)):
            progress(
                f"{count}/{len(symbols)} ETF; elapsed={time.monotonic() - started:.2f}s; decodes={readset.decodes}"
            )
    semantic_seconds = time.monotonic() - semantic_started
    observed_days = (
        np.unique(np.concatenate(list(date_arrays.values())))
        if date_arrays
        else np.array([], dtype="datetime64[ns]")
    )
    sessions = calendar.to_numpy() if calendar is not None else observed_days
    for symbol, dates in date_arrays.items():
        within = sessions[
            (sessions >= dates.min()) & (sessions <= np.datetime64(target))
        ]
        absent = np.setdiff1d(within, dates)
        last = dates.max()
        for day in absent:
            explanation = next((
                item for item in absence_intervals.get(symbol, ())
                if item["start"] <= pd.Timestamp(day) <= item["end"]
            ), None)
            if explanation is not None:
                explained_absences.append({
                    "symbol": symbol, "date": _date(day),
                    "reason": explanation["reason"], "detail": explanation["detail"],
                    "evidence": explanation["evidence"],
                })
                continue
            issues.append(
                _finding(
                    "tail_not_covered" if day > last else "missing_session_unexplained",
                    symbol,
                    "raw/qfq",
                    "bar",
                    severity="insufficient_evidence",
                    date=_date(day),
                    detail="该交易日没有行且本次未核实停牌/终止上市证明；不生成假bar"
                    if calendar is not None
                    else "其他ETF在该日有记录，本证券无行；缺少独立日历及停牌证明，待核实",
                )
            )
        if calendar is not None:
            for day in np.setdiff1d(dates, sessions):
                if calendar_bounds[0] <= pd.Timestamp(day) <= calendar_bounds[1]:
                    issues.append(
                        _finding(
                            "closed_day_bar",
                            symbol,
                            "raw/qfq",
                            "date",
                            date=_date(day),
                            evidence=str(calendar_path),
                            detail="所提供日历在该日期不开放",
                        )
                    )
    if manifest.get("content_sha256"):
        try:
            actual_content = readset.content_digest(snapshot, manifest)
            if actual_content != manifest["content_sha256"]:
                issues.append(
                    _finding(
                        "snapshot_content_identity",
                        "*",
                        "snapshot",
                        "content_sha256",
                        observed=actual_content,
                        expected=manifest["content_sha256"],
                        detail="快照实际文件/清单组合与冻结摘要不同",
                    )
                )
        except (OSError, ValueError) as exc:
            issues.append(
                _finding(
                    "snapshot_content_identity",
                    "*",
                    "snapshot",
                    "content_sha256",
                    detail=str(exc),
                )
            )
    symbol_counts = Counter((i["symbol"], i["severity"]) for i in issues)
    for summary in summaries:
        for severity in ("confirmed_error", "insufficient_evidence", "needs_review"):
            summary[severity] = symbol_counts[(summary["symbol"], severity)]
    unchanged = readset.unchanged()
    if not unchanged:
        issues.append(
            _finding(
                "input_changed_during_audit",
                "*",
                "snapshot",
                "identity",
                detail="检查期间输入发生变化，本次报告不代表一个稳定版本",
            )
        )
    counts = Counter(i["severity"] for i in issues)
    for severity in ("confirmed_error", "insufficient_evidence", "needs_review"):
        counts.setdefault(severity, 0)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss if resource else None
    report = {
        "schema_version": 1,
        "rule_version": RULE_VERSION,
        "rule_origin": RULE_ORIGIN,
        "status": "completed" if unchanged else "incomplete",
        "input_unchanged": unchanged,
        "assessment": "confirmed_errors_found"
        if counts["confirmed_error"]
        else "local_consistency_checked_evidence_gaps_remain"
        if counts["insufficient_evidence"]
        else "local_checks_passed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "snapshot_id": pointer["snapshot_id"],
        "manifest_sha256": pointer["manifest_sha256"],
        "symbol_count": len(symbols),
        "symbols_scanned": len(summaries),
        "raw_rows": sum(i["raw_rows"] for i in summaries),
        "native_volume_checked_rows": sum(
            i["native_volume_checked_rows"] for i in summaries
        ),
        "symbols_at_target": sum(i["at_target"] for i in summaries),
        "explained_absence_sessions": len(explained_absences),
        "counts": dict(counts),
        "rules": dict(Counter(i["rule"] for i in issues)),
        "affected_symbols": {
            s: len(
                {
                    i["symbol"]
                    for i in issues
                    if i["severity"] == s and i["symbol"] != "*"
                }
            )
            for s in counts
        },
        "io": {
            "full_hashes": len(readset.records),
            "bytes_hashed": readset.bytes,
            "parquet_decodes": readset.decodes,
        },
        "timing": {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "semantic_seconds": round(semantic_seconds, 3),
        },
        "peak_rss_mib": round(peak / (1024**2 if sys.platform == "darwin" else 1024), 2)
        if peak
        else None,
        "external_replays": references,
        "network_requests": 0,
        "output_dir": str(output_dir),
        "split_evidence": {
            "path": str(split_evidence_path) if split_evidence_path else None,
            "accepted_events": sum(map(len, split_events.values())),
            "boundary": "Source documents are fixed by hash; the operator must verify the stated event ratio. The tool does not authenticate announcements or select ratios automatically.",
        },
        "limits": [
            "检查关系一致，不独立认证全部供应商历史价格",
            "没有原始份数及明确单位声明的旧记录，不能独立证明单位转换正确",
            "未提供日历/停牌证据时缺日为证据不足",
            "同类连续缺失按起止日期和实际行数汇总，数值矛盾逐日列出",
            "此报告不改变数据、不授权发布、不影响查询或回测",
            "公告缺日解释仅适用于绑定文件和日期；工具核对字节摘要，不自动认证公告出处或替代人工核读；不会填造行情或推进实际行情末日",
        ],
    }
    _write_csv(output_dir / "findings.csv", issues, FINDING_FIELDS)
    _write_csv(
        output_dir / "explained_absences.csv", explained_absences,
        ["symbol", "date", "reason", "detail", "evidence"],
    )
    _write_csv(
        output_dir / "symbols.csv",
        summaries,
        list(summaries[0]) if summaries else ["symbol"],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "read-set.json").write_text(
        json.dumps(readset.records, ensure_ascii=False, indent=2) + "\n"
    )
    top = Counter(i["rule"] for i in issues).most_common(15)
    text = (
        f"# ETF只读体检\n\n版本：`{pointer['snapshot_id']}`\n\n检查 {len(summaries)}/{len(symbols)} 只，{report['raw_rows']:,} 条raw记录；{report['symbols_at_target']}只到目标日期。\n\n确定错误 {counts['confirmed_error']} 项；证据不足 {counts['insufficient_evidence']} 项；核查线索 {counts['needs_review']} 项。计数为报告条目，不是证券数。\n\n| 规则 | 条目数 |\n|---|---:|\n"
        + "".join(f"| {rule} | {count} |\n" for rule, count in top)
    )
    text += (
        "\n详单见 findings.csv（证券、日期、字段、原值/对照值）；逐证券汇总见 symbols.csv。\n\n"
        + f"经核读公告解释的无行情交易日 {len(explained_absences)} 项，另见 explained_absences.csv；实际行情末日不变。\n\n"
        + "\n".join(f"- {limit}" for limit in report["limits"])
        + f"\n\n输入未变：{unchanged}；耗时 {report['timing']['elapsed_seconds']} 秒；峰值RSS {report['peak_rss_mib']} MiB。\n"
    )
    (output_dir / "REPORT.md").write_text(text)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=quant_data_root())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--calendar", type=Path)
    parser.add_argument(
        "--split-evidence",
        type=Path,
        help="Locally verified pure split records and exact source document references (CSV)",
    )
    args = parser.parse_args(argv)
    output = args.output_dir or reports_root() / "data_checks" / datetime.now(
        timezone.utc
    ).strftime("etf-%Y%m%dT%H%M%SZ")
    try:
        report = audit_etf_snapshot(
            args.root,
            output_dir=output,
            calendar_path=args.calendar,
            split_evidence_path=args.split_evidence,
            progress=lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ETF check incomplete: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
