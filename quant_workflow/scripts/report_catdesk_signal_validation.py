#!/usr/bin/env python3
"""Validate CatDesk candidates with DSA/Qlib-style filters.

This report is a retrospective research artifact:

- Selection uses the CatDesk report date, local bars up to the signal date,
  DSA market-regime output, and the latest available Qlib score artifact.
- Validation uses the next trading day's close/snapshot and is kept separate
  from the selection score to avoid future leakage.
- The script writes reports only; it does not mutate quant_data, Qlib providers,
  broker software, or CatDesk state.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import projects_root, quant_data_root, reports_root  # noqa: E402
from _symbols import to_qlib_name  # noqa: E402
from _trading_calendar import (  # noqa: E402
    WEEKDAY_FALLBACK_SOURCE,
    load_trading_calendar,
)
from market_data_clients import fetch_sina_a_snapshots  # noqa: E402
from report_qlib_predictions import (  # noqa: E402
    find_prediction_path,
    load_prediction_scores,
)


DEFAULT_SIGNAL_DATE = "2026-07-08"
DEFAULT_TOP_N = 8
DEFAULT_SELECTED_STRATEGY = "dominant_theme_bucket"


STRATEGY_LABELS = {
    "catdesk_raw": "CatDesk 原始分",
    "dsa_conservative": "DSA 保守风控",
    "factor_balance": "因子均衡规则",
    "qlib_priority": "Qlib 分位优先",
    "theme_momentum": "题材弹性",
    "theme_momentum_qlib": "题材弹性 + Qlib 正向确认",
    "dominant_theme_bucket": "主线题材桶",
}

STRATEGY_DESCRIPTIONS = {
    "catdesk_raw": "只看 CatDesk 原始候选分，作为九点猫研自身排序基准。",
    "dsa_conservative": "当前保守规则：CatDesk 分 + 评级/胜率/动量/流动性 + Qlib 小权重 - 追高/急跌/风险关闭惩罚。",
    "factor_balance": "手写因子均衡规则：重视动量、流动性、历史命中率和分析师确信度；不调用外部选股适配层。",
    "qlib_priority": "更重视 Qlib 分位，适合看模型是否能在候选池内进一步排序。",
    "theme_momentum": "更重视题材强度和候选自身弹性，适合捕捉强主题扩散。",
    "theme_momentum_qlib": "在题材弹性基础上加入 Qlib 正向确认；Qlib 低分只轻惩罚，避免错杀高弹性主题股。",
    "dominant_theme_bucket": "先识别最强题材，并给最强题材保留固定席位；适合检验主线行情的板块内扩散。",
}

CHINA_TZ = timezone(timedelta(hours=8))
MORNING_CUTOFF_HOUR = 9
MORNING_CUTOFF_MINUTE = 30
MARKET_REGIME_PATTERN = re.compile(r"^market_regime_(\d{8})_(\d{6})\.json$")


@dataclass(frozen=True)
class PricePoint:
    close: float | None = None
    pct_chg: float | None = None
    amount: float | None = None
    turn: float | None = None
    source: str = ""
    requested_date: str = ""
    trade_date: str = ""
    fallback_reason: str = ""


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CHINA_TZ)
    return parsed.astimezone(CHINA_TZ)


def validation_cutoff_iso(validation_date: str) -> str:
    day = datetime.strptime(validation_date, "%Y-%m-%d").date()
    cutoff = datetime(
        day.year,
        day.month,
        day.day,
        MORNING_CUTOFF_HOUR,
        MORNING_CUTOFF_MINUTE,
        tzinfo=CHINA_TZ,
    )
    return cutoff.isoformat(timespec="seconds")


def epoch_ms_to_china_iso(value: Any) -> str:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return ""
    if millis <= 0:
        return ""
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc).astimezone(CHINA_TZ).isoformat(timespec="seconds")


def classify_catdesk_report(payload: dict[str, Any], signal_date: str) -> dict[str, Any]:
    generated_at = payload.get("generated_at")
    report_date = payload.get("date")
    cutoff = validation_cutoff_iso(signal_date)
    generated_dt = _parse_datetime(generated_at)
    cutoff_dt = _parse_datetime(cutoff)
    status = "as_of_morning"
    reason = ""

    if report_date and report_date != signal_date:
        status = "date_mismatch"
        reason = f"CatDesk report date {report_date} does not match signal date {signal_date}."
    elif generated_dt is None:
        status = "unknown_generated_at"
        reason = "CatDesk report has no parseable generated_at timestamp."
    elif cutoff_dt is not None and generated_dt > cutoff_dt:
        status = "backfilled_sample"
        reason = f"CatDesk report generated at {generated_dt.isoformat(timespec='seconds')} after {cutoff}."

    return {
        "catdesk_report_status": status,
        "catdesk_report_reason": reason,
        "catdesk_report_date": report_date,
        "catdesk_generated_at": generated_at,
        "catdesk_morning_cutoff": cutoff,
    }


def business_day_pairs(start_signal_date: str, days: int) -> list[tuple[str, str]]:
    calendar = load_trading_calendar()
    pairs: list[tuple[str, str]] = []
    signal_day = calendar.on_or_previous(_parse_date(start_signal_date).date()).day
    for _ in range(max(days, 1)):
        validation_day = calendar.next(signal_day).day
        pairs.append((signal_day.isoformat(), validation_day.isoformat()))
        signal_day = calendar.previous(signal_day).day
    return pairs


def resolve_validation_date(signal_date: str, validation_date: str | None) -> tuple[str, str]:
    signal_day = _parse_date(signal_date).date()
    decision = load_trading_calendar().next(signal_day)
    expected = decision.day.isoformat()
    if validation_date is None:
        return expected, decision.source

    requested_day = _parse_date(validation_date).date()
    if requested_day <= signal_day:
        raise ValueError("validation date must be after signal date")
    if validation_date != expected:
        raise ValueError(
            f"validation date must be the next A-share trading day after {signal_date}: expected {expected}, got {validation_date}"
        )
    return validation_date, decision.source


def _candidate_code(item: dict[str, Any]) -> str:
    candidate = item.get("candidate") or {}
    raw = candidate.get("code") or item.get("code") or ""
    return str(raw).strip().zfill(6)


def load_catdesk_candidates(path: Path, *, candidate_set: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_json(path)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for theme_index, theme_payload in enumerate(payload.get("themes") or [], start=1):
        theme = theme_payload.get("theme") or {}
        signal = theme_payload.get("signal") or {}
        theme_name = theme.get("name") or theme.get("title") or f"theme-{theme_index}"
        items = theme_payload.get(candidate_set) or []
        for rank, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            code = _candidate_code(item)
            if not code or code in seen:
                continue
            seen.add(code)
            candidates.append(
                {
                    "theme": theme_name,
                    "theme_rank": rank,
                    "theme_signal": signal,
                    "raw": item,
                }
            )
    return payload, candidates


def read_price_point(source_dir: Path, code: str, date: str) -> PricePoint:
    path = source_dir / f"{code}.parquet"
    if not path.exists():
        return PricePoint(source="", requested_date=date)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return PricePoint(source="", requested_date=date)
    if "date" not in frame.columns:
        return PricePoint(source="", requested_date=date)
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    row = frame.loc[dates == date].tail(1)
    if row.empty:
        return PricePoint(source="", requested_date=date)
    record = row.iloc[0]
    return PricePoint(
        close=_safe_float(record.get("close")),
        pct_chg=_safe_float(record.get("pctChg")),
        amount=_safe_float(record.get("amount")),
        turn=_safe_float(record.get("turn")),
        source=str(source_dir.name),
        requested_date=date,
        trade_date=date,
    )


def read_price_point_on_or_before(source_dir: Path, code: str, date: str) -> PricePoint:
    path = source_dir / f"{code}.parquet"
    if not path.exists():
        return PricePoint(source="", requested_date=date)
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return PricePoint(source="", requested_date=date)
    if "date" not in frame.columns:
        return PricePoint(source="", requested_date=date)

    parsed_dates = pd.to_datetime(frame["date"], errors="coerce")
    requested = pd.Timestamp(date)
    eligible = frame.loc[parsed_dates <= requested].copy()
    if eligible.empty:
        return PricePoint(source="", requested_date=date)

    eligible["_parsed_date"] = parsed_dates.loc[eligible.index]
    record = eligible.sort_values("_parsed_date").iloc[-1]
    trade_date = pd.Timestamp(record["_parsed_date"]).strftime("%Y-%m-%d")
    return PricePoint(
        close=_safe_float(record.get("close")),
        pct_chg=_safe_float(record.get("pctChg")),
        amount=_safe_float(record.get("amount")),
        turn=_safe_float(record.get("turn")),
        source=str(source_dir.name),
        requested_date=date,
        trade_date=trade_date,
        fallback_reason="" if trade_date == date else "requested_date_missing_use_previous_trade_date",
    )


def load_market_regime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "regime": "unknown",
            "position_ratio": None,
            "cash_ratio": None,
            "source": "",
        }
    payload = _read_json(path)
    return {
        "regime": payload.get("regime") or "unknown",
        "position_ratio": _safe_float(payload.get("position_ratio")),
        "cash_ratio": _safe_float(payload.get("cash_ratio")),
        "source": str(path),
        "generated_at": payload.get("generated_at"),
    }


def find_market_regime_path(reports_dir: Path, signal_date: str) -> tuple[Path, dict[str, Any]]:
    target = signal_date.replace("-", "")
    candidates: list[tuple[str, str, Path]] = []
    for path in reports_dir.glob("market_regime_*.json"):
        match = MARKET_REGIME_PATTERN.match(path.name)
        if not match:
            continue
        report_day, report_time = match.groups()
        if report_day <= target:
            candidates.append((report_day, report_time, path))

    if not candidates:
        missing = reports_dir / f"market_regime_{target}_missing.json"
        return missing, {
            "market_regime_status": "missing_as_of_signal_date",
            "market_regime_reason": f"No market regime report on or before {signal_date}.",
            "market_regime_as_of_date": "",
        }

    report_day, _report_time, path = sorted(candidates, key=lambda item: (item[0], item[1], str(item[2])))[-1]
    status = "as_of_signal_date" if report_day == target else "previous_available"
    return path, {
        "market_regime_status": status,
        "market_regime_reason": "" if status == "as_of_signal_date" else f"Using previous market regime report {report_day}.",
        "market_regime_as_of_date": f"{report_day[:4]}-{report_day[4:6]}-{report_day[6:]}",
    }


def _extract_run_id_from_prediction_path(pred_path: Path) -> str:
    try:
        if pred_path.parent.name == "artifacts":
            return pred_path.parent.parent.name
    except IndexError:
        return ""
    return ""


def load_qlib_run_metadata(pred_path: Path) -> dict[str, Any]:
    run_id = _extract_run_id_from_prediction_path(pred_path)
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "prediction_path": str(pred_path),
    }

    latest_training = reports_root() / "latest_training_result.json"
    if latest_training.exists():
        try:
            payload = _read_json(latest_training)
        except RuntimeError:
            payload = {}
        if not run_id or payload.get("run_id") == run_id:
            metadata.update(
                {
                    "run_id": payload.get("run_id") or run_id,
                    "experiment_name": payload.get("experiment_name"),
                    "run_generated_at": payload.get("generated_at"),
                    "provider": payload.get("provider"),
                    "provider_uri": payload.get("provider_uri"),
                    "source_config": payload.get("source_config"),
                    "runtime_config": payload.get("runtime_config"),
                    "training_params": {
                        key: value
                        for key, value in (payload.get("params") or {}).items()
                        if key
                        in {
                            "dataset.kwargs.handler.kwargs.end_time",
                            "dataset.kwargs.segments.train",
                            "dataset.kwargs.segments.valid",
                            "dataset.kwargs.segments.test",
                        }
                    },
                }
            )
            return metadata

    meta_path = pred_path.parent.parent / "meta.yaml" if pred_path.parent.name == "artifacts" else pred_path.parent / "meta.yaml"
    if meta_path.exists():
        try:
            lines = meta_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            if line.startswith("run_id:"):
                metadata["run_id"] = line.split(":", 1)[1].strip()
            elif line.startswith("end_time:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    metadata["run_end_time_ms"] = int(raw)
                    metadata.setdefault("run_generated_at", epoch_ms_to_china_iso(raw))
                except ValueError:
                    pass
            elif line.startswith("start_time:"):
                raw = line.split(":", 1)[1].strip()
                try:
                    metadata["run_start_time_ms"] = int(raw)
                    metadata.setdefault("run_started_at", epoch_ms_to_china_iso(raw))
                except ValueError:
                    pass
    return metadata


def _qlib_unavailable(reason: str, *, pred_path: Path | None = None, signal_date: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "available": False,
        "reason": reason,
        "prediction_path": str(pred_path) if pred_path else "",
        "latest_date": "",
        "scores": {},
        "percentiles": {},
        "signal_date": signal_date,
    }
    if metadata:
        payload.update({key: value for key, value in metadata.items() if key not in {"scores", "percentiles"}})
    return payload


def load_latest_qlib_scores(
    signal_date: str,
    *,
    top_reference_date: str | None = None,
    run_cutoff: str | None = None,
) -> dict[str, Any]:
    try:
        pred_path = find_prediction_path()
    except Exception as exc:
        return _qlib_unavailable(str(exc), signal_date=signal_date)

    metadata = load_qlib_run_metadata(pred_path)
    run_generated_at = metadata.get("run_generated_at")
    run_dt = _parse_datetime(run_generated_at)
    cutoff_dt = _parse_datetime(run_cutoff)
    if run_cutoff and cutoff_dt and run_dt is None:
        return _qlib_unavailable(
            "Qlib run generation time is unavailable; strict validation refuses to use it.",
            pred_path=pred_path,
            signal_date=signal_date,
            metadata=metadata,
        )
    if run_cutoff and run_dt and cutoff_dt and run_dt > cutoff_dt:
        return _qlib_unavailable(
            f"Qlib run generated at {run_dt.isoformat(timespec='seconds')} after validation cutoff {run_cutoff}.",
            pred_path=pred_path,
            signal_date=signal_date,
            metadata=metadata,
        )

    try:
        scores = load_prediction_scores(pred_path)
    except Exception as exc:
        return _qlib_unavailable(str(exc), pred_path=pred_path, signal_date=signal_date, metadata=metadata)

    latest_date = pd.Timestamp(scores["datetime"].max()).normalize()
    if top_reference_date:
        ref = pd.Timestamp(top_reference_date).normalize()
        eligible = scores.loc[scores["datetime"].dt.normalize() <= ref].copy()
        if eligible.empty:
            return {
                "available": False,
                "reason": f"No Qlib prediction rows on or before {top_reference_date}.",
                "prediction_path": str(pred_path),
                "latest_date": "",
                "scores": {},
                "percentiles": {},
                "signal_date": signal_date,
                **{key: value for key, value in metadata.items() if key not in {"scores", "percentiles"}},
            }
        latest_date = pd.Timestamp(eligible["datetime"].max()).normalize()
        scores = eligible

    latest_rows = scores.loc[scores["datetime"].dt.normalize() == latest_date].copy()
    latest_rows = latest_rows.dropna(subset=["score"]).sort_values("score")
    if latest_rows.empty:
        return {
            "available": False,
            "reason": "Qlib prediction artifact has no latest rows.",
            "prediction_path": str(pred_path),
            "latest_date": str(latest_date.date()),
            "scores": {},
            "percentiles": {},
            **{key: value for key, value in metadata.items() if key not in {"scores", "percentiles"}},
        }

    score_map = {str(row.instrument): _safe_float(row.score) for row in latest_rows.itertuples(index=False)}
    ranked = latest_rows.reset_index(drop=True)
    denom = max(len(ranked) - 1, 1)
    percentiles = {
        str(row.instrument): index / denom
        for index, row in enumerate(ranked.itertuples(index=False))
    }
    return {
        "available": True,
        "reason": "",
        "prediction_path": str(pred_path),
        "latest_date": str(latest_date.date()),
        "scores": score_map,
        "percentiles": percentiles,
        "row_count": int(len(latest_rows)),
        "signal_date": signal_date,
        **{key: value for key, value in metadata.items() if key not in {"scores", "percentiles"}},
    }


def rating_bonus(rating: str) -> float:
    return {
        "Outperform": 8.0,
        "Neutral": 0.0,
        "Underperform": -14.0,
    }.get(rating or "", -4.0)


def _theme_signal_value(theme_signal: dict[str, Any], key: str) -> float:
    return _safe_float(theme_signal.get(key)) or 0.0


def risk_penalty(
    *,
    rating: str,
    signal_price: PricePoint,
    factors: dict[str, Any],
    regime: str,
) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    pct = signal_price.pct_chg
    turn = signal_price.turn
    risk_penalty_value = _safe_float(factors.get("risk_penalty")) or 0.0

    if rating == "Underperform":
        penalty += 14.0
        reasons.append("analyst_underperform")
    if pct is not None and pct >= 9.5:
        penalty += 7.0
        reasons.append("limit_up_chase_risk")
    elif pct is not None and pct >= 6.0:
        penalty += 3.0
        reasons.append("high_chase_risk")
    if pct is not None and pct <= -9.5:
        penalty += 8.0
        reasons.append("limit_down_or_sharp_drop")
    elif pct is not None and pct <= -6.0:
        penalty += 4.0
        reasons.append("sharp_drop_risk")
    if turn is not None and turn >= 12:
        penalty += 3.0
        reasons.append("high_turnover_risk")
    if risk_penalty_value > 0:
        penalty += min(risk_penalty_value, 10.0)
        reasons.append("catdesk_risk_penalty")
    if regime == "risk_off" and rating != "Outperform":
        penalty += 2.0
        reasons.append("risk_off_requires_stricter_quality")
    return penalty, reasons


def score_candidate(
    item: dict[str, Any],
    *,
    signal_price: PricePoint,
    market_regime: dict[str, Any],
    qlib: dict[str, Any],
) -> dict[str, Any]:
    raw = item["raw"]
    candidate = raw.get("candidate") or {}
    analyst = raw.get("analyst") or {}
    factors = raw.get("factors") or {}
    stats = raw.get("stats") or {}
    snapshot = raw.get("snapshot") or {}
    theme_signal = item.get("theme_signal") or {}

    code = _candidate_code(raw)
    qlib_name = to_qlib_name(code)
    qlib_score = qlib.get("scores", {}).get(qlib_name)
    qlib_percentile = qlib.get("percentiles", {}).get(qlib_name)
    qlib_adjust = ((qlib_percentile - 0.5) * 6.0) if qlib_percentile is not None else 0.0

    catdesk_score = _safe_float(raw.get("score")) or 0.0
    conviction = _safe_float(analyst.get("conviction")) or 0.0
    hit_rate = _safe_float(stats.get("hit_rate")) or 0.0
    avg_after_signal = _safe_float(stats.get("avg_after_signal")) or 0.0
    momentum = _safe_float(factors.get("momentum")) or 0.0
    liquidity = _safe_float(factors.get("liquidity")) or 0.0
    rating = str(analyst.get("rating") or "")
    theme_signal_score = _theme_signal_value(theme_signal, "score")
    theme_signal_avg_pct = _theme_signal_value(theme_signal, "avg_pct")
    theme_signal_breadth = _theme_signal_value(theme_signal, "breadth")

    penalty, penalty_reasons = risk_penalty(
        rating=rating,
        signal_price=signal_price,
        factors=factors,
        regime=str(market_regime.get("regime") or ""),
    )
    alpha_quality = (
        rating_bonus(rating)
        + ((conviction - 60.0) * 0.35)
        + ((hit_rate - 0.60) * 14.0)
        + min(max(avg_after_signal, -3.0), 3.0) * 1.2
        + momentum * 0.55
        + liquidity * 0.35
    )
    filtered_score = catdesk_score + alpha_quality + qlib_adjust - penalty

    return {
        "code": code,
        "qlib_instrument": qlib_name,
        "name": candidate.get("name") or snapshot.get("name") or "",
        "industry": candidate.get("industry") or "",
        "theme": item["theme"],
        "theme_rank": item["theme_rank"],
        "catdesk_score": round(catdesk_score, 4),
        "rating": rating,
        "conviction": round(conviction, 4) if conviction else None,
        "risk_gate": analyst.get("risk_gate"),
        "hit_rate": round(hit_rate, 6) if hit_rate else None,
        "avg_after_signal": round(avg_after_signal, 6) if avg_after_signal else None,
        "momentum_factor": round(momentum, 4) if momentum else None,
        "liquidity_factor": round(liquidity, 4) if liquidity else None,
        "theme_signal_score": round(theme_signal_score, 4) if theme_signal_score else None,
        "theme_signal_avg_pct": round(theme_signal_avg_pct, 4) if theme_signal_avg_pct else None,
        "theme_signal_breadth": round(theme_signal_breadth, 4) if theme_signal_breadth else None,
        "signal_close": signal_price.close,
        "signal_pct_chg": signal_price.pct_chg,
        "signal_amount": signal_price.amount,
        "signal_turn": signal_price.turn,
        "signal_price_source": signal_price.source,
        "signal_requested_date": signal_price.requested_date,
        "signal_trade_date": signal_price.trade_date,
        "signal_price_fallback_reason": signal_price.fallback_reason,
        "qlib_score": qlib_score,
        "qlib_percentile": round(qlib_percentile, 6) if qlib_percentile is not None else None,
        "qlib_adjust": round(qlib_adjust, 4),
        "risk_penalty": round(penalty, 4),
        "risk_reasons": penalty_reasons,
        "filtered_score": round(filtered_score, 4),
        "catdesk_rationale": candidate.get("rationale") or "",
    }


def _row_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _safe_float(row.get(key))
    return default if value is None else value


def _qlib_percentile(row: dict[str, Any]) -> float:
    value = _safe_float(row.get("qlib_percentile"))
    return 0.5 if value is None else value


def _severe_risk_penalty(row: dict[str, Any]) -> float:
    reasons = set(row.get("risk_reasons") or [])
    penalty = 0.0
    if "analyst_underperform" in reasons:
        penalty += 10.0
    if "limit_down_or_sharp_drop" in reasons:
        penalty += 8.0
    if "sharp_drop_risk" in reasons:
        penalty += 3.0
    if "high_turnover_risk" in reasons:
        penalty += 2.0
    return penalty


def strategy_score(row: dict[str, Any], strategy: str) -> float:
    catdesk = _row_float(row, "catdesk_score")
    conviction = _row_float(row, "conviction", 60.0)
    hit_rate = _row_float(row, "hit_rate", 0.5)
    avg_after_signal = _row_float(row, "avg_after_signal")
    momentum = _row_float(row, "momentum_factor")
    liquidity = _row_float(row, "liquidity_factor")
    theme_score = _row_float(row, "theme_signal_score", 50.0)
    theme_avg_pct = _row_float(row, "theme_signal_avg_pct")
    theme_breadth = _row_float(row, "theme_signal_breadth")
    qlib_pct = _qlib_percentile(row)
    risk = _row_float(row, "risk_penalty")
    rating = str(row.get("rating") or "")

    if strategy == "catdesk_raw":
        return catdesk
    if strategy == "dsa_conservative":
        return _row_float(row, "filtered_score")
    if strategy == "factor_balance":
        return (
            catdesk * 0.55
            + (conviction - 55.0) * 0.45
            + (hit_rate - 0.50) * 26.0
            + avg_after_signal * 1.8
            + momentum * 1.25
            + liquidity * 0.9
            + rating_bonus(rating) * 0.65
            - risk * 0.55
        )
    if strategy == "qlib_priority":
        return (
            qlib_pct * 70.0
            + catdesk * 0.28
            + momentum * 0.8
            + liquidity * 0.5
            + max(avg_after_signal, -2.0) * 0.8
            - _severe_risk_penalty(row)
        )
    if strategy == "theme_momentum":
        return (
            catdesk * 0.72
            + theme_score * 0.38
            + theme_avg_pct * 5.0
            + theme_breadth * 8.0
            + momentum * 1.6
            + liquidity * 0.8
            + max(avg_after_signal, -1.5) * 0.8
            + rating_bonus(rating) * 0.35
            - _severe_risk_penalty(row)
        )
    if strategy == "theme_momentum_qlib":
        qlib_bonus = max(qlib_pct - 0.5, 0.0) * 18.0 - max(0.5 - qlib_pct, 0.0) * 4.0
        return strategy_score(row, "theme_momentum") + qlib_bonus
    if strategy == "dominant_theme_bucket":
        return (
            theme_score * 1.15
            + catdesk * 0.65
            + theme_avg_pct * 6.0
            + momentum * 1.4
            + liquidity * 0.7
            + max(avg_after_signal, -1.5) * 0.8
            + max(qlib_pct - 0.5, 0.0) * 8.0
            - _severe_risk_penalty(row)
        )
    raise ValueError(f"Unknown strategy: {strategy}")


def attach_strategy_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["strategy_scores"] = {
            strategy: round(strategy_score(item, strategy), 6)
            for strategy in STRATEGY_LABELS
        }
        enriched.append(item)
    return enriched


def rank_by_strategy(rows: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (row.get("strategy_scores") or {}).get(strategy, float("-inf")),
        reverse=True,
    )


def select_dominant_theme_bucket(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    theme_scores: dict[str, float] = {}
    for row in rows:
        theme = str(row.get("theme") or "")
        theme_scores[theme] = max(theme_scores.get(theme, float("-inf")), _row_float(row, "theme_signal_score", 0.0))
    dominant_theme = max(theme_scores, key=theme_scores.get)
    main_quota = min(
        sum(1 for row in rows if row.get("theme") == dominant_theme),
        max(1, math.ceil(top_n * 0.625)),
    )
    dominant_rows = [
        row for row in rank_by_strategy(rows, "dominant_theme_bucket")
        if row.get("theme") == dominant_theme
    ][:main_quota]
    selected_codes = {row["code"] for row in dominant_rows}
    remaining = [
        row for row in rank_by_strategy(rows, "theme_momentum_qlib")
        if row["code"] not in selected_codes
    ]
    return (dominant_rows + remaining)[:top_n]


def strategy_comparison(rows: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    actual_winners = sorted(
        rows,
        key=lambda row: row.get("validation_return_pct") if row.get("validation_return_pct") is not None else -999.0,
        reverse=True,
    )
    winner_codes = {row["code"] for row in actual_winners[: max(top_n, 1)]}
    for strategy in STRATEGY_LABELS:
        ranked = rank_by_strategy(rows, strategy)
        selected = (
            select_dominant_theme_bucket(rows, top_n)
            if strategy == "dominant_theme_bucket"
            else ranked[:top_n]
        )
        selected_codes = {row["code"] for row in selected}
        comparison[strategy] = {
            "strategy": strategy,
            "label": STRATEGY_LABELS[strategy],
            "description": STRATEGY_DESCRIPTIONS[strategy],
            "summary": summarize(selected),
            "selected": selected,
            "missed_top_actual_winners": [
                row for row in actual_winners[:top_n] if row["code"] not in selected_codes
            ],
            "captured_top_actual_winner_count": len(selected_codes & winner_codes),
        }
    return comparison


def fetch_validation_snapshots(codes: list[str], *, timeout: float, use_system_proxy: bool) -> dict[str, dict[str, Any]]:
    rows = fetch_sina_a_snapshots(codes, timeout=timeout, use_system_proxy=use_system_proxy)
    return {str(row.get("symbol") or "").zfill(6): row for row in rows if row.get("symbol")}


def apply_validation(
    rows: list[dict[str, Any]],
    *,
    validation_date: str,
    validation_source_dir: Path,
    sina_snapshots: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for row in rows:
        code = row["code"]
        local = read_price_point(validation_source_dir, code, validation_date)
        snapshot = sina_snapshots.get(code) or {}
        close = local.close
        pct = local.pct_chg
        source = local.source
        trade_date = validation_date if local.source else ""
        trade_time = ""

        if snapshot and snapshot.get("trade_date") == validation_date:
            close = _safe_float(snapshot.get("close"))
            pct = _safe_float(snapshot.get("pct"))
            source = snapshot.get("source") or "sina_snapshot"
            trade_date = str(snapshot.get("trade_date") or "")
            trade_time = str(snapshot.get("trade_time") or "")

        signal_close = _safe_float(row.get("signal_close"))
        close_to_close_return = None
        if close is not None and signal_close:
            close_to_close_return = ((close / signal_close) - 1.0) * 100.0

        enriched = dict(row)
        enriched.update(
            {
                "validation_close": close,
                "validation_pct": pct,
                "validation_return_pct": round(close_to_close_return, 6) if close_to_close_return is not None else None,
                "validation_source": source,
                "validation_trade_date": trade_date,
                "validation_trade_time": trade_time,
                "hit": close_to_close_return is not None and close_to_close_return > 0,
                "strong_hit": close_to_close_return is not None and close_to_close_return >= 5.0,
                "leader_hit": close_to_close_return is not None and close_to_close_return >= 8.0,
            }
        )
        validated.append(enriched)
    return validated


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row.get("validation_return_pct") is not None]
    returns = [float(row["validation_return_pct"]) for row in evaluable]
    if not evaluable:
        return {
            "count": len(rows),
            "evaluable": 0,
            "hit_rate": None,
            "strong_hit_rate": None,
            "leader_hit_rate": None,
            "avg_return_pct": None,
            "median_return_pct": None,
        }
    series = pd.Series(returns)
    return {
        "count": len(rows),
        "evaluable": len(evaluable),
        "hit_count": sum(1 for row in evaluable if row.get("hit")),
        "strong_hit_count": sum(1 for row in evaluable if row.get("strong_hit")),
        "leader_hit_count": sum(1 for row in evaluable if row.get("leader_hit")),
        "hit_rate": round(sum(1 for row in evaluable if row.get("hit")) / len(evaluable), 6),
        "strong_hit_rate": round(sum(1 for row in evaluable if row.get("strong_hit")) / len(evaluable), 6),
        "leader_hit_rate": round(sum(1 for row in evaluable if row.get("leader_hit")) / len(evaluable), 6),
        "avg_return_pct": round(float(series.mean()), 6),
        "median_return_pct": round(float(series.median()), 6),
        "min_return_pct": round(float(series.min()), 6),
        "max_return_pct": round(float(series.max()), 6),
    }


def markdown_table(rows: list[dict[str, Any]], *, limit: int | None = None) -> str:
    display = rows[:limit] if limit else rows
    lines = [
        "| Rank | Code | Name | Theme | Score | Signal pct | Validation return | Hit | Source |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for index, row in enumerate(display, start=1):
        score = row.get("filtered_score")
        signal_pct = row.get("signal_pct_chg")
        ret = row.get("validation_return_pct")
        lines.append(
            "| {rank} | `{code}` | {name} | {theme} | {score} | {signal_pct} | {ret} | {hit} | {source} |".format(
                rank=index,
                code=row.get("code"),
                name=row.get("name") or "",
                theme=str(row.get("theme") or "").replace("|", "/"),
                score=f"{float(score):.2f}" if score is not None else "n/a",
                signal_pct=f"{float(signal_pct):.2f}%" if signal_pct is not None else "n/a",
                ret=f"{float(ret):.2f}%" if ret is not None else "n/a",
                hit="yes" if row.get("hit") else "no",
                source=row.get("validation_source") or "",
            )
        )
    return "\n".join(lines)


def strategy_summary_table(strategies: dict[str, Any]) -> str:
    lines = [
        "| Strategy | Count | Hit rate | Avg return | Strong >=5% | Leader >=8% | Captured top winners |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, payload in strategies.items():
        summary = payload.get("summary") or {}
        lines.append(
            "| {label} | {count} | {hit} | {avg}% | {strong} | {leader} | {captured} |".format(
                label=payload.get("label") or key,
                count=summary.get("count"),
                hit=summary.get("hit_rate"),
                avg=summary.get("avg_return_pct"),
                strong=summary.get("strong_hit_rate"),
                leader=summary.get("leader_hit_rate"),
                captured=payload.get("captured_top_actual_winner_count"),
            )
        )
    return "\n".join(lines)


def strategy_sections(strategies: dict[str, Any]) -> str:
    sections: list[str] = []
    for key, payload in strategies.items():
        missed = payload.get("missed_top_actual_winners") or []
        missed_text = ", ".join(f"`{row['code']}` {row.get('name')}" for row in missed) or "none"
        sections.append(
            f"""### {payload.get("label") or key}

{payload.get("description") or ""}

{markdown_table(payload.get("selected") or [])}

Missed top actual winners in this candidate pool: {missed_text}
"""
        )
    return "\n\n".join(sections)


def build_markdown(payload: dict[str, Any]) -> str:
    selected = payload["selected"]
    all_rows = payload["all_candidates"]
    selected_summary = payload["selected_summary"]
    raw_summary = payload["raw_summary"]
    strategies = payload.get("strategies") or {}
    top_winners = sorted(all_rows, key=lambda row: row.get("validation_return_pct") or -999, reverse=True)[:10]
    top_losers = sorted(all_rows, key=lambda row: row.get("validation_return_pct") if row.get("validation_return_pct") is not None else 999)[:10]
    qlib_latest_date = payload["qlib"].get("latest_date") or "n/a"
    signal_date = payload.get("signal_date")
    validation_date = payload.get("validation_date")
    selected_label = payload.get("selected_strategy_label") or "Filtered watchlist"
    qlib_role = (
        "Qlib contributes a small same-date model-score adjustment because the latest available Qlib score date matches the signal date."
        if qlib_latest_date == signal_date
        else "Qlib contributes only a small stale-score adjustment because the latest available Qlib score date does not match the signal date."
    )
    if payload.get("input_quality", {}).get("qlib_backfilled_allowed"):
        qlib_role += " This report explicitly allowed backfilled Qlib scores for research comparison; it is not proof that the Qlib score existed before the validation date."

    return f"""# {payload.get("title") or "CatDesk + DSA/Qlib Signal Validation"}

Generated: {payload["generated_at"]}

## What This Did

This report uses the CatDesk report from `{payload["inputs"]["catdesk_report"]}` as the {signal_date} candidate source.

Selection used:

- CatDesk {signal_date} candidate score, rating, conviction, history, momentum, liquidity, and risk fields.
- Local daily bars on or before {signal_date} from `{payload["inputs"]["signal_price_dir"]}`.
- DSA market regime from `{payload["inputs"]["market_regime_report"]}`.
- Qlib score artifact `{payload["qlib"]["prediction_path"] or "n/a"}`. Latest Qlib score date: `{payload["qlib"]["latest_date"] or "n/a"}`.

Validation used:

- {validation_date} close/snapshot from Sina quote snapshots when available.
- Local {validation_date} parquet only as a fallback.

It did **not** use {validation_date} data when ranking {signal_date} candidates. It does not write to `quant_data`, does not retrain Qlib, does not connect to broker software, and does not create buy/sell orders.

## Market Risk Context

- DSA regime: `{payload["market_regime"].get("regime")}`
- Suggested position ratio from DSA regime report: `{payload["market_regime"].get("position_ratio")}`
- Cash ratio: `{payload["market_regime"].get("cash_ratio")}`

Because the regime was risk-off, this report treats the final list as a small watchlist rather than a full-position portfolio.

## Result Summary

Raw CatDesk candidate pool:

- Count: {raw_summary.get("count")}
- Evaluable: {raw_summary.get("evaluable")}
- Hit rate: {raw_summary.get("hit_rate")}
- Strong hit rate, return >= 5%: {raw_summary.get("strong_hit_rate")}
- Leader hit rate, return >= 8%: {raw_summary.get("leader_hit_rate")}
- Average return: {raw_summary.get("avg_return_pct")}%
- Median return: {raw_summary.get("median_return_pct")}%

Filtered watchlist ({selected_label}):

- Count: {selected_summary.get("count")}
- Evaluable: {selected_summary.get("evaluable")}
- Hit rate: {selected_summary.get("hit_rate")}
- Strong hit rate, return >= 5%: {selected_summary.get("strong_hit_rate")}
- Leader hit rate, return >= 8%: {selected_summary.get("leader_hit_rate")}
- Average return: {selected_summary.get("avg_return_pct")}%
- Median return: {selected_summary.get("median_return_pct")}%

## Strategy Cross-Comparison

{strategy_summary_table(strategies)}

## Filtered Watchlist ({selected_label})

{markdown_table(selected)}

## Strategy Details

{strategy_sections(strategies)}

## Top 10 Actual Winners In Candidate Pool

{markdown_table(top_winners)}

## Top 10 Actual Losers In Candidate Pool

{markdown_table(top_losers)}

## Scoring Rule

CatDesk provides the initial theme/candidate score. The DSA factor-balance layer then adjusts it with rating, conviction, historical hit rate, average post-signal return, momentum, liquidity, and risk penalties. This script does not call an external stock-screening adapter. {qlib_role} Latest Qlib score date: `{qlib_latest_date}`.

Risk penalties include: underperform rating, {signal_date} chase risk, sharp-drop risk, high turnover, CatDesk risk penalty, and DSA risk-off strictness.

## Data Notes

- CatDesk report status: `{payload.get("input_quality", {}).get("catdesk_report_status", "unknown")}`. {payload.get("input_quality", {}).get("catdesk_report_reason", "")}
- Market regime status: `{payload.get("input_quality", {}).get("market_regime_status", "unknown")}`. {payload.get("input_quality", {}).get("market_regime_reason", "")}
- Qlib backfilled-score mode: `{payload.get("input_quality", {}).get("qlib_backfilled_allowed", False)}`.
- A-share trading calendar: `{payload.get("input_quality", {}).get("trading_calendar_source", "unknown")}`. Calendar fallback: `{payload.get("input_quality", {}).get("calendar_fallback", "") or "none"}`.
- Return basis: close-to-close research check only. This is not an executable entry-price backtest.
- `signal_price_fallback_reason=requested_date_missing_use_previous_trade_date` means the requested signal date had no A-share bar, so the script used the nearest previous local trading day close and recorded that actual `signal_trade_date`.
- `validation_source=sina_snapshot` means the {validation_date} result came from Sina quote snapshot and was not written back into the local database.
- `validation_return_pct` is close-to-close: {validation_date} close divided by the recorded signal close minus 1.
- This is a one-day retrospective validation, not a complete strategy backtest.
"""


def generate_report(
    *,
    catdesk_report: Path,
    signal_date: str,
    validation_date: str | None,
    candidate_set: str,
    top_n: int,
    selected_strategy: str,
    use_online_validation: bool,
    use_system_proxy: bool,
    allow_backfilled_qlib: bool,
    timeout: float,
    out_dir: Path,
) -> dict[str, Any]:
    validation_date, calendar_source = resolve_validation_date(signal_date, validation_date)
    catdesk_payload, candidates = load_catdesk_candidates(catdesk_report, candidate_set=candidate_set)
    catdesk_quality = classify_catdesk_report(catdesk_payload, signal_date)
    data_root = quant_data_root()
    signal_dir = data_root / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
    validation_dir = data_root / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
    market_regime_path, market_regime_quality = find_market_regime_path(reports_root(), signal_date)
    market_regime = load_market_regime(market_regime_path)
    market_regime.update(market_regime_quality)
    qlib_cutoff = "" if allow_backfilled_qlib else validation_cutoff_iso(validation_date)
    qlib = load_latest_qlib_scores(signal_date, top_reference_date=signal_date, run_cutoff=qlib_cutoff)

    scored: list[dict[str, Any]] = []
    for item in candidates:
        code = _candidate_code(item["raw"])
        signal_price = read_price_point_on_or_before(signal_dir, code, signal_date)
        scored.append(
            score_candidate(
                item,
                signal_price=signal_price,
                market_regime=market_regime,
                qlib=qlib,
            )
        )

    scored.sort(key=lambda row: row["filtered_score"], reverse=True)
    codes = [row["code"] for row in scored]
    sina_snapshots = (
        fetch_validation_snapshots(codes, timeout=timeout, use_system_proxy=use_system_proxy)
        if use_online_validation
        else {}
    )
    validated = apply_validation(
        scored,
        validation_date=validation_date,
        validation_source_dir=validation_dir,
        sina_snapshots=sina_snapshots,
    )
    validated = attach_strategy_scores(validated)
    strategies = strategy_comparison(validated, top_n=top_n)
    selected_payload = strategies.get(selected_strategy) or strategies[DEFAULT_SELECTED_STRATEGY]
    selected = selected_payload["selected"]
    validated = rank_by_strategy(validated, selected_strategy)

    payload: dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "title": "CatDesk + DSA/Qlib Signal Validation",
        "mode": "catdesk_dsa_qlib_one_day_validation",
        "signal_date": signal_date,
        "validation_date": validation_date,
        "candidate_set": candidate_set,
        "top_n": top_n,
        "selected_strategy": selected_strategy,
        "selected_strategy_label": STRATEGY_LABELS.get(selected_strategy, selected_strategy),
        "inputs": {
            "catdesk_report": str(catdesk_report),
            "catdesk_date": catdesk_payload.get("date"),
            "catdesk_generated_at": catdesk_payload.get("generated_at"),
            "signal_price_dir": str(signal_dir),
            "validation_price_dir": str(validation_dir),
            "market_regime_report": str(market_regime_path),
            "online_validation": use_online_validation,
            "online_validation_source": "sina_snapshot" if use_online_validation else "",
            "qlib_backfilled_allowed": allow_backfilled_qlib,
        },
        "input_quality": {
            **catdesk_quality,
            **market_regime_quality,
            "qlib_run_cutoff": qlib_cutoff,
            "qlib_backfilled_allowed": allow_backfilled_qlib,
            "qlib_status": "available" if qlib.get("available") else "unavailable",
            "qlib_reason": qlib.get("reason") or "",
            "trading_calendar_source": calendar_source,
            "calendar_fallback": WEEKDAY_FALLBACK_SOURCE if calendar_source == WEEKDAY_FALLBACK_SOURCE else "",
            "return_basis": "close_to_close_research_only",
            "return_basis_reason": "Validation uses close-to-close movement only and does not prove executable entry fills.",
        },
        "market_regime": market_regime,
        "qlib": {
            key: value
            for key, value in qlib.items()
            if key not in {"scores", "percentiles"}
        },
        "raw_summary": summarize(validated),
        "selected_summary": summarize(selected),
        "strategies": strategies,
        "selected": selected,
        "all_candidates": validated,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"catdesk_dsa_validation_{signal_date.replace('-', '')}_to_{validation_date.replace('-', '')}_{ts}.json"
    md_path = out_dir / f"catdesk_dsa_validation_{signal_date.replace('-', '')}_to_{validation_date.replace('-', '')}_{ts}.md"
    latest_json = out_dir / f"catdesk_dsa_validation_{signal_date.replace('-', '')}_to_{validation_date.replace('-', '')}_latest.json"
    latest_md = out_dir / f"catdesk_dsa_validation_{signal_date.replace('-', '')}_to_{validation_date.replace('-', '')}_latest.md"
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }

    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    markdown = build_markdown(payload)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def build_rolling_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CatDesk Rolling Signal Validation",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "This rolling report walks backward one business-day pair at a time. Each step needs that signal day's CatDesk morning JSON as the signal source. Backfilled reports are labeled as samples instead of being treated as real same-morning signals.",
        "",
        "| Signal -> Validation | Status | Selected strategy | Count | Hit rate | Avg return | Report |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in payload.get("items") or []:
        if item.get("status") in {"ok", "backfilled_sample", "unknown_generated_at", "date_mismatch"}:
            summary = item.get("selected_summary") or {}
            link = item.get("latest_markdown") or ""
            lines.append(
                "| {signal} -> {validation} | {status} | {strategy} | {count} | {hit} | {avg}% | `{link}` |".format(
                    signal=item.get("signal_date"),
                    validation=item.get("validation_date"),
                    status=item.get("status"),
                    strategy=item.get("selected_strategy_label"),
                    count=summary.get("count"),
                    hit=summary.get("hit_rate"),
                    avg=summary.get("avg_return_pct"),
                    link=link,
                )
            )
        else:
            lines.append(
                "| {signal} -> {validation} | {status} | {strategy} | 0 | n/a | n/a | {reason} |".format(
                    signal=item.get("signal_date"),
                    validation=item.get("validation_date"),
                    status=item.get("status") or "missing_signal_report",
                    strategy=item.get("selected_strategy_label") or "",
                    reason=item.get("reason") or "",
                )
            )
    return "\n".join(lines) + "\n"


def generate_rolling_report(
    *,
    start_signal_date: str,
    days: int,
    reports_web_dir: Path,
    candidate_set: str,
    top_n: int,
    selected_strategy: str,
    use_online_validation: bool,
    use_system_proxy: bool,
    allow_backfilled_qlib: bool,
    timeout: float,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, Any]] = []
    for signal_date, validation_date in business_day_pairs(start_signal_date, days):
        report_path = reports_web_dir / f"{signal_date}-morning.json"
        if not report_path.exists():
            items.append(
                {
                    "status": "missing_signal_report",
                    "signal_date": signal_date,
                    "validation_date": validation_date,
                    "selected_strategy": selected_strategy,
                    "selected_strategy_label": STRATEGY_LABELS.get(selected_strategy, selected_strategy),
                    "reason": f"Missing CatDesk report: {report_path}",
                }
            )
            continue
        report = generate_report(
            catdesk_report=report_path,
            signal_date=signal_date,
            validation_date=validation_date,
            candidate_set=candidate_set,
            top_n=top_n,
            selected_strategy=selected_strategy,
            use_online_validation=use_online_validation,
            use_system_proxy=use_system_proxy,
            allow_backfilled_qlib=allow_backfilled_qlib,
            timeout=timeout,
            out_dir=out_dir,
        )
        catdesk_status = (report.get("input_quality") or {}).get("catdesk_report_status") or "unknown_generated_at"
        item_status = "ok" if catdesk_status == "as_of_morning" else catdesk_status
        items.append(
            {
                "status": item_status,
                "signal_date": signal_date,
                "validation_date": validation_date,
                "selected_strategy": selected_strategy,
                "selected_strategy_label": report.get("selected_strategy_label"),
                "selected_summary": report.get("selected_summary"),
                "raw_summary": report.get("raw_summary"),
                "input_quality": report.get("input_quality"),
                "latest_json": report.get("artifacts", {}).get("latest_json"),
                "latest_markdown": report.get("artifacts", {}).get("latest_markdown"),
            }
        )

    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "mode": "catdesk_rolling_signal_validation",
        "start_signal_date": start_signal_date,
        "days": days,
        "candidate_set": candidate_set,
        "top_n": top_n,
        "selected_strategy": selected_strategy,
        "selected_strategy_label": STRATEGY_LABELS.get(selected_strategy, selected_strategy),
        "qlib_backfilled_allowed": allow_backfilled_qlib,
        "items": items,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"catdesk_rolling_validation_{start_signal_date.replace('-', '')}_{days}d_{ts}.json"
    md_path = out_dir / f"catdesk_rolling_validation_{start_signal_date.replace('-', '')}_{days}d_{ts}.md"
    latest_json = out_dir / f"catdesk_rolling_validation_{start_signal_date.replace('-', '')}_{days}d_latest.json"
    latest_md = out_dir / f"catdesk_rolling_validation_{start_signal_date.replace('-', '')}_{days}d_latest.md"
    payload["artifacts"] = {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    markdown = build_rolling_markdown(payload)
    json_path.write_text(serialized, encoding="utf-8")
    latest_json.write_text(serialized, encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    latest_md.write_text(markdown, encoding="utf-8")
    return payload


def main() -> int:
    default_report = projects_root() / "a-share-us-catalyst" / "reports-web" / f"{DEFAULT_SIGNAL_DATE}-morning.json"
    default_reports_web_dir = projects_root() / "a-share-us-catalyst" / "reports-web"
    parser = argparse.ArgumentParser(description="Validate CatDesk candidates with local DSA/Qlib filters.")
    parser.add_argument("--catdesk-report", type=Path, default=default_report)
    parser.add_argument("--signal-date", default=DEFAULT_SIGNAL_DATE)
    parser.add_argument("--validation-date", default=None)
    parser.add_argument("--candidate-set", choices=("top", "all"), default="top")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--selected-strategy", choices=tuple(STRATEGY_LABELS), default=DEFAULT_SELECTED_STRATEGY)
    parser.add_argument(
        "--rolling-days",
        type=int,
        default=0,
        help="Walk backward from --signal-date for N business-day signal/validation pairs.",
    )
    parser.add_argument("--reports-web-dir", type=Path, default=default_reports_web_dir)
    parser.add_argument("--out-dir", type=Path, default=reports_root())
    parser.add_argument("--no-online-validation", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    parser.add_argument(
        "--allow-backfilled-qlib",
        action="store_true",
        help="Use Qlib scores generated after the validation cutoff, marking the report as backfilled research.",
    )
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    try:
        resolved_validation_date, _calendar_source = resolve_validation_date(args.signal_date, args.validation_date)
    except ValueError as exc:
        parser.error(str(exc))

    if args.rolling_days:
        payload = generate_rolling_report(
            start_signal_date=args.signal_date,
            days=args.rolling_days,
            reports_web_dir=args.reports_web_dir,
            candidate_set=args.candidate_set,
            top_n=max(args.top_n, 1),
            selected_strategy=args.selected_strategy,
            use_online_validation=not args.no_online_validation,
            use_system_proxy=args.use_system_proxy,
            allow_backfilled_qlib=args.allow_backfilled_qlib,
            timeout=args.timeout,
            out_dir=args.out_dir,
        )
        trusted_count = sum(1 for item in payload["items"] if item.get("status") == "ok")
        backfilled_count = sum(1 for item in payload["items"] if item.get("status") == "backfilled_sample")
        missing_count = sum(1 for item in payload["items"] if item.get("status") == "missing_signal_report")
        other_count = max(len(payload["items"]) - trusted_count - backfilled_count - missing_count, 0)
        print(
            "Rolling validation complete. "
            f"trusted={trusted_count}, backfilled={backfilled_count}, missing={missing_count}, other={other_count}."
        )
        print(f"Latest markdown: {payload['artifacts']['latest_markdown']}")
        print(f"Latest JSON: {payload['artifacts']['latest_json']}")
        return 0

    payload = generate_report(
        catdesk_report=args.catdesk_report,
        signal_date=args.signal_date,
        validation_date=resolved_validation_date,
        candidate_set=args.candidate_set,
        top_n=max(args.top_n, 1),
        selected_strategy=args.selected_strategy,
        use_online_validation=not args.no_online_validation,
        use_system_proxy=args.use_system_proxy,
        allow_backfilled_qlib=args.allow_backfilled_qlib,
        timeout=args.timeout,
        out_dir=args.out_dir,
    )
    print(f"Selected {len(payload['selected'])} of {len(payload['all_candidates'])} candidates.")
    print(f"Selected strategy: {payload['selected_strategy_label']} ({payload['selected_strategy']})")
    print(f"Selected hit rate: {payload['selected_summary'].get('hit_rate')}")
    print(f"Raw hit rate: {payload['raw_summary'].get('hit_rate')}")
    print(f"Latest markdown: {payload['artifacts']['latest_markdown']}")
    print(f"Latest JSON: {payload['artifacts']['latest_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
