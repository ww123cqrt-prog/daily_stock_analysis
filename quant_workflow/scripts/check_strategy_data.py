"""Read-only, seeded strategy probes over two frozen local market snapshots.

Differences are investigation leads, not proof that either source is right.
This module never downloads, repairs, publishes, or gates ordinary reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parents[1]
for directory in (PROJECT, SCRIPT_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from _paths import reports_root  # noqa: E402
from check_etf_data import _ReadSet, _write_csv  # noqa: E402
from quant_qmt_data import bar_snapshot, read_bars  # noqa: E402
from quant_qmt_data.miniqmt_etf_source import frozen_exchange_calendar  # noqa: E402
from quant_qmt_data.quant_data_contract import DataContractError, sha256_file  # noqa: E402
from src.core.strategy_backtest import (  # noqa: E402
    PriceBar, StrategyBacktestConfig, TargetAllocation, run_portfolio_backtest,
)


VERSION = "strategy-data-check.v1"
RAW_FIELDS = ("open", "high", "low", "close", "volume", "amount", "preclose", "pctChg")
PRICE_FIELDS = ("open", "high", "low", "close")
FINDING_COLUMNS = ("batch", "symbol", "date", "side", "basis", "field", "rule",
                   "classification", "observed", "reference", "detail")
UNITS = {**dict.fromkeys((*PRICE_FIELDS, "preclose"), "CNY"), "volume": "shares",
         "amount": "CNY", "pctChg": "percent"}


def _bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _digest(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _save(path, value):
    path.write_bytes(_bytes(value))


def _value(value):
    return float(value) if pd.notna(value) and np.isfinite(float(value)) else None


def _same(left, right):
    if pd.isna(left) or pd.isna(right):
        return bool(pd.isna(left) and pd.isna(right))
    return bool(np.isclose(float(left), float(right), rtol=1e-9, atol=1e-6))


def _freeze(root, market, expected=None):
    before = sha256_file(root / "DATA_MANIFEST.json")
    snapshots = {}
    cache = {}
    for basis in ("raw", "qfq"):
        dataset = "cn_etf_daily" if market == "etf" else "ashare_daily_" + basis
        snapshots[basis] = bar_snapshot(
            dataset, quant_data_root=root, price_basis=basis, validated_cache=cache,
            expected_identity=expected[basis] if expected is not None else None)
    if sha256_file(root / "DATA_MANIFEST.json") != before:
        raise ValueError("catalog changed while freezing inputs; retry this check")
    return {"root": root, "snapshots": snapshots, "cache": cache}


def _identities(side):
    return {basis: snapshot["identity"] for basis, snapshot in side["snapshots"].items()}


def _date(text):
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError("dates must be timezone-free ISO dates")
    day = pd.Timestamp(text)
    if pd.isna(day) or day.tzinfo is not None or day != day.normalize():
        raise ValueError("dates must be timezone-free ISO dates")
    return day.date().isoformat()


def _plan(primary, reference, *, market, start, end, batches, sample_size, sessions, seed, codes,
          calendar=None, calendar_source=None):
    if not 1 <= batches <= 10 or not 2 <= sample_size <= 64 or not 2 <= sessions <= 252:
        raise ValueError("use 1..10 batches, 2..64 symbols and 2..252 test sessions")
    pool = sorted(set(primary["snapshots"]["qfq"]["file_index"].symbol.astype(str)))
    if codes:
        pool = sorted(set(codes))
        if not all(re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", symbol) for symbol in pool):
            raise ValueError("codes must be canonical SH/SZ/BJ symbols")
    if len(pool) < batches * sample_size:
        raise ValueError("not enough distinct symbols; lower batches/sample-size explicitly")
    if calendar is None:
        calendar = frozen_exchange_calendar(primary["snapshots"]["qfq"]["snapshot_path"], start, end)
    if len(calendar) < sessions + 20:
        raise ValueError("requested interval needs test sessions plus 20 warm-up sessions")
    rng = random.Random(seed)
    selected = rng.sample(pool, batches * sample_size)
    records = []
    for i in range(batches):
        lookback, rebalance = rng.choice((5, 10, 20)), rng.choice((1, 5, 10))
        last = rng.randrange(sessions + lookback - 1, len(calendar))
        days = calendar[last - sessions - lookback + 1:last + 1]
        records.append({"id": f"batch_{i + 1}",
                        "requested_symbols": sorted(selected[i * sample_size:(i + 1) * sample_size]),
                        "calendar": [day.date().isoformat() for day in days],
                        "test_start": days[lookback].date().isoformat(),
                        "test_end": days[-1].date().isoformat(), "lookback": lookback,
                        "rebalance_sessions": rebalance, "topk": rng.choice((1, max(1, sample_size // 3)))})
    return {"version": VERSION, "market": market, "seed": seed,
            "sampling": "uniform_without_replacement_from_primary_frozen_index_or_explicit_codes",
            "requested_range": [start, end], "parameters_chosen_before_results": True,
            "calendar_source": calendar_source or {"kind": "snapshot_exchange_evidence"},
            "inputs": {"primary": _identities(primary), "reference": _identities(reference)},
            "execution": "primary_raw_next_common_session_open", "algorithm": "close_momentum",
            "backtest": {"initial_capital": 1_000_000., "side_cost_pct": 0., "lot_size": 100},
            "batches": records}


def _validate_plan(plan, market):
    if (plan.get("version") != VERSION or plan.get("market") != market
            or plan.get("algorithm") != "close_momentum"
            or plan.get("execution") != "primary_raw_next_common_session_open"
            or plan.get("backtest") != {"initial_capital": 1_000_000., "side_cost_pct": 0., "lot_size": 100}):
        raise ValueError("unsupported strategy check plan")
    batches = plan.get("batches", [])
    if not 1 <= len(batches) <= 10 or len({b["id"] for b in batches}) != len(batches):
        raise ValueError("invalid batch identities")
    selected = []
    for batch in batches:
        symbols, calendar = batch["requested_symbols"], batch["calendar"]
        dates = pd.DatetimeIndex([_date(day) for day in calendar])
        lookback = batch["lookback"]
        if (not 2 <= len(symbols) <= 64 or not all(re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", s) for s in symbols)
                or not 1 <= lookback <= 20 or not lookback + 2 <= len(dates) <= lookback + 252
                or dates.has_duplicates or not dates.is_monotonic_increasing
                or batch["test_start"] != calendar[lookback] or batch["test_end"] != calendar[-1]
                or not 1 <= batch["topk"] <= len(symbols) or not 1 <= batch["rebalance_sessions"] <= 10):
            raise ValueError("invalid sampling window or parameters")
        selected.extend(symbols)
    if len(set(selected)) != len(selected):
        raise ValueError("sampling plan repeats symbols")


def _read(side, basis, symbol, start, end, files):
    snapshot = side["snapshots"][basis]
    result = read_bars(snapshot["dataset_id"], symbol, quant_data_root=side["root"], price_basis=basis,
                       start=start, end=end, expected_identity=snapshot["identity"],
                       validated_cache=side["cache"], volume_unit="base")
    row = snapshot["file_index"].loc[snapshot["file_index"].symbol == symbol].iloc[0]
    files[str(snapshot["snapshot_path"] / row.relative_path)] = str(row.sha256)
    return result.assign(date=pd.to_datetime(result.date)).set_index("date").sort_index()


def _finding(findings, batch, symbol, day, side, basis, field, rule, observed=None, reference=None,
             classification="needs_review", detail=""):
    findings.append(dict(batch=batch, symbol=symbol, date=str(pd.Timestamp(day).date()) if day else "",
                         side=side, basis=basis, field=field, rule=rule, classification=classification,
                         observed=observed, reference=reference, detail=detail))


def _compare_frames(left, right, findings, batch, symbol, *, side, basis, paired=False):
    for day in left.index.symmetric_difference(right.index):
        _finding(findings, batch, symbol, day, side, basis, "date", "missing_observation",
                 classification="insufficient_evidence", detail="An absent bar is not proof of suspension.")
    fields = ("volume", "amount") if paired else RAW_FIELDS if basis == "raw" else PRICE_FIELDS
    common = left.index.intersection(right.index)
    for field in fields:
        if field in {"preclose", "pctChg"} and field not in left and field not in right:
            continue
        if field not in left or field not in right:
            _finding(findings, batch, symbol, None, side, basis, field, "missing_field",
                     classification="insufficient_evidence")
            continue
        a, b = pd.to_numeric(left.loc[common, field], errors="coerce"), pd.to_numeric(right.loc[common, field], errors="coerce")
        a_values = a.to_numpy(dtype=float, na_value=np.nan)
        b_values = b.to_numpy(dtype=float, na_value=np.nan)
        missing = ~np.isfinite(a_values) | ~np.isfinite(b_values)
        for day in common[missing]:
            _finding(findings, batch, symbol, day, side, basis, field, "missing_numeric_evidence",
                     classification="insufficient_evidence")
        close = np.isclose(a_values, b_values, rtol=1e-9, atol=1e-6, equal_nan=True) | missing
        for day in common[~close]:
            rule = "paired_quantity_difference" if paired else (
                "cross_source_difference" if basis == "raw" else "qfq_convention_or_value_difference")
            _finding(findings, batch, symbol, day, side, basis, field, rule,
                     _value(a.loc[day]), _value(b.loc[day]),
                     classification="confirmed_error" if paired else "needs_review")


def _raw_leads(frame, findings, batch, symbol, side):
    if not {"amount", "volume", "low", "high"}.issubset(frame):
        return
    numbers = frame[["amount", "volume", "low", "high"]].apply(pd.to_numeric, errors="coerce")
    mean = numbers.amount / numbers.volume.where(numbers.volume > 0)
    suspect = (mean < numbers.low * .9999) | (mean > numbers.high * 1.0001)
    for day in frame.index[suspect]:
        _finding(findings, batch, symbol, day, side, "raw", "volume/amount", "average_price_outside_range",
                 _value(mean.loc[day]), detail="Check units and block trades. This bound alone cannot prove an error.")


def _probe(frames, execution, batch):
    dates = pd.DatetimeIndex(batch["calendar"])
    test = dates[dates >= batch["test_start"]]
    scores = pd.DataFrame({s: frame.close.reindex(dates).pipe(pd.to_numeric, errors="coerce")
                           for s, frame in frames.items()})
    scores = scores / scores.shift(batch["lookback"]).replace(0, np.nan) - 1.
    scores = scores.replace([np.inf, -np.inf], np.nan).loc[test]
    allocations, signals = [], []
    for i, day in enumerate(test[:-1]):
        if i % batch["rebalance_sessions"]:
            continue
        ranked = scores.loc[day].dropna().sort_index().sort_values(ascending=False, kind="stable")
        winners = list(ranked.index[:batch["topk"]])
        weights = {s: 1. / len(winners) for s in winners} if winners else {}
        allocations.append(TargetAllocation(str(day.date()), day.date(), weights))
        signals.append({"date": str(day.date()), "weights": weights})
    bars = [PriceBar(s, day.date(), float(row.open), float(row.close))
            for s, frame in execution.items() for day, row in frame.loc[test].iterrows()]
    report = run_portfolio_backtest(strategy_name="data-check-momentum", allocations=allocations, bars=bars,
                                    config=StrategyBacktestConfig(initial_capital=1_000_000., side_cost_pct=0., lot_size=100))
    positions, holdings = dict.fromkeys(execution, 0), []
    for point in report["equity_curve"]:
        for trade in report["trades"]:
            if trade["execution_date"] == point["date"]:
                positions[trade["symbol"]] += trade["shares"] * (1 if trade["action"] == "buy" else -1)
        holdings.append({"date": point["date"], "shares": dict(positions)})
    return {"scores": [{"date": str(day.date()), "symbol": symbol, "score": _value(value)}
                       for day, row in scores.iterrows() for symbol, value in row.items()],
            "signals": signals, "holdings": holdings, "backtest": report}


def _evidence(path, readset, observed, market):
    if path is None:
        return []
    document = readset.json(path)
    rows = document.get("observations")
    if not isinstance(rows, list) or len(rows) > 10000:
        raise ValueError("external evidence needs a bounded observations list")
    results = []
    seen = set()
    for item in rows:
        key = (item["symbol"], _date(item["date"]), item["basis"], item["field"])
        unit = "units" if item["field"] == "volume" and market == "etf" else UNITS.get(item["field"])
        if (key in seen or not unit or item["basis"] != "raw" or item.get("unit") != unit
                or not np.isfinite(float(item["value"])) or not str(item.get("source_url", "")).startswith(("https://", "http://"))):
            raise ValueError("external observation identity, unit or value is invalid")
        seen.add(key)
        relative = Path(item["evidence_path"])
        if (relative.is_absolute() or ".." in relative.parts
                or not str(relative) or any((path.parent / Path(*relative.parts[:i])).is_symlink()
                                            for i in range(1, len(relative.parts) + 1))):
            raise ValueError("external evidence path must be a local relative regular file")
        readset.read(path.parent / relative, item["evidence_sha256"])
        values = observed.get(key, {})
        results.append({**item, "observed": values,
                        "matches": {side: _same(value, item["value"]) for side, value in values.items()},
                        "status": "compared" if values else "outside_checked_scope",
                        "authority": "caller_reviewed_extraction; bytes bound, truth not automatically certified"})
    return results


def audit_strategy_data(primary_root, reference_root, *, output_dir, start=None, end=None, market="ashare",
                        batches=3, sample_size=16, sessions=63, seed=20261024, codes=None,
                        plan_path=None, evidence_path=None, calendar_path=None):
    """Run a bounded local diagnostic. Return the same summary written on disk.

    Replay pins both old snapshot identities and all random choices. Missing
    symbols stay in the plan and are never replaced by convenient survivors.
    """
    roots = [Path(p).expanduser().resolve(strict=True) for p in (primary_root, reference_root)]
    if market not in {"ashare", "etf"}:
        raise ValueError("market must be ashare or etf")
    output = Path(output_dir).expanduser().resolve()
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):
        raise ValueError("report output must be outside both data roots")
    if output.exists() and any(output.iterdir()):
        raise ValueError("refusing to overwrite a nonempty report directory")
    readset = _ReadSet()
    previous = readset.json(Path(plan_path).resolve()) if plan_path else None
    if previous is not None:
        _validate_plan(previous, market)
    elif start is None or end is None or _date(start) > _date(end):
        raise ValueError("provide a valid start/end range or a saved plan")
    primary = _freeze(roots[0], market, previous["inputs"]["primary"] if previous else None)
    same_input = roots[0] == roots[1] and (previous is None or previous["inputs"]["primary"] == previous["inputs"]["reference"])
    reference = primary if same_input else _freeze(
        roots[1], market, previous["inputs"]["reference"] if previous else None)
    sides = {"primary": primary, "reference": reference}
    calendar, calendar_source = None, None
    if calendar_path is not None:
        if previous:
            raise ValueError("a saved plan already fixes calendar dates; do not override its calendar")
        path = Path(calendar_path).expanduser().resolve()
        if path.suffix != ".parquet":
            raise ValueError("explicit calendar must be a Parquet date table")
        frame = readset.parquet(path)
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
        if dates.empty or dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing or dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("calendar dates must be unique ordered trading sessions")
        calendar = dates[(dates >= start) & (dates <= end)]
        calendar_source = {"kind": "caller_supplied_calendar", "path": str(path),
                           "sha256": readset.records[-1]["sha256"],
                           "authority": "caller verifies source and coverage; tool binds bytes"}
    plan = previous or _plan(**sides, market=market, start=_date(start), end=_date(end),
                             batches=batches, sample_size=sample_size, sessions=sessions, seed=seed, codes=codes,
                             calendar=calendar, calendar_source=calendar_source)
    _validate_plan(plan, market)
    output.mkdir(parents=True, exist_ok=True)
    _save(output / "plan.json", plan)
    files = {}
    for side in sides.values():
        for snapshot in side["snapshots"].values():
            files[str(snapshot["snapshot_path"] / "DATASET_MANIFEST.json")] = snapshot["identity"]["manifest_sha256"]
            files[str(snapshot["index_path"])] = snapshot["identity"]["file_index_sha256"]
    findings, spotchecks, observed, summaries = [], [], {}, []
    rng = random.Random(plan["seed"])
    for batch in plan["batches"]:
        name, frames, unavailable = batch["id"], {}, []
        calendar = pd.DatetimeIndex(batch["calendar"])
        for symbol in batch["requested_symbols"]:
            pair = {}
            for label, side in sides.items():
                try:
                    pair[label] = {basis: _read(side, basis, symbol, batch["calendar"][0], batch["test_end"], files)
                                   for basis in ("raw", "qfq")}
                except (DataContractError, OSError, ValueError) as exc:
                    _finding(findings, name, symbol, None, label, "input", "file", "input_unavailable",
                             classification="insufficient_evidence", detail=str(exc))
                    continue
                raw, qfq = pair[label]["raw"], pair[label]["qfq"]
                _compare_frames(raw, qfq, findings, name, symbol, side=label, basis="raw/qfq", paired=True)
                _raw_leads(raw, findings, name, symbol, label)
                for day, row in raw.iterrows():
                    for field in RAW_FIELDS:
                        if field in row:
                            observed.setdefault((symbol, str(day.date()), "raw", field), {})[label] = _value(pd.to_numeric(row[field], errors="coerce"))
            if len(pair) != 2:
                unavailable.append(symbol)
                continue
            for basis in ("raw", "qfq"):
                _compare_frames(pair["primary"][basis], pair["reference"][basis], findings, name, symbol,
                                side="primary/reference", basis=basis)
            raw_a, raw_b = pair["primary"]["raw"], pair["reference"]["raw"]
            equal_days = [day for day in raw_a.index.intersection(raw_b.index)
                          if all(field in raw_a and field in raw_b
                                 and _value(pd.to_numeric(raw_a.loc[day, field], errors="coerce")) is not None
                                 and _value(pd.to_numeric(raw_b.loc[day, field], errors="coerce")) is not None
                                 and _same(raw_a.loc[day, field], raw_b.loc[day, field])
                                 for field in (*PRICE_FIELDS, "volume", "amount"))]
            if equal_days:
                day = rng.choice(equal_days)
                for field in (*PRICE_FIELDS, "volume", "amount"):
                    spotchecks.append({"batch": name, "symbol": symbol, "date": str(day.date()),
                                       "basis": "raw", "field": field, "value": _value(raw_a.loc[day, field]),
                                       "unit": "units" if field == "volume" and market == "etf" else UNITS[field],
                                       "status": "awaiting_independent_verification"})
            # Keep missing/zero-trade observations visible. The small diagnostic
            # engine is only used on complete, actually traded paired windows.
            complete = all(calendar.isin(frame.index).all()
                           for side in pair.values() for frame in side.values())
            if complete:
                complete = all((pd.to_numeric(side["raw"].reindex(calendar).volume, errors="coerce") > 0).all()
                               for side in pair.values())
            if not complete:
                unavailable.append(symbol)
                _finding(findings, name, symbol, None, "both", "input", "date/volume", "not_comparable_window",
                         classification="insufficient_evidence",
                         detail="No invented suspension bars or silent date intersection in a strategy comparison.")
                continue
            for label in sides:
                extra = pair[label]["raw"].index.difference(calendar)
                for day in extra:
                    _finding(findings, name, symbol, day, label, "raw", "date", "outside_frozen_calendar",
                             detail="Verify calendar evidence before interpreting this row.")
            frames[symbol] = pair
        item = {"id": name, "requested_symbols": batch["requested_symbols"], "paired_symbols": len(frames),
                "unavailable_symbols": unavailable, "test_start": batch["test_start"], "test_end": batch["test_end"]}
        if frames:
            execution = {symbol: pair["primary"]["raw"] for symbol, pair in frames.items()}
            digest = _digest({s: [[str(day.date()), float(row.open), float(row.close)]
                                  for day, row in frame.iterrows()] for s, frame in execution.items()})
            probes = {label: _probe({s: pair[label]["qfq"] for s, pair in frames.items()}, execution, batch)
                      for label in sides}
            folder = output / name
            folder.mkdir()
            for label, probe in probes.items():
                _save(folder / (label + ".json"), probe)
            left, right = probes.values()
            signal_days = sum(a != b for a, b in zip(left["signals"], right["signals"], strict=True))
            holding_days = sum(a != b for a, b in zip(left["holdings"], right["holdings"], strict=True))
            returns = {label: probe["backtest"]["metrics"]["total_return_pct"] for label, probe in probes.items()}
            item.update(status="completed", signal_changed_days=signal_days, holding_changed_days=holding_days,
                        diagnostic_return_pct=returns, return_delta_pp=returns["primary"] - returns["reference"],
                        execution_input_sha256=dict.fromkeys(sides, digest))
        else:
            item["status"] = "unavailable"
        summaries.append(item)
    external = _evidence(Path(evidence_path).resolve() if evidence_path else None, readset, observed, market)
    for path, expected in files.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError(f"frozen input changed during strategy check: {path}")
    if not readset.unchanged() or any(sha256_file(Path(row["path"])) != row["sha256"] for row in readset.records):
        raise ValueError("plan or external evidence changed during strategy check")
    counts = dict(Counter(row["classification"] for row in findings))
    summary = {"version": VERSION, "status": "partial" if any(item["unavailable_symbols"] for item in summaries) else "completed",
               "plan_sha256": _digest(plan), "market": market, "inputs": plan["inputs"], "batches": summaries,
               "finding_counts": counts, "agreed_spotcheck_rows": len(spotchecks),
               "external_observations_checked": sum(row["status"] == "compared" for row in external),
               "input_files_rechecked": len(files), "input_files_unchanged": True,
               "network_requests": 0, "data_writes": 0, "model_fits": 0,
               "limits": ["Differences and return convergence cannot certify either source.",
                          "Gross diagnostic returns; no investment-performance conclusion, corporate-action cash-flow or trading-limit simulation.",
                          "Frozen current snapshots, not historical point-in-time reconstruction.",
                          "Only selected symbols/windows; momentum can miss model-specific feature defects.",
                          "Evidence values are caller-reviewed extractions, not automatically authenticated facts."]}
    _write_csv(output / "findings.csv", findings, FINDING_COLUMNS)
    _write_csv(output / "agreed_spotchecks.csv", spotchecks,
               ("batch", "symbol", "date", "basis", "field", "value", "unit", "status"))
    _save(output / "external_comparison.json", external)
    _save(output / "read-set.json", {"indexed_files": files, "other_inputs": readset.records})
    _save(output / "summary.json", summary)
    lines = ["# 随机策略数据核查", "", f"状态：{summary['status']}；计划 SHA256：`{summary['plan_sha256']}`。", "",
             "固定两套本地快照和随机配置，使用各自 QFQ 产生动量信号，两边统一使用 primary RAW 计价。",
             "它只生成诊断，不下载、不修复、不发布，不作为读取、训练或回测门禁。", "",
             "| 批次 | 请求/可配对证券 | 信号不同天数 | 持仓不同天数 | 诊断收益差（百分点） |",
             "| --- | --- | --- | --- | --- |"]
    for item in summaries:
        lines.append(f"| {item['id']} | {len(item['requested_symbols'])}/{item['paired_symbols']} | "
                     f"{item.get('signal_changed_days', '未运行')} | {item.get('holding_changed_days', '未运行')} | "
                     f"{item.get('return_delta_pp', '未运行')} |")
    lines.extend(["", "`findings.csv` 按证券、日期、字段列出矛盾、差异和证据不足；不同复权价格不自动判错。",
                  "`agreed_spotchecks.csv` 随机抽取双方一致的记录，供第三方核实；未提供独立证据就仍是待核实。",
                  "`external_comparison.json` 只比较人工核读提取的值，原文字节和摘要绑定不等于自动认证真伪。",
                  "每批两份 JSON 保存全部评分、目标、持仓、交易及诊断收益。相同计划可重放，缺失证券不补换。", "",
                  "当前首版只使用现有轻量动量和回测引擎；不训练 LightGBM。收益为统一计价的无费诊断，",
                  "不模拟分红现金流、完整停牌/涨跌停成交规则，不用来评选策略或判断真实可实现收益。",
                  "筛查只覆盖选中区间，不证明全库无误。成交均价越界须另查单位和大宗交易，不能据此直接改成交额。"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def add_arguments(parser):
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--market", choices=("ashare", "etf"), default="ashare")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=16)
    parser.add_argument("--sessions", type=int, default=63)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--codes", nargs="+")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--calendar", type=Path, help="Explicit previously verified Parquet trading calendar")
    parser.add_argument("--output-dir", type=Path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    add_arguments(parser)
    args = parser.parse_args(argv)
    output = args.output_dir or reports_root() / "data_checks" / (
        "strategy-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    try:
        result = audit_strategy_data(
            args.root, args.reference_root, output_dir=output, start=args.start, end=args.end,
            market=args.market, batches=args.batches, sample_size=args.sample_size,
            sessions=args.sessions, seed=args.seed, codes=args.codes, plan_path=args.plan, evidence_path=args.evidence,
            calendar_path=args.calendar)
    except (DataContractError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "incomplete", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "output_dir": str(output), "plan_sha256": result["plan_sha256"]}))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
