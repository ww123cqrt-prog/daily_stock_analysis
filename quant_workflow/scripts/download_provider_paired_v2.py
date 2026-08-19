#!/usr/bin/env python3
"""Build a bounded provider-native raw/qfq A-share incremental bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_baostock_v2 import (  # noqa: E402
    atomic_json,
    atomic_parquet,
    canonical_symbol,
    sha256_file,
)
from market_data_clients import (  # noqa: E402
    EASTMONEY_KLINE_FIELDS1,
    EASTMONEY_KLINE_FIELDS2,
    EASTMONEY_KLINE_URL,
    eastmoney_secid,
    fetch_eastmoney_daily,
    parse_eastmoney_kline_payload,
)


PRICE_COLUMNS = ("open", "high", "low", "close")


class ProviderPairedError(RuntimeError):
    """The provider-native raw/qfq bundle cannot satisfy its contract."""


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _load_universe(
    path: Path,
    *,
    base_latest_trade_date: str,
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProviderPairedError("frozen universe is unavailable or invalid") from exc
    if not isinstance(payload, dict):
        raise ProviderPairedError("frozen universe must be a JSON object")
    if payload.get("dataset_id") != "ashare_daily_qfq":
        raise ProviderPairedError("frozen universe is not the formal A-share qfq dataset")
    snapshot_id = payload.get("snapshot_id")
    manifest_sha256 = payload.get("manifest_sha256")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise ProviderPairedError("frozen universe snapshot identity is missing")
    if not _is_sha256(manifest_sha256):
        raise ProviderPairedError("frozen universe manifest identity is invalid")
    if payload.get("latest_trade_date") != base_latest_trade_date:
        raise ProviderPairedError("frozen universe does not match the base date")
    values = payload.get("all")
    if not isinstance(values, list) or not values:
        raise ProviderPairedError("frozen universe is empty")
    result: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        symbol, provider_code = canonical_symbol(str(value))
        if symbol not in seen:
            result.append((symbol, provider_code))
            seen.add(symbol)
    if payload.get("symbol_count") != len(result):
        raise ProviderPairedError("frozen universe symbol count is stale")
    return {
        "snapshot_id": snapshot_id,
        "manifest_sha256": str(manifest_sha256),
    }, result


def _normalize_bars(frame: pd.DataFrame, *, symbol: str, source_id: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["date", *PRICE_COLUMNS, "volume", "amount", "turn", "pctChg", "symbol", "source_id"]
        )
    required = {"date", *PRICE_COLUMNS, "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ProviderPairedError(f"{source_id} fields are incomplete for {symbol}")
    normalized = frame.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.normalize()
    for column in (*PRICE_COLUMNS, "volume", "amount", "turn", "pctChg"):
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    if normalized[["date", *PRICE_COLUMNS, "volume"]].isna().any().any():
        raise ProviderPairedError(f"invalid {source_id} values for {symbol}")
    if normalized["date"].duplicated().any():
        raise ProviderPairedError(f"duplicate {source_id} dates for {symbol}")
    normalized = normalized.sort_values("date", kind="stable")
    if (normalized[[*PRICE_COLUMNS, "volume"]] < 0).any().any():
        raise ProviderPairedError(f"negative {source_id} values for {symbol}")
    # EastMoney reports A-share volume in hands; quant_data stores shares.
    normalized["volume"] = normalized["volume"] * 100
    normalized["symbol"] = symbol
    normalized["source_id"] = source_id
    keep = [
        "date",
        *PRICE_COLUMNS,
        *[column for column in ("preclose", "volume", "amount", "turn", "pctChg") if column in normalized],
        "symbol",
        "source_id",
    ]
    return normalized[keep].reset_index(drop=True)


def _assert_pair(raw: pd.DataFrame, qfq: pd.DataFrame, *, symbol: str) -> None:
    raw_dates = raw["date"].tolist()
    qfq_dates = qfq["date"].tolist()
    if raw_dates != qfq_dates:
        raise ProviderPairedError(f"paired raw/qfq dates do not match for {symbol}")
    if not raw.empty:
        for column in PRICE_COLUMNS:
            if (raw[column] <= 0).any() or (qfq[column] <= 0).any():
                raise ProviderPairedError(f"paired raw/qfq prices are invalid for {symbol}")


def _fetch_pair(
    session: requests.Session,
    *,
    symbol: str,
    start: str,
    through: str,
    timeout: float,
    use_system_proxy: bool,
    before_request: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def fetch_one(fqt: int) -> pd.DataFrame:
        if before_request is not None:
            before_request()
        try:
            return fetch_eastmoney_daily(
                symbol,
                start,
                through,
                fqt=fqt,
                session=session,
                timeout=timeout,
                use_system_proxy=use_system_proxy,
            )
        except Exception:
            curl = None
            if sys.platform == "win32":
                windows_curl = Path(r"C:\Windows\System32\curl.exe")
                if windows_curl.is_file():
                    curl = str(windows_curl)
            curl = curl or shutil.which("curl.exe") or shutil.which("curl")
            if not curl:
                raise
            if before_request is not None:
                before_request()
            params = urlencode(
                {
                    "secid": eastmoney_secid(symbol),
                    "fields1": EASTMONEY_KLINE_FIELDS1,
                    "fields2": EASTMONEY_KLINE_FIELDS2,
                    "klt": "101",
                    "fqt": str(fqt),
                    "beg": start.replace("-", ""),
                    "end": through.replace("-", ""),
                }
            )
            response = subprocess.run(
                [
                    curl,
                    "-L",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--max-time",
                    str(int(timeout)),
                    f"{EASTMONEY_KLINE_URL}?{params}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            return parse_eastmoney_kline_payload(json.loads(response.stdout))

    raw = _normalize_bars(
        fetch_one(0),
        symbol=symbol,
        source_id="eastmoney_raw",
    )
    qfq = _normalize_bars(
        fetch_one(1),
        symbol=symbol,
        source_id="eastmoney_qfq_provider",
    )
    _assert_pair(raw, qfq, symbol=symbol)
    return raw, qfq


def _resume_contract_matches(
    manifest: dict[str, Any],
    *,
    args: argparse.Namespace,
    universe_identity: dict[str, str],
    universe_sha256: str,
) -> bool:
    return (
        manifest.get("bundle_kind") == "provider_paired_raw_qfq"
        and manifest.get("bundle_mode") == "incremental"
        and manifest.get("base_latest_trade_date") == args.base_latest_trade_date
        and manifest.get("base_qfq_snapshot_id") == universe_identity["snapshot_id"]
        and manifest.get("base_qfq_manifest_sha256")
        == universe_identity["manifest_sha256"]
        and manifest.get("requested_start") == args.start
        and manifest.get("requested_through") == args.through
        and manifest.get("universe_file_sha256") == universe_sha256
    )


def _cached_pair_is_valid(
    cached: dict[str, Any],
    *,
    output_dir: Path,
    raw_path: Path,
    qfq_path: Path,
    symbol: str,
    start: str,
    through: str,
) -> bool:
    expected_raw = raw_path.relative_to(output_dir).as_posix()
    expected_qfq = qfq_path.relative_to(output_dir).as_posix()
    if (
        cached.get("raw_path") != expected_raw
        or cached.get("qfq_provider_path") != expected_qfq
        or not _is_sha256(cached.get("raw_sha256"))
        or not _is_sha256(cached.get("qfq_provider_sha256"))
        or raw_path.is_symlink()
        or qfq_path.is_symlink()
        or not raw_path.is_file()
        or not qfq_path.is_file()
    ):
        return False
    try:
        if sha256_file(raw_path) != cached["raw_sha256"] or sha256_file(
            qfq_path
        ) != cached["qfq_provider_sha256"]:
            return False
        raw = pd.read_parquet(raw_path)
        qfq = pd.read_parquet(qfq_path)
        required = {"date", *PRICE_COLUMNS, "volume", "symbol", "source_id"}
        if required.difference(raw.columns) or required.difference(qfq.columns):
            return False
        raw_dates = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
        qfq_dates = pd.to_datetime(qfq["date"], errors="coerce").dt.normalize()
        if (
            raw_dates.isna().any()
            or qfq_dates.isna().any()
            or raw_dates.duplicated().any()
            or qfq_dates.duplicated().any()
            or not raw_dates.reset_index(drop=True).equals(qfq_dates.reset_index(drop=True))
            or len(raw) != int(cached.get("raw_rows", -1))
        ):
            return False
        if raw.empty:
            return cached.get("first_date") is None and cached.get("last_date") is None
        first_date = raw_dates.iloc[0].date().isoformat()
        last_date = raw_dates.iloc[-1].date().isoformat()
        return (
            cached.get("first_date") == first_date
            and cached.get("last_date") == last_date
            and start <= first_date <= last_date <= through
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def build_bundle(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    universe_path = Path(args.universe_file).expanduser().resolve()
    universe_identity, symbols = _load_universe(
        universe_path,
        base_latest_trade_date=args.base_latest_trade_date,
    )
    universe_sha256 = sha256_file(universe_path)
    source_retrieved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    index_path = output_dir / "BUNDLE_FILE_INDEX.parquet"
    existing: dict[str, dict[str, Any]] = {}
    manifest_path = output_dir / "BUNDLE_MANIFEST.json"
    if args.resume and index_path.is_file() and manifest_path.is_file():
        try:
            old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if _resume_contract_matches(
                old_manifest,
                args=args,
                universe_identity=universe_identity,
                universe_sha256=universe_sha256,
            ):
                old_index = pd.read_parquet(index_path)
                for row in old_index.to_dict(orient="records"):
                    existing[str(row["symbol"])] = row
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            existing = {}

    session = requests.Session()
    session.trust_env = args.use_system_proxy
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    requests_used = 0
    consecutive_failures = 0
    raw_dir = output_dir / "raw"
    qfq_dir = output_dir / "qfq_provider"
    raw_dir.mkdir(parents=True, exist_ok=True)
    qfq_dir.mkdir(parents=True, exist_ok=True)

    def consume_request() -> None:
        nonlocal requests_used
        if requests_used >= args.max_requests:
            raise ProviderPairedError("provider request budget exhausted")
        requests_used += 1

    for position, (symbol, provider_code) in enumerate(symbols, start=1):
        raw_path = raw_dir / f"{symbol}.parquet"
        qfq_path = qfq_dir / f"{symbol}.parquet"
        cached = existing.get(symbol)
        if (
            args.resume
            and cached
            and _cached_pair_is_valid(
                cached,
                output_dir=output_dir,
                raw_path=raw_path,
                qfq_path=qfq_path,
                symbol=symbol,
                start=args.start,
                through=args.through,
            )
        ):
            rows.append(cached)
            continue
        try:
            raw, qfq = _fetch_pair(
                session,
                symbol=symbol,
                start=args.start,
                through=args.through,
                timeout=args.timeout,
                use_system_proxy=args.use_system_proxy,
                before_request=consume_request,
            )
            if raw.empty and qfq.empty:
                first_date = last_date = None
            else:
                first_date = str(raw["date"].iloc[0].date())
                last_date = str(raw["date"].iloc[-1].date())
            atomic_parquet(raw, raw_path)
            atomic_parquet(qfq, qfq_path)
            rows.append(
                {
                    "symbol": symbol,
                    "provider_code": provider_code,
                    "raw_path": raw_path.relative_to(output_dir).as_posix(),
                    "qfq_provider_path": qfq_path.relative_to(output_dir).as_posix(),
                    "raw_rows": len(raw),
                    "first_date": first_date,
                    "last_date": last_date,
                    "raw_sha256": sha256_file(raw_path),
                    "qfq_provider_sha256": sha256_file(qfq_path),
                }
            )
            consecutive_failures = 0
        except Exception as exc:  # pragma: no cover - network failure path
            failures.append({"symbol": symbol, "error": str(exc)})
            consecutive_failures += 1
            if (
                requests_used >= args.max_requests
                or consecutive_failures >= args.max_consecutive_failures
            ):
                break
        if position < len(symbols):
            time.sleep(args.sleep)

    status = "ready" if not failures and len(rows) == len(symbols) else "blocked"
    index = pd.DataFrame(rows)
    atomic_parquet(index, index_path)
    atomic_json(
        output_dir / "failures.json",
        {"failed_symbols": failures, "count": len(failures)},
    )
    atomic_json(
        output_dir / "BUNDLE_MANIFEST.json",
        {
            "schema_version": 1,
            "bundle_kind": "provider_paired_raw_qfq",
            "bundle_mode": "incremental",
            "status": status,
            "source": {
                "provider": "eastmoney",
                "interface": "push2his_kline_fqt_0_and_fqt_1",
                "execution_environment": "private_remote_windows",
            },
            "source_retrieved_at": source_retrieved_at,
            "base_latest_trade_date": args.base_latest_trade_date,
            "base_qfq_snapshot_id": universe_identity["snapshot_id"],
            "base_qfq_manifest_sha256": universe_identity["manifest_sha256"],
            "requested_start": args.start,
            "requested_through": args.through,
            "requested_symbols": len(symbols),
            "completed_symbols": len(rows),
            "empty_symbols": int(sum(1 for row in rows if not row.get("raw_rows"))),
            "failed_symbols": len(failures),
            "requests_used": requests_used,
            "index_path": "BUNDLE_FILE_INDEX.parquet",
            "universe_file_sha256": universe_sha256,
        },
    )
    if status != "ready":
        raise ProviderPairedError(
            f"provider-paired bundle blocked: completed={len(rows)} failed={len(failures)}"
        )
    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-latest-trade-date", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--through", required=True)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--max-requests", type=int, default=12000)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-system-proxy", action="store_true")
    args = parser.parse_args(argv)
    if args.sleep < 0.25:
        parser.error("--sleep must be at least 0.25 seconds")
    if args.max_consecutive_failures < 1 or args.max_consecutive_failures > 3:
        parser.error("--max-consecutive-failures must be between 1 and 3")
    try:
        output = build_bundle(args)
    except (OSError, ProviderPairedError, ValueError) as exc:
        print(f"provider-paired v2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
