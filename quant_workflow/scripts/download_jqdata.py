#!/usr/bin/env python3
"""Download A-share daily data from JQData SDK.

The script supports a browser-safe dry run that authenticates, resolves the
stock universe, and prints a small preview without writing parquet files.  Real
downloads still write one parquet file per 6-digit symbol.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, MutableMapping, Sequence

import pandas as pd

try:
    multiprocessing.set_start_method("fork", force=True)
except RuntimeError:
    pass

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import projects_root, quant_data_root, workflow_root  # noqa: E402


WORKFLOW_ROOT = workflow_root()
PROJECTS_ROOT = projects_root()
QUANT_DATA = quant_data_root()
PARQUET_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "jqdata"
JQDATA_ENV_FILE_ENV = "JQDATA_ENV_FILE"
PRIVATE_ENV_FILE = Path.home() / ".agents" / "private" / "quant-workbench" / "jqdata.env"
LEGACY_ENV_FILE = PROJECTS_ROOT / "qlib" / ".env"
DEFAULT_DRY_RUN_LIMIT = 3
CUSTOM_CODE_PATTERN = re.compile(r"^(?P<symbol>\d{6})(?:\.(?P<exchange>XSHG|XSHE|XBSE))?$")
_USE_DEFAULT_ENV_FILES = object()


def load_env_file(path: Path, environ: MutableMapping[str, str] | None = None) -> None:
    """Load KEY=VALUE pairs without overriding already configured variables."""
    target = os.environ if environ is None else environ
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and key not in target:
            target[key] = value.strip()


def default_env_files(environ: MutableMapping[str, str] | None = None) -> list[Path]:
    """Return JQData env files in safe precedence order."""
    target = os.environ if environ is None else environ
    configured = target.get(JQDATA_ENV_FILE_ENV) or os.environ.get(JQDATA_ENV_FILE_ENV)
    paths: list[Path] = []
    if configured:
        paths.append(Path(configured).expanduser())
    paths.append(PRIVATE_ENV_FILE)
    paths.append(LEGACY_ENV_FILE)
    return paths


def load_env_files(paths: Sequence[Path], environ: MutableMapping[str, str] | None = None) -> None:
    for path in paths:
        load_env_file(path, environ=environ)


def _load_jq_module():
    import jqdatasdk as jq

    return jq


def authenticate(
    jq_module: Any,
    environ: MutableMapping[str, str],
    env_file: Path | Sequence[Path] | None | object = _USE_DEFAULT_ENV_FILES,
) -> bool:
    if env_file is _USE_DEFAULT_ENV_FILES:
        load_env_files(default_env_files(environ), environ=environ)
    elif env_file is not None:
        if isinstance(env_file, (str, os.PathLike)):
            load_env_file(Path(env_file), environ=environ)
        else:
            load_env_files([Path(item) for item in env_file], environ=environ)
    username = environ.get("JQ_USERNAME")
    password = environ.get("JQ_PASSWORD")
    if not username or not password:
        print("ERROR: JQData credentials are not configured. Set JQ_USERNAME and JQ_PASSWORD.", file=sys.stderr)
        return False
    jq_module.auth(username, password)
    if not jq_module.is_auth():
        print("ERROR: JQData auth failed", file=sys.stderr)
        return False
    return True


def normalize_security_code(raw_code: str) -> str:
    """Normalize browser/user stock codes to JQData security codes."""
    raw = (raw_code or "").strip().upper()
    if not raw:
        raise ValueError("Empty security code is not allowed.")

    if raw.startswith(("SH", "SZ", "BJ")) and len(raw) == 8 and raw[2:].isdigit():
        prefix = raw[:2]
        symbol = raw[2:]
        exchange = {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBSE"}[prefix]
        return f"{symbol}.{exchange}"

    matched = CUSTOM_CODE_PATTERN.match(raw)
    if not matched:
        raise ValueError(f"Unsupported security code format: {raw_code}")

    symbol = matched.group("symbol")
    exchange = matched.group("exchange")
    if exchange:
        return f"{symbol}.{exchange}"

    if symbol.startswith(("600", "601", "603", "605", "688", "689", "900")):
        return f"{symbol}.XSHG"
    if symbol.startswith(("000", "001", "002", "003", "300", "301", "200")):
        return f"{symbol}.XSHE"
    if symbol.startswith(("430", "83", "87", "88", "920")):
        return f"{symbol}.XBSE"
    raise ValueError(f"Cannot infer exchange for security code: {raw_code}")


def parse_custom_universe(universe: str) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    raw_items = re.split(r"[,;\s]+", universe or "")
    for item in raw_items:
        if not item.strip():
            continue
        code = normalize_security_code(item)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def resolve_universe(jq_module: Any, universe: str, end: str) -> list[str]:
    name = (universe or "csi300").strip().lower()
    if name == "csi300":
        return list(jq_module.get_index_stocks("000300.XSHG", date=end))
    if name == "csi500":
        return list(jq_module.get_index_stocks("000905.XSHG", date=end))
    if name in {"all", "stock", "stocks"}:
        return list(jq_module.get_all_securities(types=["stock"]).index)

    custom = parse_custom_universe(universe)
    if custom:
        return custom
    return list(jq_module.get_all_securities(types=["stock"]).index)


def download_one(jq_module: Any, code: str, start: str, end: str, out_dir: Path) -> int:
    try:
        df = jq_module.get_price(
            code,
            start_date=start,
            end_date=end,
            frequency="daily",
            skip_paused=True,
            fq="pre",
        )
        if df is None or df.empty:
            return 0
        df = df.reset_index()
        date_col = "time" if "time" in df.columns else "date" if "date" in df.columns else df.columns[0]
        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"])
        keep = [col for col in ["date", "open", "close", "high", "low", "volume"] if col in df.columns]
        df = df[keep].sort_values("date").drop_duplicates(subset=["date"])

        sym = code.split(".")[0]
        path = out_dir / f"{sym}.parquet"
        df.to_parquet(path, index=False)
        return len(df)
    except Exception as exc:  # noqa: BLE001 - surface per-symbol failures while continuing.
        print(f"  FAIL {code}: {exc}", file=sys.stderr)
        return -1


def _query_count_text(jq_module: Any) -> str:
    try:
        return str(jq_module.get_query_count())
    except Exception:
        return "unavailable"


def main(
    argv: Sequence[str] | None = None,
    jq_module: Any | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Download A-share daily data from JQData SDK.")
    parser.add_argument("--universe", default="csi300")
    parser.add_argument("--start", default="2025-03-03")
    parser.add_argument("--end", default="2025-06-10")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=PARQUET_DIR)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional credential env file. Defaults to JQDATA_ENV_FILE, private jqdata.env, then legacy qlib/.env.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Authenticate and resolve symbols without writing parquet files.")
    args = parser.parse_args(argv)

    env = os.environ if environ is None else environ
    jq = _load_jq_module() if jq_module is None else jq_module

    env_file = args.env_file if args.env_file is not None else _USE_DEFAULT_ENV_FILES
    if not authenticate(jq, env, env_file=env_file):
        return 1

    try:
        stocks = resolve_universe(jq, args.universe, args.end)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    resolved_count = len(stocks)
    if args.limit > 0:
        stocks = stocks[: args.limit]
    elif args.dry_run:
        stocks = stocks[:DEFAULT_DRY_RUN_LIMIT]

    print(f"JQData auth OK. Query count: {_query_count_text(jq)}")
    print(
        f"Universe: {args.universe}; resolved stocks: {resolved_count}; "
        f"selected stocks: {len(stocks)}; range: {args.start} -> {args.end}"
    )
    if args.dry_run:
        preview = ", ".join(stocks[:10])
        print(f"JQData dry run OK. No parquet files were written. Preview: {preview}")
        return 0

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    ok, fail = 0, 0
    t0 = time.time()
    for i, code in enumerate(stocks, 1):
        n = download_one(jq, code, args.start, args.end, out_dir)
        if n > 0:
            ok += 1
            if i <= 3 or i % 20 == 0:
                print(f"  [{i}/{len(stocks)}] {code}: {n} rows")
        else:
            fail += 1
        time.sleep(args.sleep)

    dt = time.time() - t0
    print(f"\nDone in {dt:.0f}s. OK={ok}, FAIL={fail}")
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
