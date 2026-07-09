"""
Merge multi-source A-share daily bars (Baostock/JQData/EastMoney/AkShare),
then convert to Qlib bin.

All sources are in pre-adjusted (qfq/pre) price space, but each provider
computes its qfq curve against its own base/factors. To avoid splicing two
differently-normalized curves, the FIRST available source (Baostock long
history) is authoritative for every date it covers; later sources only FILL
dates the base still lacks (mid-history gaps and the latest tail), whole-row.
OHLC on overlapping dates is never overwritten. This keeps a single, internally
consistent price/volume curve per symbol.

`--no-overwrite` means "skip replacing existing output files"; it may still
create missing directories and missing files required to complete the provider.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri, quant_data_root, reports_root, workflow_root  # noqa: E402
from _symbols import to_qlib_name  # noqa: E402


WORKFLOW_ROOT = workflow_root()
QUANT_DATA = quant_data_root()
BS_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "baostock_qfq"
JQ_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "jqdata"
EASTMONEY_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "eastmoney_qfq"
AKSHARE_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "akshare_qfq"
MERGED_DIR = QUANT_DATA / "parquet" / "ashare_1d" / "merged_baostock_jqdata"
QLIB_BIN_DIR = qlib_provider_uri()

REQUIRED_FIELDS = ["open", "close", "high", "low", "volume"]
OPTIONAL_FIELDS = ["amount", "turn", "pctChg"]
QLIB_FIELDS = ["open", "close", "high", "low", "volume", "factor"]


def normalize_frame(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        # A truncated/corrupt parquet (e.g. an interrupted download) must not
        # crash the whole merge. Skip this source file with a warning; other
        # sources still contribute for the symbol.
        print(f"  WARN unreadable parquet, skipping: {path} ({exc})")
        return pd.DataFrame(columns=["date"] + REQUIRED_FIELDS + OPTIONAL_FIELDS)
    if "date" not in df.columns and df.index.name == "date":
        df = df.reset_index()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    keep = ["date"] + [c for c in REQUIRED_FIELDS + OPTIONAL_FIELDS if c in df.columns]
    df = df[keep]
    for col in keep:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    return df


def merge_sources(sym: str, ordered_sources: list[tuple[str, Path | None]]) -> pd.DataFrame | None:
    frames: list[tuple[str, pd.DataFrame]] = []
    for name, path in ordered_sources:
        if path is None or not path.exists():
            continue
        frame = normalize_frame(path)
        if not frame.empty:
            frames.append((name, frame))

    if not frames:
        return None

    _, first = frames[0]
    merged = first.set_index("date")

    for name, frame in frames[1:]:
        # Base source (Baostock long history) is authoritative for every date it
        # already covers. Newer sources ONLY fill dates the base lacks — mid-
        # history gaps and the latest tail. We deliberately do NOT overwrite
        # OHLC on overlapping dates: each provider computes qfq (forward-adjust)
        # against its own base/factors, so overwriting the overlap would splice
        # two differently-normalized curves and inject a price discontinuity
        # into training/backtest. Rows are appended whole (all fields together)
        # so a filled date stays internally consistent within one source.
        existing_dates = merged.index
        overlay = frame.set_index("date")
        source_only_dates = overlay.index.difference(existing_dates)
        if len(source_only_dates) == 0:
            continue
        merged = merged.reindex(existing_dates.union(source_only_dates))
        for col in REQUIRED_FIELDS + OPTIONAL_FIELDS:
            if col in overlay.columns:
                merged.loc[source_only_dates, col] = overlay.loc[source_only_dates, col]

    merged = merged.reset_index().rename(columns={"index": "date"})

    if merged is None or merged.empty:
        return None

    missing = [c for c in REQUIRED_FIELDS if c not in merged.columns]
    if missing:
        print(f"  SKIP {sym}: missing {missing}")
        return None

    merged = merged.dropna(subset=["date"] + REQUIRED_FIELDS)
    if merged.empty:
        return None

    if "factor" not in merged.columns:
        merged["factor"] = 1.0

    keep = ["date"] + [c for c in REQUIRED_FIELDS + ["factor"] + OPTIONAL_FIELDS if c in merged.columns]
    return merged[keep].sort_values("date").drop_duplicates(subset=["date"])


def merge_one(
    sym: str,
    bs_path: Path | None,
    jq_path: Path | None,
    eastmoney_path: Path | None = None,
    akshare_path: Path | None = None,
) -> pd.DataFrame | None:
    return merge_sources(
        sym,
        [
            ("baostock", bs_path),
            ("jqdata", jq_path),
            ("eastmoney", eastmoney_path),
            ("akshare", akshare_path),
        ],
    )


def _source_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.glob("*.parquet")} if directory.exists() else {}


def build_merged_parquet(
    bs_dir: Path,
    jq_dir: Path,
    out_dir: Path,
    overwrite: bool,
    extra_source_dirs: list[tuple[str, Path]] | None = None,
) -> int:
    if overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_maps: list[tuple[str, dict[str, Path]]] = [
        ("baostock", _source_files(bs_dir)),
        ("jqdata", _source_files(jq_dir)),
    ]
    for name, directory in extra_source_dirs or []:
        source_maps.append((name, _source_files(directory)))

    symbols = sorted(set().union(*(set(files) for _, files in source_maps)))

    for name, files in source_maps:
        print(f"{name} stocks: {len(files)}")
    print(f"Union stocks:    {len(symbols)}")
    if len(source_maps) >= 2:
        base_name, base_files = source_maps[0]
        for name, files in source_maps[1:]:
            print(f"Overlap {base_name}/{name}: {len(set(base_files) & set(files))}")

    merged_count = 0
    skipped_existing = 0
    t0 = time.time()
    for i, sym in enumerate(symbols, 1):
        out_path = out_dir / f"{sym}.parquet"
        if not overwrite and out_path.exists():
            merged_count += 1
            skipped_existing += 1
            if i <= 3 or i % 500 == 0 or i == len(symbols):
                print(f"  [{i}/{len(symbols)}] {sym}: skip existing {out_path}")
            continue
        df = merge_sources(sym, [(name, files.get(sym)) for name, files in source_maps])
        if df is None or df.empty:
            continue
        df.to_parquet(out_path, index=False)
        merged_count += 1
        if i <= 3 or i % 500 == 0 or i == len(symbols):
            print(
                f"  [{i}/{len(symbols)}] {sym}: {len(df)} rows "
                f"({df['date'].min().date()} ~ {df['date'].max().date()})"
            )

    skipped_text = f", skipped existing {skipped_existing}" if skipped_existing else ""
    print(f"Merged {merged_count} stocks{skipped_text} in {time.time() - t0:.0f}s -> {out_dir}")
    return merged_count


def write_smoke_fixture(base_dir: Path, overwrite: bool) -> tuple[Path, Path, Path, Path]:
    """Create a tiny isolated Baostock/JQData fixture for browser smoke runs."""
    if overwrite and base_dir.exists():
        shutil.rmtree(base_dir)
    bs_dir = base_dir / "baostock_qfq"
    jq_dir = base_dir / "jqdata"
    merged_dir = base_dir / "merged"
    qlib_dir = base_dir / "qlib"
    bs_dir.mkdir(parents=True, exist_ok=True)
    jq_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "open": [10.0, 10.2],
            "high": [10.4, 10.5],
            "low": [9.8, 10.0],
            "close": [10.1, 10.3],
            "volume": [1000.0, 1200.0],
            "amount": [10100.0, 12360.0],
            "turn": [0.5, 0.6],
            "pctChg": [0.0, 1.98],
        }
    ).to_parquet(bs_dir / "000001.parquet", index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-04"],
            "open": [10.25, 10.4],
            "high": [10.55, 10.8],
            "low": [10.05, 10.2],
            "close": [10.35, 10.7],
            "volume": [9999.0, 1400.0],
        }
    ).to_parquet(jq_dir / "000001.parquet", index=False)
    return bs_dir, jq_dir, merged_dir, qlib_dir


def run_smoke_fixture(base_dir: Path, overwrite: bool = True) -> int:
    """Run the full merge + Qlib bin conversion on a local tiny fixture."""
    bs_dir, jq_dir, merged_dir, qlib_dir = write_smoke_fixture(base_dir, overwrite=overwrite)
    print(f"[smoke] Merge fixture root: {base_dir}")
    merged_count = build_merged_parquet(bs_dir, jq_dir, merged_dir, overwrite=overwrite)
    if merged_count == 0:
        return 1
    return convert_to_bin(merged_dir, qlib_dir, overwrite=overwrite)


def convert_to_bin(src_dir: Path, dst_dir: Path, overwrite: bool, dropna_fields: list[str] | None = None) -> int:
    if overwrite and dst_dir.exists():
        shutil.rmtree(dst_dir)

    pq_files = sorted(src_dir.glob("*.parquet"))
    if not pq_files:
        print(f"ERROR: No Parquet files in {src_dir}")
        return 1

    all_dates: set[str] = set()
    instruments: list[tuple[str, str, str]] = []
    stock_data: dict[str, pd.DataFrame] = {}

    required_for_row = REQUIRED_FIELDS if dropna_fields is None else list(dropna_fields)

    print(f"Reading {len(pq_files)} merged Parquet files...")
    for i, pq in enumerate(pq_files, 1):
        sym = pq.stem
        qlib_name = to_qlib_name(sym)
        df = pd.read_parquet(pq)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"] + required_for_row)
        df = df.sort_values("date").drop_duplicates(subset=["date"])
        if df.empty:
            continue

        if "factor" not in df.columns:
            df["factor"] = 1.0

        date_str = df["date"].dt.strftime("%Y-%m-%d")
        all_dates.update(date_str.tolist())
        instruments.append((qlib_name, date_str.iloc[0], date_str.iloc[-1]))
        stock_data[qlib_name] = df[["date"] + QLIB_FIELDS].copy()

        if i % 1000 == 0:
            print(f"  read {i}/{len(pq_files)}")

    calendar = sorted(all_dates)
    if not calendar:
        print("ERROR: Empty calendar")
        return 1

    date_to_idx = {d: i for i, d in enumerate(calendar)}
    cal_dir = dst_dir / "calendars"
    inst_dir = dst_dir / "instruments"
    feat_dir = dst_dir / "features"
    cal_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    for name in ["day.txt", "day_future.txt"]:
        path = cal_dir / name
        if not overwrite and path.exists():
            continue
        with path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(calendar) + "\n")

    instruments_path = inst_dir / "all.txt"
    if overwrite or not instruments_path.exists():
        with instruments_path.open("w", encoding="utf-8") as handle:
            for name, start, end in sorted(instruments):
                handle.write(f"{name}\t{start}\t{end}\n")

    print(f"Writing Qlib bin for {len(stock_data)} stocks...")
    skipped_existing = 0
    for i, (qlib_name, df) in enumerate(sorted(stock_data.items()), 1):
        date_str = df["date"].dt.strftime("%Y-%m-%d")
        cal_idx = date_str.map(date_to_idx).astype(np.int64).to_numpy()
        start_idx = int(cal_idx.min())
        length = int(cal_idx.max()) - start_idx + 1
        offsets = cal_idx - start_idx

        out_dir = feat_dir / qlib_name.lower()
        out_dir.mkdir(parents=True, exist_ok=True)

        for field in QLIB_FIELDS:
            field_path = out_dir / f"{field}.day.bin"
            if not overwrite and field_path.exists():
                skipped_existing += 1
                continue
            data = np.full(length, np.nan, dtype=np.float32)
            data[offsets] = df[field].astype(np.float32).to_numpy()
            with field_path.open("wb") as handle:
                np.array([start_idx], dtype="<f").tofile(handle)
                data.tofile(handle)

        if i % 1000 == 0:
            print(f"  wrote {i}/{len(stock_data)}")

    total_size = sum(p.stat().st_size for p in dst_dir.rglob("*") if p.is_file())
    print(f"Qlib bin: {len(stock_data)} stocks, {len(calendar)} days -> {dst_dir}")
    print(f"  Date range: {calendar[0]} ~ {calendar[-1]}")
    print(f"  Size: {total_size / 1024 / 1024:.0f} MB")
    if skipped_existing:
        print(f"  Skipped existing bin files: {skipped_existing}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-dir", type=Path, default=BS_DIR)
    parser.add_argument("--jqdata-dir", type=Path, default=JQ_DIR)
    parser.add_argument("--eastmoney-dir", type=Path, default=EASTMONEY_DIR)
    parser.add_argument("--akshare-dir", type=Path, default=AKSHARE_DIR)
    parser.add_argument("--merged-dir", type=Path, default=MERGED_DIR)
    parser.add_argument("--qlib-dir", type=Path, default=QLIB_BIN_DIR)
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip replacing existing output files; missing output files may still be created.",
    )
    parser.add_argument("--smoke-fixture", action="store_true", help="Run an isolated tiny merge+Qlib conversion fixture.")
    parser.add_argument("--smoke-dir", type=Path, default=reports_root() / "smoke" / "merge_to_qlib")
    args = parser.parse_args()

    overwrite = not args.no_overwrite
    if args.smoke_fixture:
        return run_smoke_fixture(args.smoke_dir, overwrite=overwrite)

    extra_sources = [
        (name, directory)
        for name, directory in [("eastmoney", args.eastmoney_dir), ("akshare", args.akshare_dir)]
        if directory.exists()
    ]
    merged_count = build_merged_parquet(
        args.baostock_dir,
        args.jqdata_dir,
        args.merged_dir,
        overwrite,
        extra_source_dirs=extra_sources,
    )
    if merged_count == 0:
        return 1
    return convert_to_bin(args.merged_dir, args.qlib_dir, overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
