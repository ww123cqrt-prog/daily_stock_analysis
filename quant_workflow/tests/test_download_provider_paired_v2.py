from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bars(symbol: str, source_id: str) -> pd.DataFrame:
    close = pd.Series([10.0, 10.2], dtype=float)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-04", "2026-08-05"]),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": [1000.0, 1100.0],
            "amount": close * 1000,
            "turn": [0.1, 0.1],
            "pctChg": [0.0, 2.0],
            "symbol": symbol,
            "source_id": source_id,
        }
    )


def test_provider_paired_producer_writes_incremental_bundle(tmp_path: Path, monkeypatch) -> None:
    module = load_script("download_provider_paired_v2")
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "qfq-base",
                "manifest_sha256": "a" * 64,
                "latest_trade_date": "2026-08-04",
                "symbol_count": 1,
                "all": ["SH600000"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "_fetch_pair",
        lambda *args, **kwargs: (_bars("SH600000", "eastmoney_raw"), _bars("SH600000", "eastmoney_qfq_provider")),
    )
    args = argparse.Namespace(
        universe_file=universe,
        output_dir=tmp_path / "bundle",
        base_latest_trade_date="2026-08-04",
        start="2026-08-01",
        through="2026-08-05",
        sleep=0.25,
        timeout=30.0,
        max_requests=2,
        max_consecutive_failures=3,
        resume=False,
        use_system_proxy=False,
    )

    bundle = module.build_bundle(args)

    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    index = pd.read_parquet(bundle / "BUNDLE_FILE_INDEX.parquet")
    assert manifest["status"] == "ready"
    assert manifest["bundle_kind"] == "provider_paired_raw_qfq"
    assert manifest["source"]["provider"] == "eastmoney"
    assert manifest["base_qfq_snapshot_id"] == "qfq-base"
    assert manifest["base_qfq_manifest_sha256"] == "a" * 64
    assert index.loc[0, "qfq_provider_path"] == "qfq_provider/SH600000.parquet"


def test_provider_paired_resume_refetches_corrupt_cached_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script("download_provider_paired_v2")
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "qfq-base",
                "manifest_sha256": "a" * 64,
                "latest_trade_date": "2026-08-04",
                "symbol_count": 1,
                "all": ["SH600000"],
            }
        ),
        encoding="utf-8",
    )
    calls = 0

    def fetch_pair(*args, **kwargs):
        nonlocal calls
        calls += 1
        return (
            _bars("SH600000", "eastmoney_raw"),
            _bars("SH600000", "eastmoney_qfq_provider"),
        )

    monkeypatch.setattr(module, "_fetch_pair", fetch_pair)
    args = argparse.Namespace(
        universe_file=universe,
        output_dir=tmp_path / "bundle",
        base_latest_trade_date="2026-08-04",
        start="2026-08-01",
        through="2026-08-05",
        sleep=0.25,
        timeout=30.0,
        max_requests=4,
        max_consecutive_failures=3,
        resume=False,
        use_system_proxy=False,
    )
    bundle = module.build_bundle(args)
    (bundle / "qfq_provider/SH600000.parquet").write_bytes(b"corrupt")

    args.resume = True
    module.build_bundle(args)

    assert calls == 2
    pd.read_parquet(bundle / "qfq_provider/SH600000.parquet")


def test_provider_paired_producer_accepts_symbol_without_trade_on_base_date(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_script("download_provider_paired_v2")
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "dataset_id": "ashare_daily_qfq",
                "snapshot_id": "qfq-base",
                "manifest_sha256": "a" * 64,
                "latest_trade_date": "2026-08-04",
                "symbol_count": 1,
                "all": ["SH600000"],
            }
        ),
        encoding="utf-8",
    )
    suspended = _bars("SH600000", "eastmoney_raw").iloc[:1].copy()
    suspended["date"] = pd.to_datetime(["2026-08-03"])
    qfq = suspended.copy()
    qfq["source_id"] = "eastmoney_qfq_provider"
    monkeypatch.setattr(module, "_fetch_pair", lambda *args, **kwargs: (suspended, qfq))
    args = argparse.Namespace(
        universe_file=universe,
        output_dir=tmp_path / "bundle",
        base_latest_trade_date="2026-08-04",
        start="2026-08-01",
        through="2026-08-05",
        sleep=0.25,
        timeout=30.0,
        max_requests=2,
        max_consecutive_failures=3,
        resume=False,
        use_system_proxy=False,
    )

    bundle = module.build_bundle(args)

    manifest = json.loads((bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ready"
