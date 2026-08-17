from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import quant_data_contract as contract


def load_module():
    path = SCRIPTS / "quant_data_increment_validation.py"
    spec = importlib.util.spec_from_file_location(
        "quant_data_increment_validation", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _bars(symbol: str, dates: list[str], closes: list[float]) -> pd.DataFrame:
    close = pd.Series(closes, dtype="float64")
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": pd.Series([1000 + index for index in range(len(dates))], dtype="int64"),
            "symbol": symbol,
            "source_id": "baostock_raw",
        }
    )


def _factor_frame(
    dates: list[str], factors: list[float], *, source_date: str = "2026-01-01"
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "source_event_date": pd.to_datetime([source_date] * len(dates)),
            "factor": factors,
            "factor_source": "test_factor",
            "factor_backfilled_before_first_event": False,
            "factor_paired_fallback": False,
            "factor_terminal_normalized": False,
            "provider_qfq_terminal_factor": float("nan"),
        }
    )


def _factor_actions(
    rows: list[tuple[str, str, float, float, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "code": symbol,
                "event_date": pd.Timestamp(event_date),
                "foreAdjustFactor": fore_factor,
                "backAdjustFactor": back_factor,
                "adjustFactor": adjust_factor,
                "symbol": symbol,
                "source_id": "baostock_query_adjust_factor",
                "action_type": "adjustment_factor_event",
            }
            for symbol, event_date, fore_factor, back_factor, adjust_factor in rows
        ]
    )


def _write_snapshot(
    root: Path,
    *,
    dataset_id: str,
    snapshot_id: str,
    price_basis: str,
    bars: dict[str, pd.DataFrame],
    factors: dict[str, pd.DataFrame] | None = None,
    upstream: list[dict[str, str]] | None = None,
    source: dict[str, object] | None = None,
    factor_changed_count: int = 0,
    corporate_actions: pd.DataFrame | None = None,
) -> Path:
    staging = root / "staging" / snapshot_id / snapshot_id
    for symbol, frame in bars.items():
        frame = frame.copy()
        if price_basis == "qfq":
            assert factors is not None
            factor_values = factors[symbol].loc[:, ["date", "factor"]].copy()
            frame = frame.merge(factor_values, on="date", how="left", validate="one_to_one")
            frame["source_id"] = "baostock_raw_x_foreAdjustFactor"
        path = staging / "bars" / f"{symbol}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    if price_basis == "qfq":
        assert factors is not None
        for symbol, frame in factors.items():
            path = staging / "factors" / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        actions = (
            corporate_actions
            if corporate_actions is not None
            else pd.DataFrame(columns=["event_date", "symbol", "action_type"])
        )
        actions.to_parquet(staging / "corporate_actions.parquet", index=False)
        adjustment = {
            "method": "local_factor",
            "adjusted_at": "2026-08-16T12:00:00+08:00",
            "adjustment_as_of": max(
                str(frame["date"].max().date()) for frame in bars.values()
            ),
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_valid_trade_date_factor_equals_1",
            "algorithm_version": "test-factor-v1",
            "factor_changed_symbols": factor_changed_count,
            "volume_adjusted": False,
        }
    else:
        pd.DataFrame(
            columns=["event_date", "symbol", "action_type"]
        ).to_parquet(staging / "corporate_actions.parquet", index=False)
        adjustment = {"method": "none"}
    contract.finalize_price_snapshot(
        staging,
        dataset_id=dataset_id,
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={
            "provider": "baostock",
            "interface": "incremental_raw_plus_complete_factor_history",
            **(source or {}),
        },
        price_basis=price_basis,
        adjustment=adjustment,
        producer={"script": "test_quant_data_increment_validation.py", "version": "1"},
        source_retrieved_at="2026-08-16T12:00:00+08:00",
        upstream_snapshots=upstream,
    )
    dataset_root = root / "parquet" / "ashare_1d" / dataset_id
    return contract.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=root,
    )


def _publish_pair(
    tmp_path: Path,
    *,
    price_basis: str = "raw",
    drift_old_history: bool = False,
    factor_changed: bool = False,
    one_symbol_without_increment: bool = False,
    declared_base_snapshot_id: str | None = None,
    bind_source_bundle: bool = False,
) -> tuple[Path, str, str]:
    root = tmp_path / "quant_data"
    dataset_id = f"ashare_daily_{price_basis}"
    dates = ["2026-08-01", "2026-08-04"]
    symbols = ("SH600000", "SZ000001")
    base_bars = {
        symbol: _bars(symbol, dates, [10.0 + offset, 10.5 + offset])
        for offset, symbol in enumerate(symbols)
    }
    base_factors = (
        {symbol: _factor_frame(dates, [1.0, 1.0]) for symbol in symbols}
        if price_basis == "qfq"
        else None
    )
    base_actions = (
        _factor_actions(
            [(symbol, "2026-01-01", 1.0, 1.0, 1.0) for symbol in symbols]
        )
        if price_basis == "qfq"
        else None
    )
    base_id = f"{dataset_id}-base"
    raw_base_id = "ashare-daily-raw-base-for-qfq"
    base_upstream = (
        [
            {
                "dataset_id": "ashare_daily_raw",
                "snapshot_id": raw_base_id,
                "content_sha256": "a" * 64,
                "price_basis": "raw",
            }
        ]
        if price_basis == "qfq"
        else None
    )
    base = _write_snapshot(
        root,
        dataset_id=dataset_id,
        snapshot_id=base_id,
        price_basis=price_basis,
        bars=base_bars,
        factors=base_factors,
        corporate_actions=base_actions,
        upstream=base_upstream,
    )

    candidate_bars: dict[str, pd.DataFrame] = {}
    candidate_factors: dict[str, pd.DataFrame] | None = (
        {} if price_basis == "qfq" else None
    )
    for offset, symbol in enumerate(symbols):
        old_closes = [10.0 + offset, 10.5 + offset]
        if drift_old_history and symbol == "SH600000":
            old_closes[0] += 1.0
        if factor_changed and symbol == "SH600000":
            old_closes = [value / 2 for value in old_closes]
        candidate_dates = (
            dates
            if one_symbol_without_increment and symbol == "SZ000001"
            else [*dates, "2026-08-05", "2026-08-06"]
        )
        candidate_closes = (
            old_closes
            if one_symbol_without_increment and symbol == "SZ000001"
            else [*old_closes, 11.0 + offset, 11.5 + offset]
        )
        candidate_bars[symbol] = _bars(symbol, candidate_dates, candidate_closes)
        if candidate_factors is not None:
            candidate_factors[symbol] = _factor_frame(
                candidate_dates,
                (
                    [0.5, 0.5, 1.0, 1.0]
                    if factor_changed and symbol == "SH600000"
                    else [1.0] * len(candidate_dates)
                ),
                source_date=("2026-08-05" if factor_changed and symbol == "SH600000" else "2026-01-01"),
            )
    candidate_id = f"{dataset_id}-candidate"
    candidate_actions = base_actions
    if price_basis == "qfq" and factor_changed:
        candidate_actions = _factor_actions(
            [
                ("SH600000", "2026-01-01", 0.5, 1.0, 1.0),
                ("SH600000", "2026-08-05", 1.0, 2.0, 2.0),
                ("SZ000001", "2026-01-01", 1.0, 1.0, 1.0),
            ]
        )
    source_bundle_sha256 = None
    if bind_source_bundle:
        probe = root / "bundle-manifest-probe.json"
        contract.atomic_write_json(
            probe,
            _bundle_manifest_payload(base_id, symbol_count=len(symbols)),
        )
        source_bundle_sha256 = contract.sha256_file(probe)
        probe.unlink()
    candidate = _write_snapshot(
        root,
        dataset_id=dataset_id,
        snapshot_id=candidate_id,
        price_basis=price_basis,
        bars=candidate_bars,
        factors=candidate_factors,
        upstream=(
            [
                {
                    "dataset_id": "ashare_daily_raw",
                    "snapshot_id": "ashare-daily-raw-candidate-for-qfq",
                    "content_sha256": "b" * 64,
                    "price_basis": "raw",
                }
            ]
            if price_basis == "qfq"
            else [
                {
                    "dataset_id": dataset_id,
                    "snapshot_id": base_id,
                    "content_sha256": json.loads(
                        (base / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
                    )["content_sha256"],
                    "price_basis": price_basis,
                }
            ]
        ),
        source={
            "base_snapshot_id": declared_base_snapshot_id
            or (raw_base_id if price_basis == "qfq" else base_id),
            "requested_start": "2026-07-29",
            "requested_through": "2026-08-06",
            **(
                {"source_bundle_sha256": source_bundle_sha256}
                if source_bundle_sha256 is not None
                else {}
            ),
        },
        factor_changed_count=1 if factor_changed else 0,
        corporate_actions=candidate_actions,
    )
    dataset_root = root / "parquet" / "ashare_1d" / dataset_id
    contract.write_root_manifest(
        root,
        {
            dataset_id: contract.catalog_entry_from_snapshot(
                candidate,
                quant_data_root=root,
                dataset_root=dataset_root,
            )
        },
    )
    return root, base_id, candidate_id


def _bundle_manifest_payload(base_id: str, *, symbol_count: int) -> dict[str, object]:
    return {
        "schema_version": 2,
        "status": "ready",
        "bundle_mode": "incremental",
        "bundle_kind": "baostock_raw_factor_qfq",
        "base_latest_trade_date": "2026-08-04",
        "requested_start": "2026-07-29",
        "requested_through": "2026-08-06",
        "factor_requested_start": "1990-01-01",
        "completed_symbols": symbol_count,
        "index_path": "BUNDLE_FILE_INDEX.parquet",
        "universe": {"snapshot_id": base_id, "symbol_count": symbol_count},
    }


def _write_bundle(root: Path, candidate_snapshot: Path, base_id: str) -> Path:
    bundle = root / "bundle"
    rows: list[dict[str, object]] = []
    for path in sorted((candidate_snapshot / "bars").glob("*.parquet")):
        symbol = path.stem
        frame = pd.read_parquet(path)
        update = frame.loc[frame["date"] >= pd.Timestamp("2026-07-29")].copy()
        update = update.drop(columns=["factor"], errors="ignore")
        update["source_id"] = "baostock_raw"
        raw_path = bundle / "raw" / f"{symbol}.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        update.to_parquet(raw_path, index=False)
        events_path = bundle / "factor_events" / f"{symbol}.parquet"
        events_path.parent.mkdir(parents=True, exist_ok=True)
        events = pd.DataFrame(
            {
                "code": [symbol],
                "dividOperateDate": pd.to_datetime(["2026-01-01"]),
                "foreAdjustFactor": [1.0],
                "backAdjustFactor": [1.0],
                "adjustFactor": [1.0],
                "symbol": [symbol],
                "source_id": ["baostock_query_adjust_factor"],
            }
        )
        events.to_parquet(events_path, index=False)
        rows.append(
            {
                "symbol": symbol,
                "raw_path": raw_path.relative_to(bundle).as_posix(),
                "factor_events_path": events_path.relative_to(bundle).as_posix(),
                "raw_rows": len(update),
                "raw_sha256": contract.sha256_file(raw_path),
                "factor_events_sha256": contract.sha256_file(events_path),
            }
        )
    index = pd.DataFrame(rows)
    index.to_parquet(bundle / "BUNDLE_FILE_INDEX.parquet", index=False)
    contract.atomic_write_json(
        bundle / "BUNDLE_MANIFEST.json",
        _bundle_manifest_payload(base_id, symbol_count=len(index)),
    )
    return bundle


def _comparison_result(
    module,
    root: Path,
    *,
    dataset_id: str,
    base_id: str,
    candidate_id: str,
) -> dict[str, object]:
    return module.compare_snapshot_increment(
        dataset_id,
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
    )


def test_compare_snapshot_increment_accepts_unchanged_raw_history(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(tmp_path)

    result = module.compare_snapshot_increment(
        "ashare_daily_raw",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
    )

    assert result["status"] == "increment_comparison_passed"
    assert result["checked_symbols"] == 2
    assert result["unchanged_history_symbols"] == 2
    assert result["factor_changed_symbols"] == []
    assert result["increment_rows"] == 4


def test_compare_snapshot_increment_rejects_raw_history_drift(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path, drift_old_history=True
    )

    with pytest.raises(module.IncrementValidationError, match="historical rows changed"):
        module.compare_snapshot_increment(
            "ashare_daily_raw",
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
        )


def test_compare_snapshot_increment_allows_proven_factor_recalculation(
    tmp_path: Path,
) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path, price_basis="qfq", factor_changed=True
    )

    result = module.compare_snapshot_increment(
        "ashare_daily_qfq",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
    )

    assert result["status"] == "increment_comparison_passed"
    assert result["factor_changed_symbols"] == ["SH600000"]
    assert result["unchanged_history_symbols"] == 1


def test_compare_snapshot_increment_rejects_qfq_with_wrong_declared_base(
    tmp_path: Path,
) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path,
        price_basis="qfq",
        declared_base_snapshot_id="another-qfq-snapshot",
    )

    with pytest.raises(
        module.IncrementValidationError,
        match="does not declare the selected base snapshot",
    ):
        module.compare_snapshot_increment(
            "ashare_daily_qfq",
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
        )


def test_compare_qfq_separates_event_changes_from_factor_metadata_drift(
    tmp_path: Path,
) -> None:
    module = load_module()
    root = tmp_path / "quant_data"
    dataset_id = "ashare_daily_qfq"
    symbols = ("SH600000", "SZ000001")
    base_dates = ["2026-08-01", "2026-08-04"]
    candidate_dates = [*base_dates, "2026-08-05", "2026-08-06"]
    base_bars = {
        symbol: _bars(symbol, base_dates, [10.0 + offset, 10.5 + offset])
        for offset, symbol in enumerate(symbols)
    }
    candidate_bars = {
        symbol: _bars(
            symbol,
            candidate_dates,
            [10.0 + offset, 10.5 + offset, 11.0 + offset, 11.5 + offset],
        )
        for offset, symbol in enumerate(symbols)
    }
    base_factors = {
        symbol: _factor_frame(base_dates, [1.0, 1.0]) for symbol in symbols
    }
    candidate_factors = {
        symbol: _factor_frame(
            candidate_dates,
            [1.0, 1.0, 1.0, 1.0],
            source_date="2026-08-05",
        )
        for symbol in symbols
    }
    base_actions = _factor_actions(
        [(symbol, "2026-01-01", 1.0, 1.0, 1.0) for symbol in symbols]
    )
    candidate_actions = pd.concat(
        [
            base_actions,
            _factor_actions(
                [("SH600000", "2026-08-05", 1.0, 1.1, 1.1)]
            ),
        ],
        ignore_index=True,
    )
    candidate_actions["event_date"] = candidate_actions["event_date"].astype(
        "datetime64[ms]"
    )
    base_id = "ashare-daily-qfq-base"
    base = _write_snapshot(
        root,
        dataset_id=dataset_id,
        snapshot_id=base_id,
        price_basis="qfq",
        bars=base_bars,
        factors=base_factors,
        corporate_actions=base_actions,
        upstream=[
            {
                "dataset_id": "ashare_daily_raw",
                "snapshot_id": "ashare-daily-raw-base-for-qfq",
                "content_sha256": "a" * 64,
                "price_basis": "raw",
            }
        ],
    )
    candidate_id = "ashare-daily-qfq-candidate"
    _write_snapshot(
        root,
        dataset_id=dataset_id,
        snapshot_id=candidate_id,
        price_basis="qfq",
        bars=candidate_bars,
        factors=candidate_factors,
        corporate_actions=candidate_actions,
        upstream=[
            {
                "dataset_id": "ashare_daily_raw",
                "snapshot_id": "ashare-daily-raw-candidate-for-qfq",
                "content_sha256": "b" * 64,
                "price_basis": "raw",
            }
        ],
        source={
            "base_snapshot_id": "ashare-daily-raw-base-for-qfq",
            "requested_start": "2026-07-29",
            "requested_through": "2026-08-06",
        },
        factor_changed_count=1,
    )
    dataset_root = root / "parquet" / "ashare_1d" / dataset_id
    candidate = dataset_root / "snapshots" / candidate_id
    contract.write_root_manifest(
        root,
        {
            dataset_id: contract.catalog_entry_from_snapshot(
                candidate,
                quant_data_root=root,
                dataset_root=dataset_root,
            )
        },
    )

    result = module.compare_snapshot_increment(
        dataset_id,
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
    )

    assert result["factor_changed_symbols"] == ["SH600000"]
    assert result["factor_value_changed_symbol_count"] == 0
    assert result["factor_metadata_changed_symbol_count"] == 2
    assert result["qfq_history_changed_symbol_count"] == 0


def test_sample_increment_is_deterministic_and_matches_bundle(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(tmp_path, bind_source_bundle=True)
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_raw/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)
    comparison = _comparison_result(
        module,
        root,
        dataset_id="ashare_daily_raw",
        base_id=base_id,
        candidate_id=candidate_id,
    )

    first = module.sample_increment_against_bundle(
        "ashare_daily_raw",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
        bundle_dir=bundle,
        sample_size=1,
        seed="daily-check",
        comparison_result=comparison,
    )
    second = module.sample_increment_against_bundle(
        "ashare_daily_raw",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
        bundle_dir=bundle,
        sample_size=1,
        seed="daily-check",
        comparison_result=comparison,
    )

    assert first["status"] == "increment_sample_passed"
    assert first["sampled_symbols"] == second["sampled_symbols"]
    assert first["sampled_symbol_count"] == 1
    assert first["unchecked_symbol_count"] == 1
    assert first["sampled_increment_rows"] == 2


def test_sample_increment_requires_completed_full_comparison(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(tmp_path, bind_source_bundle=True)
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_raw/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)

    with pytest.raises(
        module.IncrementValidationError,
        match="full increment comparison report is required",
    ):
        module.sample_increment_against_bundle(
            "ashare_daily_raw",
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
            bundle_dir=bundle,
            sample_size=1,
        )


def test_sample_increment_requires_candidate_bundle_identity(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(tmp_path)
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_raw/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)
    comparison = _comparison_result(
        module,
        root,
        dataset_id="ashare_daily_raw",
        base_id=base_id,
        candidate_id=candidate_id,
    )

    with pytest.raises(
        module.IncrementValidationError,
        match="candidate source bundle identity is missing",
    ):
        module.sample_increment_against_bundle(
            "ashare_daily_raw",
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
            bundle_dir=bundle,
            sample_size=1,
            comparison_result=comparison,
        )


def test_sample_increment_rejects_forged_comparison_when_history_drifted(
    tmp_path: Path,
) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path,
        drift_old_history=True,
        bind_source_bundle=True,
    )
    dataset_id = "ashare_daily_raw"
    dataset_root = root / "parquet/ashare_1d" / dataset_id / "snapshots"
    base_manifest = json.loads(
        (dataset_root / base_id / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    candidate_manifest = json.loads(
        (dataset_root / candidate_id / "DATASET_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    forged = {
        "schema_version": 1,
        "status": "increment_comparison_passed",
        "dataset_id": dataset_id,
        "price_basis": "raw",
        "base_snapshot": {
            "snapshot_id": base_id,
            "content_sha256": base_manifest["content_sha256"],
            "latest_trade_date": base_manifest["latest_trade_date"],
        },
        "candidate_snapshot": {
            "snapshot_id": candidate_id,
            "content_sha256": candidate_manifest["content_sha256"],
            "latest_trade_date": candidate_manifest["latest_trade_date"],
        },
        "checked_symbols": 2,
        "factor_changed_symbols": [],
        "errors": [],
    }
    candidate = dataset_root / candidate_id
    bundle = _write_bundle(root, candidate, base_id)

    with pytest.raises(module.IncrementValidationError, match="historical rows changed"):
        module.sample_increment_against_bundle(
            dataset_id,
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
            bundle_dir=bundle,
            sample_size=1,
            comparison_result=forged,
        )


def test_sample_increment_rejects_bundle_value_mismatch(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(tmp_path, bind_source_bundle=True)
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_raw/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)
    index = pd.read_parquet(bundle / "BUNDLE_FILE_INDEX.parquet")
    raw_path = bundle / str(index.iloc[0]["raw_path"])
    raw = pd.read_parquet(raw_path)
    raw.loc[raw["date"] == pd.Timestamp("2026-08-05"), "close"] += 9.0
    raw.to_parquet(raw_path, index=False)
    index.loc[0, "raw_sha256"] = contract.sha256_file(raw_path)
    index.to_parquet(bundle / "BUNDLE_FILE_INDEX.parquet", index=False)
    comparison = _comparison_result(
        module,
        root,
        dataset_id="ashare_daily_raw",
        base_id=base_id,
        candidate_id=candidate_id,
    )

    with pytest.raises(module.IncrementValidationError, match="bundle values differ"):
        module.sample_increment_against_bundle(
            "ashare_daily_raw",
            quant_data_root=root,
            base_snapshot_id=base_id,
            candidate_snapshot_id=candidate_id,
            bundle_dir=bundle,
            sample_size=2,
            seed="daily-check",
            comparison_result=comparison,
        )


def test_sample_increment_accepts_minimal_empty_bundle_file_for_no_increment(
    tmp_path: Path,
) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path,
        one_symbol_without_increment=True,
        bind_source_bundle=True,
    )
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_raw/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)
    index = pd.read_parquet(bundle / "BUNDLE_FILE_INDEX.parquet")
    row_index = index.index[index["symbol"] == "SZ000001"][0]
    raw_path = bundle / str(index.loc[row_index, "raw_path"])
    pd.DataFrame(
        columns=["date", "open", "high", "low", "close", "volume"]
    ).to_parquet(raw_path, index=False)
    index.loc[row_index, "raw_rows"] = 0
    index.loc[row_index, "raw_sha256"] = contract.sha256_file(raw_path)
    index.to_parquet(bundle / "BUNDLE_FILE_INDEX.parquet", index=False)
    comparison = _comparison_result(
        module,
        root,
        dataset_id="ashare_daily_raw",
        base_id=base_id,
        candidate_id=candidate_id,
    )

    result = module.sample_increment_against_bundle(
        "ashare_daily_raw",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
        bundle_dir=bundle,
        sample_size=2,
        seed="daily-check",
        comparison_result=comparison,
    )

    assert result["status"] == "increment_sample_passed"
    assert result["category_coverage"]["no_increment"] == ["SZ000001"]


def test_sample_increment_rebuilds_qfq_from_bundle_evidence(tmp_path: Path) -> None:
    module = load_module()
    root, base_id, candidate_id = _publish_pair(
        tmp_path,
        price_basis="qfq",
        bind_source_bundle=True,
    )
    candidate = (
        root
        / "parquet/ashare_1d/ashare_daily_qfq/snapshots"
        / candidate_id
    )
    bundle = _write_bundle(root, candidate, base_id)
    comparison = _comparison_result(
        module,
        root,
        dataset_id="ashare_daily_qfq",
        base_id=base_id,
        candidate_id=candidate_id,
    )

    result = module.sample_increment_against_bundle(
        "ashare_daily_qfq",
        quant_data_root=root,
        base_snapshot_id=base_id,
        candidate_snapshot_id=candidate_id,
        bundle_dir=bundle,
        sample_size=2,
        seed="qfq-check",
        comparison_result=comparison,
    )

    assert result["status"] == "increment_sample_passed"
    assert result["price_basis"] == "qfq"
    assert result["sampled_increment_rows"] == 4
