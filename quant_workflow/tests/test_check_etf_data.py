from __future__ import annotations

import hashlib
import importlib.util
import json
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def checker():
    path = SCRIPTS / "check_etf_data.py"
    assert path.is_file(), "the independent ETF healthcheck entry is missing"
    sys.path.insert(0, str(SCRIPTS)) if str(SCRIPTS) not in sys.path else None
    spec = importlib.util.spec_from_file_location("check_etf_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def frames(symbol="SH510010"):
    raw = pd.DataFrame(
        {
            "symbol": [symbol, symbol],
            "date": pd.to_datetime(["2026-10-15", "2026-10-16"]),
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [100.0, 200.0],
            "amount": [110000.0, 240000.0],
            "source_id": ["provider:receipt-a", "provider:receipt-b"],
        }
    )
    factor = pd.DataFrame(
        {"date": raw.date, "factor": [1.0, 1.0], "offset": [0.0, 0.0]}
    )
    return raw, raw.copy(), factor


def frozen_fixture(root: Path, symbols=("SH510010",)):
    snapshot = root / "parquet/etf_1d/cn_etf_daily/snapshots/test-snapshot"
    indexes = {name: [] for name in ("raw", "qfq", "factors")}
    for symbol in symbols:
        for name, frame in zip(indexes, frames(symbol), strict=True):
            path = snapshot / name / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            indexes[name].append(
                {
                    "symbol": symbol,
                    "relative_path": f"{name}/{symbol}.parquet",
                    "row_count": 2,
                    "first_date": "2026-10-15",
                    "last_date": "2026-10-16",
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
    names = {
        "raw": "RAW_FILE_INDEX.parquet",
        "qfq": "QFQ_FILE_INDEX.parquet",
        "factors": "FACTOR_INDEX.parquet",
    }
    for component, rows in indexes.items():
        pd.DataFrame(rows).to_parquet(snapshot / names[component], index=False)
    manifest = {
        "schema_version": 2,
        "dataset_id": "cn_etf_daily",
        "snapshot_id": "test-snapshot",
        "symbol_count": len(symbols),
        "row_count": 2 * len(symbols),
        "latest_trade_date": "2026-10-16",
        "file_index_path": names["qfq"],
        "raw_file_index_path": names["raw"],
        "factor_index_path": names["factors"],
        "adjustment": {"volume_adjusted": False},
        "source": {"provider": "synthetic"},
        "components": {
            k: {
                "file_index_path": v,
                "row_count": 2 * len(symbols),
                "symbol_count": len(symbols),
            }
            for k, v in names.items()
        },
    }
    body = json.dumps(manifest, sort_keys=True).encode()
    (snapshot / "DATASET_MANIFEST.json").write_bytes(body)
    pointer = {
        "dataset_id": "cn_etf_daily",
        "snapshot_id": "test-snapshot",
        "snapshot_path": str(snapshot.relative_to(root)),
        "manifest_sha256": hashlib.sha256(body).hexdigest(),
    }
    (snapshot.parent.parent / "LATEST.json").write_text(json.dumps(pointer))
    (root / "DATA_MANIFEST.json").write_text(
        json.dumps({"datasets": {"cn_etf_daily": pointer}})
    )
    return snapshot


def digest_tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


def test_valid_affine_prices_are_not_forced_into_multiplicative_rule():
    mod = checker()
    raw, qfq, factor = frames()
    factor.loc[0, ["factor", "offset"]] = [0.5, -0.25]
    qfq.loc[0, ["open", "high", "low", "close"]] = [4.75, 5.75, 4.25, 5.25]
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert not [i for i in issues if i["severity"] == "confirmed_error"]


@pytest.mark.parametrize(
    "component,field,value,rule",
    [
        ("raw", "high", 8.0, "ohlc_relationship"),
        ("raw", "volume", -1.0, "invalid_numeric"),
        ("qfq", "close", np.nan, "invalid_numeric"),
        ("qfq", "amount", -1.0, "invalid_numeric"),
        ("factors", "factor", 0.0, "invalid_factor"),
    ],
)
def test_bad_values_have_exact_symbol_date_and_field(component, field, value, rule):
    mod = checker()
    data = dict(zip(("raw", "qfq", "factors"), frames(), strict=True))
    data[component].loc[0, field] = value
    issues = mod.inspect_etf_frames("SH510010", **data)
    assert any(
        i["severity"] == "confirmed_error"
        and i["rule"] == rule
        and i["symbol"] == "SH510010"
        and i["date"] == "2026-10-15"
        and i["field"] == field
        for i in issues
    )


def test_missing_amount_is_evidence_gap_not_fabricated_error_or_zero():
    mod = checker()
    raw, qfq, factor = frames()
    raw["amount"] = np.nan
    qfq["amount"] = np.nan
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert any(
        i["severity"] == "insufficient_evidence" and i["field"] == "amount"
        for i in issues
    )
    assert not any(i["severity"] == "confirmed_error" for i in issues)
    assert raw.amount.isna().all() and qfq.amount.isna().all()


@pytest.mark.parametrize("field", ["volume", "amount"])
def test_qfq_cannot_change_unadjusted_fields(field):
    mod = checker()
    raw, qfq, factor = frames()
    qfq.loc[0, field] += 1
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert any(
        i["rule"] == "raw_qfq_unadjusted_field"
        and i["field"] == field
        and i["date"] == "2026-10-15"
        for i in issues
    )


def test_duplicate_keys_and_missing_pair_dates_are_reported_without_cleaning():
    mod = checker()
    raw, qfq, factor = frames()
    qfq = pd.concat([qfq.iloc[:1], qfq.iloc[:1]], ignore_index=True)
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert {"duplicate_key", "pair_key_mismatch"}.issubset({i["rule"] for i in issues})
    assert len(qfq) == 2


def test_auxiliary_check_requires_declared_same_basis_semantics():
    mod = checker()
    raw, qfq, factor = frames()
    qfq["preclose"] = [np.nan, 10.0]
    unknown = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert not any(i["rule"] == "auxiliary_mismatch" for i in unknown)
    declared = mod.inspect_etf_frames(
        "SH510010", raw, qfq, factor, auxiliary_semantics="same_basis_previous_close"
    )
    assert any(
        i["rule"] == "auxiliary_mismatch" and i["date"] == "2026-10-16"
        for i in declared
    )


def test_fitted_factor_is_not_independent_company_action_proof():
    mod = checker()
    raw, qfq, factor = frames()
    factor["factor_source"] = "derived_same_provider_raw_qfq_piecewise_affine_ohlc"
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    assert any(
        i["rule"] == "factor_authority_unproven"
        and i["severity"] == "insufficient_evidence"
        for i in issues
    )
    assert not any(i["rule"] == "qfq_transform" for i in issues)


def test_real_snapshot_readonly_and_one_decode_per_object(tmp_path, monkeypatch):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root, ("SH510010", "SZ159915"))
    before = digest_tree(root)
    read = pd.read_parquet
    calls = []

    def tracked_read(path, *args, **kwargs):
        calls.append(path)
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", tracked_read)
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **kw: pytest.fail("offline check used network"),
    )
    report = mod.audit_etf_snapshot(root, output_dir=tmp_path / "report")
    assert report["status"] == "completed"
    assert report["symbol_count"] == 2
    assert report["counts"]["confirmed_error"] == 0
    assert report["io"]["parquet_decodes"] == 9
    assert len(calls) == 9
    assert digest_tree(root) == before
    assert (tmp_path / "report/findings.csv").is_file()
    assert (
        json.loads((tmp_path / "report/summary.json").read_text())["snapshot_id"]
        == snapshot.name
    )


def test_tampered_symbol_does_not_prevent_healthy_symbol_audit(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root, ("SH510010", "SZ159915"))
    p = snapshot / "raw/SH510010.parquet"
    df = pd.read_parquet(p)
    df.loc[0, "close"] += 0.1
    df.to_parquet(p, index=False)
    report = mod.audit_etf_snapshot(root, output_dir=tmp_path / "report")
    assert report["counts"]["confirmed_error"] >= 1
    assert report["symbols_scanned"] == 2
    issues = pd.read_csv(tmp_path / "report/findings.csv")
    assert ((issues.rule == "file_identity") & (issues.symbol == "SH510010")).any()
    assert "SZ159915" in pd.read_csv(tmp_path / "report/symbols.csv").symbol.tolist()


def test_report_cannot_be_written_inside_data_root(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    frozen_fixture(root)
    before = digest_tree(root)
    with pytest.raises(ValueError, match="outside"):
        mod.audit_etf_snapshot(root, output_dir=root / "report")
    assert digest_tree(root) == before


def test_input_replaced_during_check_is_not_reported_as_stable(tmp_path, monkeypatch):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root)
    original = pd.read_parquet
    switched = False

    def replace_after_read(path, *args, **kwargs):
        nonlocal switched
        result = original(path, *args, **kwargs)
        if not switched:
            switched = True
            target = snapshot / "DATASET_MANIFEST.json"
            target.write_bytes(target.read_bytes() + b" ")
        return result

    monkeypatch.setattr(pd, "read_parquet", replace_after_read)
    report = mod.audit_etf_snapshot(root, output_dir=tmp_path / "report")
    assert report["status"] == "incomplete"
    assert report["input_unchanged"] is False


def test_closed_day_and_missing_session_have_different_classification(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    frozen_fixture(root)
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame(
        {
            "date": ["2026-10-14", "2026-10-15", "2026-10-16"],
            "is_open": [True, True, False],
        }
    ).to_csv(calendar, index=False)
    mod.audit_etf_snapshot(root, output_dir=tmp_path / "report", calendar_path=calendar)
    issues = pd.read_csv(tmp_path / "report/findings.csv")
    assert (
        (issues.rule == "closed_day_bar")
        & (issues.severity == "confirmed_error")
        & (issues.date == "2026-10-16")
    ).any()


def test_unified_cli_check_is_diagnostic_and_never_calls_publish(
    tmp_path, capsys, monkeypatch
):
    root = tmp_path / "data"
    frozen_fixture(root)
    sys.path.insert(0, str(SCRIPTS)) if str(SCRIPTS) not in sys.path else None
    import quant_data_cli as cli

    monkeypatch.setattr(
        cli,
        "publish_snapshot_group",
        lambda *a, **kw: pytest.fail("check published data"),
    )
    before = digest_tree(root)
    assert (
        cli.main(
            ["--root", str(root), "check-etf", "--output-dir", str(tmp_path / "report")]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["symbol_count"] == 1
    assert digest_tree(root) == before


def test_whole_snapshot_digest_includes_index_bytes(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root)
    from quant_data_contract import _content_hash

    manifest_path = snapshot / "DATASET_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["content_sha256"] = _content_hash(snapshot, manifest)
    manifest_path.write_text(json.dumps(manifest))
    pointer_path = snapshot.parent.parent / "LATEST.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer))
    index_path = snapshot / "RAW_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    index["unbound_extra_column"] = "changed"
    index.to_parquet(index_path, index=False)
    mod.audit_etf_snapshot(root, output_dir=tmp_path / "report")
    issues = pd.read_csv(tmp_path / "report/findings.csv")
    assert (issues.rule == "snapshot_content_identity").any()


def test_parquet_calendar_and_missing_session_stay_readonly(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root)
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-10-15", "2026-10-16"]), "is_open": [True, False]}
    ).to_parquet(calendar, index=False)
    report = mod.audit_etf_snapshot(
        root, output_dir=tmp_path / "report", calendar_path=calendar
    )
    assert report["counts"]["confirmed_error"] == 1
    assert (snapshot / "raw/SH510010.parquet").is_file()


def test_saved_external_sample_compares_exact_dates_and_all_ohlc():
    mod = checker()
    _, qfq, _ = frames()
    body = json.dumps(
        {
            "code": 0,
            "data": {
                "sh510010": {
                    "qfqday": [["2026-10-16", "11", "12.1", "13", "10", "200"]]
                }
            },
        }
    ).encode()
    issues, result = mod._reference_comparison("SH510010", qfq, body, "saved.body")
    assert result["status"] == "disagreed"
    assert len(issues) == 1
    assert issues[0]["field"] == "close" and issues[0]["date"] == "2026-10-16"
    assert issues[0]["severity"] == "needs_review"


def test_symlink_object_does_not_escape_snapshot(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    snapshot = frozen_fixture(root)
    target = snapshot / "raw/SH510010.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    mod.audit_etf_snapshot(root, output_dir=tmp_path / "report")
    issues = pd.read_csv(tmp_path / "report/findings.csv")
    assert (issues.rule == "file_identity").any()


def test_known_split_cannot_be_hidden_by_fitting_identity_factors():
    mod = checker()
    raw, qfq, factor = frames()
    events = [
        {
            "effective_trade_date": "2026-10-16",
            "ratio": 8.0,
            "source_url": "https://example.test/official.pdf",
        }
    ]
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor, split_events=events)
    matched = [i for i in issues if i["rule"] == "split_adjustment_missing"]
    assert len(matched) == 1
    assert matched[0]["observed"] == 1.0 and matched[0]["expected"] == 8.0
    assert matched[0]["severity"] == "confirmed_error"
    factor.loc[0, "factor"] = 0.125
    qfq.loc[0, ["open", "high", "low", "close"]] = [1.25, 1.5, 1.125, 1.375]
    assert not any(
        i["rule"] == "split_adjustment_missing"
        for i in mod.inspect_etf_frames(
            "SH510010", raw, qfq, factor, split_events=events
        )
    )


def test_split_evidence_bytes_are_checked_before_using_ratio(tmp_path):
    mod = checker()
    root = tmp_path / "data"
    frozen_fixture(root)
    evidence = tmp_path / "official.pdf"
    evidence.write_bytes(b"saved source bytes")
    csv_path = tmp_path / "actions.csv"
    pd.DataFrame(
        [
            {
                "symbol": "SH510010",
                "effective_trade_date": "2026-10-16",
                "ratio": 8.0,
                "action_type": "split",
                "evidence_path": "official.pdf",
                "evidence_sha256": "0" * 64,
                "source_url": "https://example.test/official.pdf",
            }
        ]
    ).to_csv(csv_path, index=False)
    mod.audit_etf_snapshot(
        root, output_dir=tmp_path / "report", split_evidence_path=csv_path
    )
    issues = pd.read_csv(tmp_path / "report/findings.csv")
    assert (issues.rule == "split_evidence_unavailable").any()
    assert not (issues.rule == "split_adjustment_missing").any()


def test_declared_share_to_rounded_lot_conversion_is_checked():
    mod = checker()
    raw, qfq, factor = frames()
    raw["native_volume"] = [10050.0, 20000.0]
    raw["volume_unit"] = "lot_100_shares_rounded"
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    matches = [i for i in issues if i["rule"] == "volume_unit_conversion"]
    assert len(matches) == 1 and matches[0]["date"] == "2026-10-15"
    assert matches[0]["observed"] == 100.0 and matches[0]["expected"] == 101.0
    raw.loc[0, "native_volume"] = 10049.0
    assert not any(
        i["rule"] == "volume_unit_conversion"
        for i in mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    )


def test_jump_report_keeps_price_and_return_units_distinct():
    mod = checker()
    raw, qfq, factor = frames()
    for frame in (raw, qfq):
        frame.loc[1, ["open", "high", "low", "close"]] = [18.0, 20.0, 17.0, 19.0]
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor)
    issue = next(i for i in issues if i["rule"] == "large_adjusted_return")
    assert issue["observed"] == 19.0
    assert "%" in issue["detail"]


def test_verified_split_is_not_counted_as_still_pending_jump():
    mod = checker()
    raw, qfq, factor = frames()
    for frame in (raw, qfq):
        frame.loc[1, ["open", "high", "low", "close"]] = [1.375, 1.625, 1.25, 1.5]
    events = [{"effective_trade_date": "2026-10-16", "ratio": 8.0}]
    issues = mod.inspect_etf_frames("SH510010", raw, qfq, factor, split_events=events)
    assert any(i["rule"] == "split_adjustment_missing" for i in issues)
    assert not any(i["rule"] == "large_adjusted_return" for i in issues)
