"""The strategy probe diagnoses frozen inputs; it never repairs or publishes."""

import hashlib
import importlib
import json
import shutil
import socket
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_qmt_data import quant_data_contract as contract


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS)) if str(SCRIPTS) not in sys.path else None


def checker():
    return importlib.import_module("check_strategy_data")


def publish(root, *, snapshot="fixture", volume=1000., cash=False, symbols=4, market="ashare", negative=False):
    dates = pd.bdate_range("2024-01-02", periods=65)
    raw = root / "staging" / (snapshot + "-raw")
    qfq = root / "staging" / (snapshot + "-qfq")
    (raw / "bars").mkdir(parents=True)
    (qfq / "bars").mkdir(parents=True)
    (qfq / "factors").mkdir()
    pd.DataFrame({"date": dates}).to_parquet(qfq / "CALENDAR.parquet", index=False)
    calendar_sha = hashlib.sha256((qfq / "CALENDAR.parquet").read_bytes()).hexdigest()
    for i in range(symbols):
        symbol = f"SH{(510000 if market == 'etf' else 600000) + i:06d}"
        close = 10 + np.arange(len(dates)) * (.01 + i * .005) + np.sin(np.arange(len(dates)) / (3 + i))
        frame = pd.DataFrame({"symbol": symbol, "date": dates, "open": close - .05,
                              "high": close + .1, "low": close - .1, "close": close,
                              "volume": volume, "amount": close * 1000.})
        frame.to_parquet(raw / "bars" / (symbol + ".parquet"), index=False)
        adjusted = frame.copy()
        factor = np.ones(len(frame))
        offset = -np.linspace(2. * (i + 1), 0., len(frame)) if cash else np.zeros(len(frame))
        if negative:
            offset = -np.linspace(25., 0., len(frame))
        for field in ("open", "high", "low", "close"):
            adjusted[field] += offset
        adjusted.to_parquet(qfq / "bars" / (symbol + ".parquet"), index=False)
        pd.DataFrame({"date": dates, "factor": factor, "offset": offset}).to_parquet(
            qfq / "factors" / (symbol + ".parquet"), index=False)
    pd.DataFrame(columns=["symbol", "ex_date", "effective_trade_date", "action_type"]).to_parquet(
        qfq / "corporate_actions.parquet", index=False)
    common = dict(role="canonical", status="ready", source_retrieved_at="2024-04-01T16:00:00+08:00",
                  source={"provider": "fixture", "volume_unit": "shares"},
                  producer={"script": "test_check_strategy_data", "version": "1"})
    contract.finalize_price_snapshot(
        raw, dataset_id="ashare_daily_raw", snapshot_id=raw.name,
        price_basis="raw", adjustment={"method": "none"}, **common)
    common["source"]["trading_calendar"] = {
        "relative_path": "CALENDAR.parquet", "sha256": calendar_sha,
        "start": str(dates[0].date()), "end": str(dates[-1].date()), "origin": "fixture"}
    if market == "etf":
        (qfq / "bars").rename(qfq / "qfq")
        shutil.copytree(raw / "bars", qfq / "raw")
        pd.DataFrame({"symbol": [f"SH{510000+i}" for i in range(symbols)]}).to_parquet(
            qfq / "ETF_UNIVERSE.parquet", index=False)
        common["source"]["volume_unit"] = "units"
        contract.finalize_paired_price_snapshot(
            qfq, dataset_id="cn_etf_daily", snapshot_id=qfq.name, **common,
            adjustment={"method": "provider_native_paired", "adjusted_at": common["source_retrieved_at"],
                        "adjustment_as_of": str(dates[-1].date()),
                        "adjusted_fields": ["open", "high", "low", "close"], "volume_adjusted": False,
                        "normalization_rule": "latest_valid_trade_date_factor_equals_1_and_offset_equals_0",
                        "algorithm_version": "provider-native-affine-paired-v1", "paired_raw_component_path": "raw"})
        contract.write_root_manifest(root, {})
        contract.publish_snapshot_group([(qfq, root / "parquet/etf_1d/cn_etf_daily")],
                                        quant_data_root=root, update_root_catalog=True)
        return
    if negative:
        common["source"].update(provider="miniqmt", qfq_event_rule="cash-share-affine-front-v1")
    contract.finalize_price_snapshot(
        qfq, dataset_id="ashare_daily_qfq", snapshot_id=qfq.name, price_basis="qfq", **common,
        paired_raw_snapshot=raw if negative else None,
        upstream_snapshots=[{"dataset_id": "ashare_daily_raw", "snapshot_id": raw.name,
                             "content_sha256": json.loads((raw / "DATASET_MANIFEST.json").read_bytes())["content_sha256"]}],
        adjustment={"method": "provider_native_paired", "adjusted_at": common["source_retrieved_at"],
                    "adjustment_as_of": str(dates[-1].date()),
                    "adjusted_fields": ["open", "high", "low", "close"],
                    "normalization_rule": "latest_valid_trade_date_factor_equals_1_and_offset_equals_0",
                    "algorithm_version": "provider-native-affine-paired-v1",
                    "paired_raw_snapshot_id": raw.name})
    if not (root / "DATA_MANIFEST.json").exists():
        contract.write_root_manifest(root, {})
    contract.publish_snapshot_group(
        [(raw, root / "parquet/ashare_1d/ashare_daily_raw"),
         (qfq, root / "parquet/ashare_1d/ashare_daily_qfq")],
        quant_data_root=root, update_root_catalog=True)


def run(tmp_path, left, right, name="check", **kwargs):
    return checker().audit_strategy_data(
        left, right, output_dir=tmp_path / name, start="2024-01-02", end="2024-04-01",
        batches=1, sample_size=4, sessions=20, seed=41, **kwargs)


def test_real_snapshots_offline_deterministic_replay_and_no_writes(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for root in (left, right) for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("network is forbidden"))
    first = run(tmp_path, left, right)
    assert first["status"] == "completed"
    assert first["batches"][0]["paired_symbols"] == 4
    assert first["batches"][0]["signal_changed_days"] == 0
    assert first["batches"][0]["holding_changed_days"] == 0
    assert first["batches"][0]["return_delta_pp"] == 0
    assert first["agreed_spotcheck_rows"] > 0
    second = run(tmp_path, left, right, "replay", plan_path=tmp_path / "check/plan.json")
    assert second["plan_sha256"] == first["plan_sha256"]
    assert second["batches"] == first["batches"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert (tmp_path / "check/REPORT.md").is_file()


def test_volume_difference_is_a_lead_even_if_strategy_agrees(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left, volume=10.)
    publish(right)
    result = run(tmp_path, left, right)
    findings = pd.read_csv(tmp_path / "check/findings.csv")
    assert ((findings.rule == "cross_source_difference") & (findings.field == "volume")).any()
    assert (findings.rule == "average_price_outside_range").any()
    assert not (findings.classification == "confirmed_error").any()
    assert result["batches"][0]["signal_changed_days"] == 0


def test_shared_bad_volume_still_yields_independent_check_queue(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left, volume=10.)
    publish(right, volume=10.)
    run(tmp_path, left, right)
    findings = pd.read_csv(tmp_path / "check/findings.csv")
    assert (findings.rule == "average_price_outside_range").any()
    assert not (findings.rule == "cross_source_difference").any()
    assert pd.read_csv(tmp_path / "check/agreed_spotchecks.csv").shape[0] > 0


def test_convention_difference_uses_same_raw_execution_without_declaring_wrong_prices(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right, cash=True)
    result = run(tmp_path, left, right)
    batch = result["batches"][0]
    assert batch["signal_changed_days"] > 0
    assert batch["execution_input_sha256"]["primary"] == batch["execution_input_sha256"]["reference"]
    findings = pd.read_csv(tmp_path / "check/findings.csv")
    assert (findings.rule == "qfq_convention_or_value_difference").any()
    assert not (findings.classification == "confirmed_error").any()


def test_missing_reference_symbol_is_not_replaced_and_healthy_pairs_continue(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right, symbols=3)
    result = run(tmp_path, left, right)
    batch = result["batches"][0]
    assert len(batch["requested_symbols"]) == 4
    assert batch["paired_symbols"] == 3
    assert batch["unavailable_symbols"] == ["SH600003"]
    assert result["status"] == "partial"


def test_replay_keeps_old_identity_after_a_new_publication(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    first = run(tmp_path, left, right)
    publish(left, snapshot="later", volume=10.)
    second = run(tmp_path, left, right, "replay", plan_path=tmp_path / "check/plan.json")
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["batches"] == second["batches"]


def test_output_inside_input_and_overwriting_a_report_are_refused(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    with pytest.raises(ValueError, match="outside"):
        checker().audit_strategy_data(left, right, output_dir=left / "report", start="2024-01-02", end="2024-04-01")
    run(tmp_path, left, right)
    with pytest.raises((ValueError, FileExistsError)):
        run(tmp_path, left, right)


def test_evidence_hash_and_units_are_checked_without_automatically_repairing(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    run(tmp_path, left, right)
    spot = pd.read_csv(tmp_path / "check/agreed_spotchecks.csv").iloc[0]
    body = tmp_path / "provider.json"
    body.write_text('{"verified_manually": true}', encoding="utf-8")
    evidence = tmp_path / "evidence.json"
    record = {"symbol": spot.symbol, "date": spot.date, "field": spot.field,
              "basis": "raw", "value": float(spot.value), "unit": spot.unit,
              "source_url": "https://example.org/original", "evidence_path": body.name,
              "evidence_sha256": hashlib.sha256(body.read_bytes()).hexdigest()}
    evidence.write_text(json.dumps({"observations": [record]}), encoding="utf-8")
    result = run(tmp_path, left, right, "evidence-check", evidence_path=evidence,
                 plan_path=tmp_path / "check/plan.json")
    assert result["external_observations_checked"] == 1
    body.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        run(tmp_path, left, right, "bad-evidence", evidence_path=evidence,
            plan_path=tmp_path / "check/plan.json")


def test_cli_dispatches_the_real_checker_without_update(tmp_path, monkeypatch):
    cli = importlib.import_module("quant_data_cli")
    called = []
    monkeypatch.setattr(checker(), "main", lambda args: called.append(args) or 0)
    assert cli.main(["--root", str(tmp_path / "primary"), "check-strategy",
                     "--reference-root", str(tmp_path / "reference"),
                     "--start", "2024-01-02", "--end", "2024-04-01",
                     "--output-dir", str(tmp_path / "report")]) == 0
    assert called and "--reference-root" in called[0]


def test_etf_paired_snapshot_flow_retains_base_units(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left, market="etf")
    publish(right, market="etf")
    result = run(tmp_path, left, right, market="etf")
    assert result["status"] == "completed"
    assert result["batches"][0]["return_delta_pp"] == 0
    spots = pd.read_csv(tmp_path / "check/agreed_spotchecks.csv")
    assert set(spots.loc[spots.field == "volume", "unit"]) == {"units"}


def test_cash_adjusted_nonpositive_prices_are_not_declared_invalid_raw(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left, negative=True)
    publish(right, negative=True)
    result = run(tmp_path, left, right)
    assert result["status"] == "completed"
    findings = pd.read_csv(tmp_path / "check/findings.csv")
    assert not (findings.classification == "confirmed_error").any()


def test_original_input_tampering_during_probe_cannot_complete(tmp_path, monkeypatch):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    original = checker()._probe
    changed = False

    def probe(*args):
        nonlocal changed
        result = original(*args)
        if not changed:
            file = next((left / "parquet/ashare_1d/ashare_daily_raw/snapshots").rglob("bars/SH600000.parquet"))
            file.write_bytes(file.read_bytes() + b"changed")
            changed = True
        return result

    monkeypatch.setattr(checker(), "_probe", probe)
    with pytest.raises(ValueError, match="changed"):
        run(tmp_path, left, right)
    assert not (tmp_path / "check/summary.json").exists()


def test_explicit_calendar_is_bound_and_replay_does_not_reselect_dates(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    publish(left)
    publish(right)
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=65)}).to_parquet(calendar, index=False)
    first = run(tmp_path, left, right, calendar_path=calendar)
    plan = json.loads((tmp_path / "check/plan.json").read_bytes())
    assert plan["calendar_source"]["sha256"] == hashlib.sha256(calendar.read_bytes()).hexdigest()
    second = run(tmp_path, left, right, "replay", plan_path=tmp_path / "check/plan.json")
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["batches"] == second["batches"]


def test_shared_nullable_unknowns_are_not_agreed_facts_or_confirmed_errors():
    index = pd.date_range("2024-01-02", periods=2)
    frame = pd.DataFrame({"volume": pd.Series([1., None], dtype="Float64").array,
                          "amount": pd.Series([None, None], dtype="Float64").array}, index=index)
    findings = []
    checker()._compare_frames(frame, frame.copy(), findings, "batch_1", "SH600000",
                              side="primary", basis="raw/qfq", paired=True)
    assert len(findings) == 3
    assert {row["classification"] for row in findings} == {"insufficient_evidence"}
    assert {row["rule"] for row in findings} == {"missing_numeric_evidence"}
