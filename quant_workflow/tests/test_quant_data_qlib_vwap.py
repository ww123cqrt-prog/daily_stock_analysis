"""VWAP uses the exact frozen adjustment evidence, including cash offsets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_qmt_data import quant_data_contract as contract, read_bars
from quant_workflow import quant_data_qlib_provider as direct
from quant_workflow.tests.test_quant_data_qlib_provider import (
    _publish_index,
    _publish_membership,
    _symlink_or_skip,
    direct_fixture,  # noqa: F401
)


def _publish_price_pair(
    root: Path, *, cash: bool, name: str = "vwap", first_offset: float = -2.0,
) -> tuple[Path, Path]:
    raw_snapshot = root / "staging" / name / f"{name}-raw"
    qfq_snapshot = root / "staging" / name / f"{name}-qfq"
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    factors = pd.DataFrame({"date": dates, "factor": [0.5, 0.75, 1.0]})
    if cash:
        factors["offset"] = [first_offset, -1.0, 0.0]
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": [9.0, 11.0, 13.0],
            "high": [12.0, 14.0, 16.0],
            "low": [8.0, 10.0, 12.0],
            "close": [10.0, 12.0, 14.0],
            "volume": [1200.0 if cash else 12.0] * 3,
            "amount": [13200.0, 15600.0, 16800.0],
            "source_id": "fixture.raw",
        }
    )
    qfq = raw.copy()
    for field in ("open", "high", "low", "close"):
        qfq[field] = raw[field] * factors["factor"] + factors.get("offset", 0.0)
    # A vendor bar column is not the identity-bound factors file.
    qfq["factor"] = 42.0
    for snapshot, frame in ((raw_snapshot, raw), (qfq_snapshot, qfq)):
        (snapshot / "bars").mkdir(parents=True)
        frame.to_parquet(snapshot / "bars/SH600000.parquet", index=False)
    (qfq_snapshot / "factors").mkdir()
    factors.to_parquet(qfq_snapshot / "factors/SH600000.parquet", index=False)
    pd.DataFrame(columns=["symbol", "event_date"]).to_parquet(
        qfq_snapshot / "corporate_actions.parquet", index=False
    )
    source = {"provider": "miniqmt" if cash else "baostock", "volume_unit": "shares" if cash else "lots"}
    if cash:
        source["qfq_event_rule"] = "cash-share-affine-front-v1"
    common = dict(
        role="canonical", status="ready", source=source,
        producer={"script": "test_quant_data_qlib_vwap.py", "version": "1"},
        source_retrieved_at="2026-01-07T08:00:00+08:00",
    )
    contract.finalize_price_snapshot(
        raw_snapshot, dataset_id="ashare_daily_raw", snapshot_id=raw_snapshot.name,
        price_basis="raw", adjustment={"method": "none"}, **common,
    )
    contract.finalize_price_snapshot(
        qfq_snapshot, dataset_id="ashare_daily_qfq", snapshot_id=qfq_snapshot.name,
        price_basis="qfq",
        adjustment={
            "method": "provider_native_paired" if cash else "local_factor",
            "algorithm_version": "provider-native-affine-paired-v1" if cash else "fixture-ratio-v1",
            "adjusted_at": "2026-01-07T08:00:00+08:00",
            "adjustment_as_of": "2026-01-06",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": "latest_valid_trade_date_factor_equals_1_and_offset_equals_0",
            "volume_adjusted": False,
            "paired_raw_snapshot_id": raw_snapshot.name,
        },
        upstream_snapshots=[{
            "dataset_id": "ashare_daily_raw", "snapshot_id": raw_snapshot.name,
            "content_sha256": json.loads((raw_snapshot / "DATASET_MANIFEST.json").read_text())["content_sha256"],
        }],
        paired_raw_snapshot=raw_snapshot, **common,
    )
    if not (root / "DATA_MANIFEST.json").exists():
        contract.write_root_manifest(root, {})
    return tuple(contract.publish_snapshot_group(
        [(raw_snapshot, root / "parquet/ashare_1d/ashare_daily_raw"),
         (qfq_snapshot, root / "parquet/ashare_1d/ashare_daily_qfq")],
        quant_data_root=root, update_root_catalog=True,
    ))


def _provider(
    root: Path, *, cash: bool, first_offset: float = -2.0,
) -> tuple[direct.QuantDataFeatureProvider, Path]:
    _, snapshot = _publish_price_pair(root, cash=cash, first_offset=first_offset)
    _publish_index(root)
    _publish_membership(root)
    descriptor = direct.build_direct_provider_descriptor(quant_data_root=root)
    return direct.QuantDataFeatureProvider(descriptor_path=descriptor, quant_data_root=root), snapshot


@pytest.mark.parametrize(
    "cash,want_vwap,want_close,want_ratio",
    [(False, [5.5, 9.75, 14.0], [5.0, 9.0, 14.0], [1.1, 13.0 / 12.0, 1.0]),
     (True, [3.5, 8.75, 14.0], [3.0, 8.0, 14.0], [7.0 / 6.0, 35.0 / 32.0, 1.0])],
    ids=["multiplicative-lots", "affine-cash-shares"],
)
def test_vwap_uses_the_frozen_daily_price_basis(tmp_path, cash, want_vwap, want_close, want_ratio):
    provider, _ = _provider(tmp_path / "data", cash=cash)
    vwap = provider.feature("SH600000", "$vwap", 0, 2, "day")
    close = provider.feature("SH600000", "$close", 0, 2, "day")
    np.testing.assert_allclose(vwap, want_vwap)
    np.testing.assert_allclose(close, want_close)
    np.testing.assert_allclose(vwap / close, want_ratio, rtol=1e-6)
    assert provider.feature("SH600000", "$amount", 0, 2, "day").tolist() == [13200.0, 15600.0, 16800.0]
    assert provider.feature("SH600000", "$volume", 0, 2, "day").tolist() == [1200.0] * 3
    assert provider.feature("SH600000", "$factor", 0, 2, "day").tolist() == [1.0] * 3
    low = provider.feature("SH600000", "$low", 0, 2, "day")
    high = provider.feature("SH600000", "$high", 0, 2, "day")
    assert ((low <= vwap) & (vwap <= high)).all()
    raw = read_bars("ashare_daily_raw", "SH600000", quant_data_root=tmp_path / "data")
    assert raw["close"].tolist() == [10.0, 12.0, 14.0]
    assert raw["amount"].tolist() == [13200.0, 15600.0, 16800.0]


def test_vwap_keeps_the_unadjusted_index_basis(direct_fixture):
    root, _, descriptor = direct_fixture
    provider = direct.QuantDataFeatureProvider(descriptor_path=descriptor, quant_data_root=root)
    assert provider.feature("SH000300", "$vwap", 0, 2, "day").tolist() == [100.0, 101.0, 102.0]


@pytest.mark.parametrize("offset,want_vwap,want_close", [(-5.5, 0.0, -0.5), (-12.0, -6.5, -7.0)])
def test_vwap_preserves_cash_front_nonpositive_values(tmp_path, offset, want_vwap, want_close):
    provider, _ = _provider(tmp_path / "data", cash=True, first_offset=offset)
    assert provider.feature("SH600000", "$vwap", 0, 0, "day").iloc[0] == pytest.approx(want_vwap)
    assert provider.feature("SH600000", "$close", 0, 0, "day").iloc[0] == pytest.approx(want_close)


def test_legacy_descriptor_keeps_ohlc_readable_but_requires_vwap_evidence(tmp_path):
    root = tmp_path / "data"
    provider, snapshot = _provider(root, cash=False)
    descriptor = provider._context.descriptor_path
    original_manifest = (snapshot / "DATASET_MANIFEST.json").read_bytes()
    payload = json.loads(descriptor.read_bytes())
    del payload["dataset"]["factor_index_sha256"]
    legacy_dir = descriptor.parent.with_name("canonical")
    legacy_dir.mkdir()
    for name in ("CALENDAR.parquet", "MEMBERSHIP_INTERVALS.parquet"):
        (legacy_dir / name).write_bytes((descriptor.parent / name).read_bytes())
    legacy_descriptor = legacy_dir / direct.DESCRIPTOR_NAME
    legacy_descriptor.write_text(json.dumps(payload), encoding="utf-8")
    legacy_bytes = legacy_descriptor.read_bytes()
    legacy = direct.QuantDataFeatureProvider(descriptor_path=legacy_descriptor, quant_data_root=root)
    assert legacy.feature("SH600000", "$close", 0, 2, "day").tolist() == [5.0, 9.0, 14.0]
    with pytest.raises(direct.DirectProviderError, match="rebuild.*descriptor"):
        legacy.feature("SH600000", "$vwap", 0, 2, "day")
    rebuilt = direct.build_direct_provider_descriptor(quant_data_root=root)
    assert rebuilt != legacy_descriptor
    assert legacy_descriptor.read_bytes() == legacy_bytes
    assert (snapshot / "DATASET_MANIFEST.json").read_bytes() == original_manifest


def test_vwap_does_not_repeat_full_snapshot_validation(tmp_path, monkeypatch):
    from quant_qmt_data import quant_data_contract, quant_data_reader

    provider, _ = _provider(tmp_path / "data", cash=False)
    provider.feature("SH600000", "$close", 0, 2, "day")

    def refuse_snapshot_scan(*args, **kwargs):
        pytest.fail("VWAP attempted full snapshot content validation")

    monkeypatch.setattr(quant_data_contract, "_content_hash", refuse_snapshot_scan)
    monkeypatch.setattr(quant_data_reader, "_content_hash", refuse_snapshot_scan)
    assert provider.feature("SH600000", "$vwap", 0, 2, "day").tolist() == [5.5, 9.75, 14.0]


@pytest.mark.parametrize("fault", ["missing", "factor", "offset", "factor_and_index", "symlink"])
def test_vwap_rejects_changed_factor_evidence_after_bar_cache_warmup(tmp_path, fault):
    provider, snapshot = _provider(tmp_path / "data", cash=False)
    provider.feature("SH600000", "$close", 0, 2, "day")
    path = snapshot / "factors/SH600000.parquet"
    if fault == "missing":
        path.unlink()
    elif fault == "symlink":
        target = tmp_path / "factor-copy.parquet"
        target.write_bytes(path.read_bytes())
        path.unlink()
        _symlink_or_skip(path, target)
    else:
        frame = pd.read_parquet(path)
        if fault == "offset":
            frame["offset"] = [-1.0, 0.0, 0.0]
        else:
            frame.loc[0, "factor"] = 0.9
        frame.to_parquet(path, index=False)
        if fault == "factor_and_index":
            index_path = snapshot / "FACTOR_INDEX.parquet"
            index = pd.read_parquet(index_path)
            index.loc[0, "sha256"] = contract.sha256_file(path)
            index.to_parquet(index_path, index=False)
    with pytest.raises(direct.DirectProviderError):
        provider.feature("SH600000", "$vwap", 0, 2, "day")
