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
    _publish_cleaning_plan,
    _publish_index,
    _publish_membership,
    _symlink_or_skip,
    direct_fixture,  # noqa: F401
)


def _publish_price_pair(
    root: Path, *, cash: bool, name: str = "vwap", first_offset: float = -2.0,
    first_factor: float = 0.5, second_offset: float = -1.0,
) -> tuple[Path, Path]:
    raw_snapshot = root / "staging" / name / f"{name}-raw"
    qfq_snapshot = root / "staging" / name / f"{name}-qfq"
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    factors = pd.DataFrame({"date": dates, "factor": [first_factor, 0.75, 1.0]})
    if cash:
        factors["offset"] = [first_offset, second_offset, 0.0]
    raw = pd.DataFrame(
        {
            "date": dates,
            "open": [9.0, 11.0, 13.0],
            "high": [12.0, 14.0, 16.0],
            "low": [8.0, 10.0, 12.0],
            "close": [10.0, 12.0, 14.0],
            "volume": [1200.0 if cash else 12.0] * 3,
            "amount": [13200.0, 15600.0, 16800.0],
            "pctChg": [0.0, 0.0, 0.0],
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
    first_factor: float = 0.5, second_offset: float = -1.0,
) -> tuple[direct.QuantDataFeatureProvider, Path]:
    _, snapshot = _publish_price_pair(
        root, cash=cash, first_offset=first_offset,
        first_factor=first_factor, second_offset=second_offset,
    )
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
    assert provider.feature("SH600000", "$volume", 0, 2, "day").tolist() == (
        [1200.0] * 3 if cash else [2400.0, 1600.0, 1200.0]
    )
    assert provider.feature("SH600000", "$factor", 0, 2, "day").tolist() == (
        [1.0] * 3 if cash else [0.5, 0.75, 1.0]
    )
    assert provider._load_frame("SH600000")["volume"].tolist() == [1200.0] * 3
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
    assert provider.feature("SH000300", "$factor", 0, 2, "day").tolist() == [1.0] * 3
    assert provider.feature("SH000300", "$volume", 0, 2, "day").tolist() == [1000.0, 1100.0, 1200.0]


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
    assert provider.feature("SH600000", "$factor", 0, 2, "day").tolist() == [0.5, 0.75, 1.0]
    assert provider.feature("SH600000", "$volume", 0, 2, "day").tolist() == [2400.0, 1600.0, 1200.0]


def _exchange(root: Path, *, first_factor: float, volume_threshold=None):
    from qlib.backtest.exchange import Exchange

    provider, _ = _provider(root, cash=False, first_factor=first_factor)
    direct.init_qlib_from_quant_data(
        descriptor_path=provider._context.descriptor_path, quant_data_root=root, kernels=1,
        exp_manager={
            "class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": (root.parent / "unused-mlruns").as_uri(), "default_exp_name": "Experiment"},
        },
    )
    day = pd.Timestamp("2026-01-02")
    return Exchange(
        start_time=day, end_time=day, codes=["SH600000"], deal_price="close",
        limit_threshold=0.095, open_cost=0, close_cost=0, min_cost=0,
        impact_cost=0, volume_threshold=volume_threshold,
    ), day


@pytest.mark.parametrize("scale", [0.25, 0.5])
def test_ratio_execution_preserves_real_100_share_lots_and_cash(tmp_path, scale):
    from qlib.backtest.decision import Order
    from qlib.backtest.position import Position

    exchange, day = _exchange(tmp_path / "data", first_factor=scale)
    price = exchange.get_deal_price("SH600000", day, day, direction=Order.BUY)
    factor = exchange.get_factor("SH600000", day, day)
    amount = exchange.round_amount_by_trade_unit(10550.0 / price, factor)
    position = Position(cash=10550.0)
    order = Order(stock_id="SH600000", amount=amount, direction=Order.BUY, start_time=day, end_time=day)
    value, cost, _ = exchange.deal_order(order, position=position)

    assert order.deal_amount * scale == pytest.approx(1000.0)
    assert position.get_cash() == pytest.approx(550.0)
    assert value == pytest.approx(10000.0)
    assert cost == 0.0
    assert position.calculate_value() == pytest.approx(10550.0)


@pytest.mark.parametrize("scale", [0.25, 0.5])
def test_ratio_execution_preserves_turnover_and_raw_share_volume_limit(tmp_path, scale):
    from qlib.backtest.decision import Order
    from qlib.backtest.position import Position

    exchange, day = _exchange(
        tmp_path / "data", first_factor=scale, volume_threshold=("current", "0.25*$volume")
    )
    price = exchange.get_deal_price("SH600000", day, day, direction=Order.BUY)
    volume = exchange.get_volume("SH600000", day, day)
    assert price * volume == pytest.approx(12000.0)
    position = Position(cash=100000.0)
    order = Order(stock_id="SH600000", amount=10000.0, direction=Order.BUY, start_time=day, end_time=day)
    value, cost, _ = exchange.deal_order(order, position=position)
    assert order.deal_amount * scale == pytest.approx(300.0)
    assert value == pytest.approx(3000.0)
    assert cost == 0.0
    assert position.get_cash() == pytest.approx(97000.0)


@pytest.mark.parametrize("first_offset", [-2.0, 1e-12])
@pytest.mark.parametrize("cleaned", [False, True])
def test_cash_execution_keeps_full_symbol_semantics_after_date_filtering(tmp_path, first_offset, cleaned):
    root = tmp_path / "data"
    provider, _ = _provider(root, cash=True, first_offset=first_offset, second_offset=0.0)
    if cleaned:
        _publish_cleaning_plan(root, event_dates={"SH600000": ["2026-01-02"]})
        descriptor = direct.build_direct_provider_descriptor(quant_data_root=root, cleaning_mode="cleaned")
        provider = direct.QuantDataFeatureProvider(descriptor_path=descriptor, quant_data_root=root)
    assert provider.feature("SH600000", "$factor", 1, 1, "day").iloc[0] == 1.0
    assert provider.feature("SH600000", "$volume", 1, 1, "day").iloc[0] == 1200.0
    assert provider.feature("SH600000", "$vwap", 1, 1, "day").iloc[0] == 9.75


def test_zero_cash_offsets_allow_ratio_execution_when_entire_history_is_proven(tmp_path):
    provider, _ = _provider(tmp_path / "data", cash=True, first_offset=0.0, second_offset=0.0)
    assert provider.feature("SH600000", "$factor", 0, 2, "day").tolist() == [0.5, 0.75, 1.0]
    assert provider.feature("SH600000", "$volume", 0, 2, "day").tolist() == [2400.0, 1600.0, 1200.0]


def _copy_descriptor(provider, name, payload):
    descriptor = provider._context.descriptor_path
    directory = descriptor.parent.with_name(name)
    directory.mkdir(exist_ok=True)
    if directory != descriptor.parent:
        for filename in ("CALENDAR.parquet", "MEMBERSHIP_INTERVALS.parquet"):
            (directory / filename).write_bytes((descriptor.parent / filename).read_bytes())
    copied = directory / direct.DESCRIPTOR_NAME
    copied.write_text(json.dumps(payload), encoding="utf-8")
    return copied


@pytest.mark.parametrize("legacy", ["explicit", "missing_mode", "missing_contract"])
def test_legacy_descriptors_preserve_unit_factor_raw_volume_and_prices(tmp_path, legacy):
    root = tmp_path / "data"
    provider, _ = _provider(root, cash=False)
    payload = json.loads(provider._context.descriptor_path.read_bytes())
    if legacy == "explicit":
        payload["field_contract"]["factor_semantics"] = "unit_on_available_bar_no_second_adjustment"
    elif legacy == "missing_mode":
        del payload["field_contract"]["factor_semantics"]
    else:
        del payload["field_contract"]
    descriptor = _copy_descriptor(provider, f"legacy-{legacy}", payload)
    old = direct.QuantDataFeatureProvider(descriptor_path=descriptor, quant_data_root=root)
    assert old.feature("SH600000", "$factor", 0, 2, "day").tolist() == [1.0] * 3
    assert old.feature("SH600000", "$volume", 0, 2, "day").tolist() == [1200.0] * 3
    assert old.feature("SH600000", "$close", 0, 2, "day").tolist() == [5.0, 9.0, 14.0]
    assert old.feature("SH600000", "$vwap", 0, 2, "day").tolist() == [5.5, 9.75, 14.0]


def test_ratio_execution_descriptor_gets_new_identity_without_rewriting_legacy(tmp_path):
    root = tmp_path / "data"
    provider, snapshot = _provider(root, cash=False)
    payload = json.loads(provider._context.descriptor_path.read_bytes())
    payload["field_contract"]["factor_semantics"] = "unit_on_available_bar_no_second_adjustment"
    legacy = _copy_descriptor(provider, "canonical-vwap-affine-v1", payload)
    legacy_bytes = legacy.read_bytes()
    manifest_bytes = (snapshot / "DATASET_MANIFEST.json").read_bytes()
    current = direct.build_direct_provider_descriptor(quant_data_root=root)
    assert current != legacy
    assert legacy.read_bytes() == legacy_bytes
    assert (snapshot / "DATASET_MANIFEST.json").read_bytes() == manifest_bytes
    assert json.loads(current.read_bytes())["field_contract"]["factor_semantics"] == "ratio_factor_inverse_volume_v1"


@pytest.mark.parametrize("mode", ["unknown_execution_mode", None])
def test_explicit_unknown_execution_semantics_are_rejected(tmp_path, mode):
    root = tmp_path / "data"
    provider, _ = _provider(root, cash=False)
    payload = json.loads(provider._context.descriptor_path.read_bytes())
    payload["field_contract"]["factor_semantics"] = mode
    descriptor = _copy_descriptor(provider, "unknown-execution-mode", payload)
    with pytest.raises(direct.DirectProviderError, match="factor semantics"):
        direct.QuantDataFeatureProvider(descriptor_path=descriptor, quant_data_root=root)


@pytest.mark.parametrize("fault", ["missing", "factor", "offset", "factor_and_index", "symlink"])
@pytest.mark.parametrize("field", ["$vwap", "$factor", "$volume"])
def test_vwap_rejects_changed_factor_evidence_after_bar_cache_warmup(tmp_path, fault, field):
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
        provider.feature("SH600000", field, 0, 2, "day")
