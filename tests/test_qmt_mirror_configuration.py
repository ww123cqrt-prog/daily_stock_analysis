from types import SimpleNamespace
import pytest


def test_mac_stock_and_etf_research_bases_are_not_silently_switched(monkeypatch):
    from src.services import script_universe_loader as loader
    monkeypatch.setenv('QMT_STOCK_ADJUSTMENT','front_ratio')
    monkeypatch.setenv('QMT_ETF_ADJUSTMENT','front')
    client=SimpleNamespace(instrument_types=lambda symbols:{
        'SH600519':{'stock':True},'SH510300':{'etf':True}})
    assert loader._research_adjustment(client,['SH600519'])=='front_ratio'
    assert loader._research_adjustment(client,['SH510300'])=='front'
    with pytest.raises(ValueError,match='mixed|separate'):
        loader._research_adjustment(client,['SH600519','SH510300'])


def test_default_windows_basis_does_not_need_new_sdk_operation(monkeypatch):
    from src.services import script_universe_loader as loader
    monkeypatch.delenv('QMT_STOCK_ADJUSTMENT',raising=False)
    monkeypatch.delenv('QMT_ETF_ADJUSTMENT',raising=False)
    monkeypatch.delenv('QMT_ADJUSTMENT',raising=False)
    assert loader._research_adjustment(object(),['SH600519'])=='front'


@pytest.mark.parametrize('basis', ['front_ratio', 'back_ratio'])
def test_ratio_basis_retains_training_provenance(basis):
    from quant_workflow.scripts._mlflow_run_provenance import normalize_provider_lineage
    identity = dict(backend='qmt_native', experiment_id='mirror-experiment',
                    bars_sha256='a'*64, inputs_sha256='b'*64, price_basis=basis)
    assert normalize_provider_lineage(identity) == identity


@pytest.mark.parametrize('selected,benchmark', [
    ('SH600519','SH510300'), ('SH510300','SH600519'),
])
def test_cross_asset_benchmark_cannot_silently_switch_basis(monkeypatch, selected, benchmark):
    from src.services import script_universe_loader as loader
    monkeypatch.setenv('QMT_STOCK_ADJUSTMENT','front_ratio')
    monkeypatch.setenv('QMT_ETF_ADJUSTMENT','front')
    client=SimpleNamespace(instrument_types=lambda symbols:{
        'SH600519':{'stock':True},'SH510300':{'etf':True},'SZ399006':{'index':True}})
    with pytest.raises(ValueError,match='mixed|separate'):
        loader._research_adjustment(client,[selected],benchmarks=[benchmark,'SZ399006'])


def test_etf_with_neutral_index_benchmark_keeps_cash_basis(monkeypatch):
    from src.services import script_universe_loader as loader
    monkeypatch.setenv('QMT_STOCK_ADJUSTMENT','front_ratio')
    monkeypatch.setenv('QMT_ETF_ADJUSTMENT','front')
    client=SimpleNamespace(instrument_types=lambda symbols:{
        'SH510300':{'etf':True},'SZ399006':{'index':True}})
    assert loader._research_adjustment(client,['SH510300'],benchmarks=['SZ399006'])=='front'


def test_unresolved_remote_download_is_not_frozen_as_a_complete_input(monkeypatch):
    import pandas as pd
    from src.services import script_universe_loader as loader
    monkeypatch.delenv('QMT_DATASET_DIR',raising=False)
    frame=pd.DataFrame([dict(symbol='SH600519',date=pd.Timestamp('2024-01-02'))])
    frame.attrs['download_attempts']=[dict(symbol='SH600519',status='no_rows')]
    client=SimpleNamespace(read_bars=lambda *a,**kw:frame,
        instrument_types=lambda codes:{'SH600519':{'stock':True}})
    monkeypatch.setattr(loader.QMTClient,'from_env',lambda:client)
    def forbidden(*a,**kw):raise AssertionError('Unresolved input must not be frozen')
    monkeypatch.setattr(loader,'save_dataset',forbidden)
    with pytest.raises(loader.MarketDataUnavailable,match='SH600519'):
        loader.load_market_input(pool='custom',custom_symbols=['SH600519'],start='2024-01-02',end='2024-01-05')


def test_explicit_mirror_input_serves_training_provider_and_dsa_view(tmp_path, monkeypatch):
    import hashlib
    import json
    import struct
    import subprocess
    import pandas as pd
    from src.services import script_universe_loader as loader
    from quant_workflow.qmt_native_provider import QMTNativeFeatureProvider

    root = tmp_path / 'mirror'
    (root / 'objects').mkdir(parents=True)
    days = pd.date_range('2024-01-03', periods=3)
    files = {}
    for symbol, initial, kind in [('SH600519', 10., 'stock'),
                                  ('SH510300', 2., 'etf'), ('SZ399006', 100., 'index')]:
        records = []
        for i, day in enumerate(days):
            price = round((initial + i) * 1000)
            stamp = int(day.tz_localize('Asia/Shanghai').timestamp())
            records.append(struct.pack('<16I', stamp, price, price, price, price,
                                       0, 20, 451, 20000, 0, 0, 1065353216,
                                       1065353216, price, 0, 32766))
        body = bytes.fromhex('feffffffffffff7f') + b''.join(records)
        digest = hashlib.sha256(body).hexdigest()
        (root / 'objects' / f'{digest}.DAT').write_bytes(body)
        files[symbol] = dict(path=f'objects/{digest}.DAT', sha256=digest, size=len(body),
                             types={kind: True}, actions=[], suspend_flags={})
    manifest = dict(schema_version=1, source_data_dir=str(tmp_path / 'qmt-source'),
                    sdk_version='synthetic', synced_at='2024-01-06T00:00:00+08:00',
                    files=files, universes={}, calendars={
                        market: [day.date().isoformat() for day in days] for market in ('SH', 'SZ')})
    (root / 'MIRROR.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.delenv('QMT_DATASET_DIR', raising=False)
    monkeypatch.delenv('QMT_STOCK_ADJUSTMENT', raising=False)
    monkeypatch.delenv('QMT_ETF_ADJUSTMENT', raising=False)
    monkeypatch.setenv('QMT_ADJUSTMENT', 'front')
    monkeypatch.setenv('QMT_TRANSPORT', 'mirror')
    monkeypatch.setenv('QMT_MIRROR_DIR', str(root))
    monkeypatch.setenv('QMT_DOWNLOAD_MISSING', 'false')
    monkeypatch.setattr(loader, 'reports_root', lambda: tmp_path / 'reports')
    def forbidden(*args, **kwargs):
        raise AssertionError('Mirror preparation and frozen reuse must stay SDK-free')
    monkeypatch.setattr(subprocess, 'run', forbidden)

    selected = ['SH600519', 'SH510300']
    frame, directory = loader.load_market_input(pool='custom', custom_symbols=selected,
                                               start='2024-01-03', end='2024-01-05')
    assert frame.attrs['source_kind'] == 'qmt_native_mirror'
    assert frame.attrs['mirror_synced_at'] == manifest['synced_at']
    assert frame.attrs['universe'] == selected
    assert frame.attrs['volume_units']['SH510300'] == 'units'
    assert len(frame) == 9
    monkeypatch.setattr(loader.QMTClient, 'from_env', forbidden)
    provider = QMTNativeFeatureProvider(input_dir=directory)
    assert provider.feature('SH600519', '$close', 0, 2, 'day').tolist() == [10., 11., 12.]
    view, _ = loader.load_market_view('custom', custom_symbols=selected, quant_data_root=directory,
                                      cache_dir=tmp_path / 'bundles')
    assert view.close[:, list(view.symbols).index('SH600519')].tolist() == [10., 11., 12.]
    assert view.volume[0, list(view.symbols).index('SH510300')] == 2000.
