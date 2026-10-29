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
