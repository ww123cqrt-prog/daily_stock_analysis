"""Keep frozen QMT inputs separate from existing DSA research providers."""
import builtins
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from quant_workflow.scripts import _paths
from quant_workflow.tests.test_qmt_native_training import frozen_input


def test_native_initialization_never_imports_publication_contract_in_fresh_process(
    tmp_path, frozen_input,
):
    directory, _, _, _ = frozen_input
    program = textwrap.dedent("""
        import importlib
        import json
        from pathlib import Path
        import sys

        def assert_native_imports():
            assert not [name for name in sys.modules
                        if name.rsplit('.', 1)[-1] == 'quant_data_contract']
            assert 'quant_workflow.quant_data_qlib_provider' not in sys.modules

        assert_native_imports()
        root, directory = map(Path, sys.argv[1:])
        sys.path.insert(0, str(root))
        from quant_workflow.scripts import _paths
        assert_native_imports()
        _paths.init_qlib_workflow(provider_uri=directory, kernels=1)
        from qlib.data import D
        symbols = list(_paths.provider_instrument_ranges(directory))
        calendar = _paths.provider_calendar_days(directory)
        values = D.features([symbols[0]], ['$close'], start_time=calendar[0],
                            end_time=calendar[0], disk_cache=0)
        assert not values.empty
        import yaml
        config = yaml.safe_load((root / 'quant_workflow/configs/baseline_qmt_native.yaml').read_text(encoding='utf-8'))
        handler = config['task']['dataset']['kwargs']['handler']
        assert handler['module_path'] == 'quant_workflow.qmt_native_provider'
        native = importlib.import_module(handler['module_path'])
        assert getattr(native, handler['class']).get_feature_config()[0]
        importlib.import_module('quant_workflow.scripts.train_baseline_lgb')
        assert_native_imports()
        print(json.dumps({'native_read': True, 'publication_contract_loaded': False}))
    """)
    result = subprocess.run(
        [sys.executable, "-B", "-c", program, str(Path(__file__).resolve().parents[2]), str(directory)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"publication_contract_loaded": false' in result.stdout


def test_binary_provider_metadata_does_not_read_qmt_input(tmp_path, monkeypatch):
    provider = tmp_path / "binary"
    (provider / "calendars").mkdir(parents=True)
    (provider / "instruments").mkdir()
    (provider / "calendars" / "day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    (provider / "instruments" / "all.txt").write_text(
        "SH600000\t2024-01-02\t2024-01-03\n", encoding="utf-8"
    )
    (provider / "instruments" / "csi800.txt").write_text(
        "SH600000\t2024-01-02\t2024-01-03\n", encoding="utf-8"
    )
    monkeypatch.setenv("QMT_DATASET_DIR", str(tmp_path / "unrelated-qmt-input"))

    assert _paths.provider_calendar_days(provider) == ["2024-01-02", "2024-01-03"]
    assert _paths.provider_instrument_ranges(provider) == {
        "SH600000": ("2024-01-02", "2024-01-03")
    }
    assert _paths.provider_has_instrument_pool(provider, "csi800")
    assert _paths.qlib_init_config_for_uri(provider)["provider_uri"] == str(provider.resolve())


def test_native_remains_default_and_explicit_canonical_stays_separate(monkeypatch, frozen_input):
    directory, _, _, _ = frozen_input
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    assert _paths.qlib_workflow_provider_uri() == directory
    config = _paths.qlib_workflow_init_config(kernels=1)
    assert config["feature_provider"]["kwargs"] == {"input_dir": str(directory)}
    with pytest.raises(ValueError, match="format|QMT|qmt_native"):
        _paths.qlib_workflow_init_config(provider="canonical", provider_uri=directory)


def test_native_reads_without_historical_catalog_or_publication_state(monkeypatch, frozen_input):
    from qlib.data import D
    from quant_workflow import quant_data_qlib_provider as provider

    directory, frame, calendar, symbols = frozen_input
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    monkeypatch.setenv("QUANT_DATA_ROOT", str(directory / "missing-historical-catalog"))

    def historical_access_forbidden(*args, **kwargs):
        raise AssertionError("Native reads must not resolve historical catalogs or publication state")

    for name in ("resolve_dataset", "_load_descriptor", "build_direct_provider_descriptor"):
        monkeypatch.setattr(provider, name, historical_access_forbidden)
    monkeypatch.setattr(_paths, "qlib_direct_provider_descriptor", historical_access_forbidden)
    _paths.init_qlib_workflow(kernels=1)
    assert _paths.provider_calendar_days(directory) == calendar.strftime("%Y-%m-%d").tolist()
    assert set(_paths.provider_instrument_ranges(directory)) == set(symbols)
    actual = D.features([symbols[0]], ["$close"],
        start_time=calendar[0], end_time=calendar[0], disk_cache=0)
    assert actual["$close"].iloc[0] == pytest.approx(frame.iloc[0].close)


def test_explicit_canonical_helpers_use_existing_descriptor(monkeypatch, tmp_path):
    descriptor = tmp_path / "PROVIDER_DESCRIPTOR.json"
    descriptor.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("QLIB_PROVIDER_URI", "")
    monkeypatch.setenv("QLIB_CLEANED_PROVIDER_URI", "")
    calls = []

    def descriptor_for(root, index_dataset_id, cleaning_mode):
        calls.append(cleaning_mode)
        return descriptor

    monkeypatch.setattr(_paths, "_resolve_direct_qlib_descriptor", descriptor_for, raising=False)
    assert _paths.qlib_workflow_provider_uri("canonical") == tmp_path
    assert _paths.qlib_workflow_provider_uri("cleaned") == tmp_path
    assert calls == ["canonical", "cleaned"]


@pytest.mark.parametrize("volume_unit,multiplier", [
    (None, 1), ("shares", 1), ("units", 1), ("lots", 100), ("lot_100_shares_rounded", 100),
])
def test_existing_parquet_reader_has_no_retired_package_dependency(
    tmp_path, monkeypatch, volume_unit, multiplier,
):
    from quant_workflow import quant_data_qlib_provider as provider
    from quant_workflow.scripts import quant_data_contract as contract

    snapshot = tmp_path / "stock-snapshot"
    (snapshot / "bars").mkdir(parents=True)
    dates = pd.to_datetime(["2024-01-02", "2024-01-04"])
    frame = pd.DataFrame({"date": dates, "close": [10.0, 11.0], "volume": [100.0, 120.0]})
    path = snapshot / "bars" / "SH600000.parquet"
    frame.to_parquet(path, index=False)
    file_index = pd.DataFrame([{
        "symbol": "SH600000", "relative_path": "bars/SH600000.parquet",
        "row_count": 2, "first_date": dates[0], "last_date": dates[-1],
        "sha256": contract.sha256_file(path),
    }])
    context = SimpleNamespace(
        quant_data_root=tmp_path, snapshot_path=snapshot, file_index=file_index,
        calendar=pd.date_range("2024-01-02", "2024-01-04"),
        descriptor={"dataset": {"dataset_id": "ashare_daily_qfq"}},
        manifest={"source": {"volume_unit": volume_unit}} if volume_unit else {},
        index_snapshot_path=None, index_file_index=None,
        cleaning={"mode": "canonical"},
    )
    monkeypatch.setattr(provider, "_load_descriptor", lambda *args: context, raising=False)
    reader = provider.QuantDataFeatureProvider(
        descriptor_path=tmp_path / "PROVIDER_DESCRIPTOR.json", quant_data_root=tmp_path
    )
    assert reader.feature("SH600000", "$close", 0, 2, "day").tolist()[::2] == [10.0, 11.0]
    assert np.isnan(reader.feature("SH600000", "$volume", 0, 2, "day").iloc[1])
    assert reader.feature("SH600000", "$volume", 0, 2, "day").iloc[0] == 100.0 * multiplier
    reader._frames.clear()
    frame.assign(close=[90.0, 91.0]).to_parquet(path, index=False)
    with pytest.raises(provider.DirectProviderError, match="hash"):
        reader.feature("SH600000", "$close", 0, 2, "day")


def test_context_cache_can_be_cleared_for_native_inputs(frozen_input):
    from quant_workflow import qmt_native_provider as provider
    directory, _, _, _ = frozen_input
    provider._load_input(str(directory))
    provider._clear_context_cache()
    assert provider._load_input.cache_info().currsize == 0


def test_binary_configuration_does_not_import_custom_providers(tmp_path, monkeypatch):
    provider = tmp_path / "binary"
    (provider / "calendars").mkdir(parents=True)
    (provider / "calendars" / "day.txt").write_text("2024-01-02\n", encoding="utf-8")
    original_import = builtins.__import__

    def import_without_custom_provider(name, *args, **kwargs):
        if name == "quant_workflow.quant_data_qlib_provider":
            raise AssertionError("Binary inputs use Qlib's own storage providers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_custom_provider)
    assert _paths.qlib_init_config_for_uri(provider)["provider_uri"] == str(provider)


def test_switching_from_native_to_binary_keeps_original_values(tmp_path, frozen_input):
    from qlib.data import D

    directory, _, _, _ = frozen_input
    _paths.init_qlib_workflow(provider_uri=directory, kernels=1)
    provider = tmp_path / "binary"
    for folder in ("calendars", "instruments", "features/sh600000"):
        (provider / folder).mkdir(parents=True)
    (provider / "calendars/day.txt").write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    (provider / "instruments/all.txt").write_text(
        "SH600000\t2024-01-02\t2024-01-03\n", encoding="utf-8"
    )
    np.asarray([0.0, 10.0, 11.0], dtype="<f4").tofile(provider / "features/sh600000/close.day.bin")
    _paths.init_qlib_workflow(provider="canonical", provider_uri=provider, kernels=1)
    actual = D.features(["SH600000"], ["$close"],
        start_time="2024-01-02", end_time="2024-01-03", disk_cache=0)
    assert actual["$close"].tolist() == [10.0, 11.0]
