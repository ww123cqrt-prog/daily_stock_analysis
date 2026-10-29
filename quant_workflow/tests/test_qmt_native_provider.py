"""Storage boundary checks for Qlib's frozen QMT input provider."""
import pytest
from quant_workflow.tests.test_qmt_native_training import frozen_input


def test_qlib_providers_share_one_verified_input_load(monkeypatch, frozen_input):
    from quant_workflow import qmt_native_provider as provider
    directory, _, _, _ = frozen_input
    original = provider.load_dataset
    calls = []
    def observed_load(path):
        calls.append(path)
        return original(path)
    monkeypatch.setattr(provider, "load_dataset", observed_load)
    provider.init_qlib_from_quant_data(input_dir=directory, kernels=1)
    assert calls == [str(directory)]


def test_provider_rejects_vwap_instead_of_synthesizing_it(frozen_input):
    from qlib.data import D
    from quant_workflow.qmt_native_provider import init_qlib_from_quant_data, DirectProviderError
    directory, _, calendar, _ = frozen_input
    init_qlib_from_quant_data(input_dir=directory, kernels=1)
    with pytest.raises(DirectProviderError, match="vwap"):
        D.features(["SH600519"], ["$vwap"], start_time=calendar[10],end_time=calendar[12],disk_cache=0)
