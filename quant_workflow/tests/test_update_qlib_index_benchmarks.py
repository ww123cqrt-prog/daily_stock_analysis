import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_module(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frames() -> dict[str, pd.DataFrame]:
    return {
        symbol: pd.DataFrame(
            {
                "open": [10.0, 10.2],
                "high": [10.3, 10.4],
                "low": [9.9, 10.1],
                "close": [10.2, 10.3],
                "volume": [1000.0, 1100.0],
            },
            index=["2020-01-02", "2020-01-03"],
        )
        for symbol in ("SZ399006", "SH000688")
    }


def test_targeted_download_requests_exactly_two_indexes() -> None:
    module = load_module("update_qlib_index_benchmarks")
    calls: list[tuple[str, ...]] = []

    def tencent(_start: str, _end: str, **kwargs):
        calls.append(tuple(kwargs["symbols"]))
        return _frames()

    def eastmoney(*_args, **_kwargs):
        raise AssertionError("fallback must not run after a valid Tencent response")

    result, source = module._download_pair(
        "1990-12-19",
        "2020-01-03",
        use_system_proxy=False,
        tencent_download=tencent,
        eastmoney_download=eastmoney,
        source_order=("tencent", "eastmoney"),
    )

    assert calls == [("SZ399006", "SH000688")]
    assert set(result) == {"SZ399006", "SH000688"}
    assert source["provider_codes"] == {
        "SZ399006": "sz399006",
        "SH000688": "sh000688",
    }


def test_targeted_frame_validation_rejects_a_third_symbol() -> None:
    module = load_module("update_qlib_index_benchmarks")
    frames = _frames()
    frames["SH000300"] = frames["SZ399006"].copy()

    with pytest.raises(module.DataContractError, match="outside the requested pair"):
        module._validate_target_frames(
            frames,
            ["2020-01-02", "2020-01-03"],
        )


def test_targeted_frame_validation_accepts_complete_pair() -> None:
    module = load_module("update_qlib_index_benchmarks")

    result = module._validate_target_frames(
        _frames(),
        ["2020-01-02", "2020-01-03"],
    )

    assert set(result) == {"SZ399006", "SH000688"}
