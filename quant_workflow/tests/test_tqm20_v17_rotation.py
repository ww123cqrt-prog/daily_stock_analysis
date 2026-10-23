"""TQM-20 v1.7 策略测试"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "tqm20_v17_rotation.py"
    spec = importlib.util.spec_from_file_location("tqm20_v17_rotation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame(symbol: str, closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "date": dates,
            "open": closes,
            "close": closes,
            "symbol": symbol,
        }
    )


def test_tqm_factor_basic():
    """测试TQM因子基本计算。"""
    module = load_module()
    # 创建一个简单的上涨序列（更渐进的增长）
    closes = [100.0] + [100.0 * (1.01 ** i) for i in range(1, 22)]
    result = module._tqm_factor(closes)
    assert result is not None
    assert result["momentum"] > 0
    assert result["efficiency"] > 0
    assert result["score"] > 0


def test_tqm_factor_volatility_filter():
    """测试波动率过滤。"""
    module = load_module()
    # 创建高波动率序列
    import numpy as np
    np.random.seed(42)
    closes = [100.0] + [100 * (1 + np.random.normal(0, 0.1)) for _ in range(20)]
    result = module._tqm_factor(closes, vol_limit=0.10)  # 低波动率限制
    assert result is None  # 应该被过滤掉


def test_build_tqm_signal_table_basic():
    """测试基本信号表构建。"""
    module = load_module()
    # 创建4个ETF的价格数据
    data = {
        "SZ159915": frame("SZ159915", [100.0] * 25 + [120.0]),
        "SH513100": frame("SH513100", [100.0] * 25 + [110.0]),
        "SH512890": frame("SH512890", [100.0] * 25 + [105.0]),
        "SH518880": frame("SH518880", [100.0] * 25 + [115.0]),
    }
    
    table = module.build_tqm_signal_table(data)
    assert "signal" in table.columns
    assert "target_position" in table.columns
    assert len(table) > 0


def test_simulate_open_to_open_basic():
    """测试基本收益模拟。"""
    module = load_module()
    # 创建简单的信号表
    dates = pd.date_range("2026-01-01", periods=4, freq="B")
    table = pd.DataFrame(
        {
            "date": dates,
            "target_position": ["CASH", "SZ159915", "SH513100", "CASH"],
            "SZ159915_open": [100.0, 100.0, 100.0, 100.0],
            "SH513100_open": [100.0, 100.0, 100.0, 100.0],
            "SH512890_open": [100.0, 100.0, 100.0, 100.0],
            "SH518880_open": [100.0, 100.0, 100.0, 100.0],
        }
    )
    result = module.simulate_open_to_open(table, side_cost_bps=10.0)
    assert "metrics" in result
    assert "periods" in result
    assert "equity_curve" in result


def test_v17_variants_are_one_four_etf_experiment_family():
    """V1.7 candidates share one version and add exactly one fourth ETF."""
    module = load_module()

    assert module.BASE_SYMBOLS == ("SZ159915", "SH513100", "SH512890")
    assert module.DEFAULT_VARIANT == "gold"
    assert set(module.V17_VARIANTS) == {
        "gold",
        "sp500",
        "germany_dax",
        "hang_seng_tech",
        "china_30y_treasury",
    }
    assert module.V17_VARIANTS["gold"]["symbol"] == "SH518880"
    assert module.V17_VARIANTS["sp500"]["symbol"] == "SH513500"
    assert module.V17_VARIANTS["germany_dax"]["symbol"] == "SH513030"
    assert "DAX" in module.V17_VARIANTS["germany_dax"]["name"]
    assert module.V17_VARIANTS["hang_seng_tech"]["symbol"] == "SH513180"
    assert module.V17_VARIANTS["china_30y_treasury"]["symbol"] == "SH511090"

    for variant in module.V17_VARIANTS:
        universe = module.build_v17_universe(variant)
        assert len(universe) == 4
        assert universe[:3] == module.BASE_SYMBOLS
        assert universe[-1] == module.V17_VARIANTS[variant]["symbol"]

    assert module.DEFAULT_SYMBOLS == module.build_v17_universe("gold")


def test_v17_rejects_unknown_fourth_asset_variant():
    module = load_module()

    with pytest.raises(ValueError, match="unknown V1.7 variant"):
        module.build_v17_universe("dow")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
