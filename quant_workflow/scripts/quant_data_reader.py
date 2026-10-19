"""Compatibility entry for DSA scripts; all bar reading lives in quant_qmt_data."""
from quant_qmt_data.quant_data_reader import build_update_universe, read_bars
from quant_qmt_data import DataContractError

__all__ = ["DataContractError", "build_update_universe", "read_bars"]
