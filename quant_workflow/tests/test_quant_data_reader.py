"""Script bar reads use the selected frozen input, never a live source."""
import pandas as pd
from pandas.testing import assert_frame_equal
from quant_workflow.tests.test_qmt_native_training import frozen_input


def test_script_reader_uses_frozen_selection_and_preserves_metadata(monkeypatch, frozen_input):
    from quant_workflow.scripts.quant_data_reader import read_bars
    directory, frame, calendar, _ = frozen_input
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    result = read_bars(["SH600519"], start=calendar[29], end=calendar[31], fields=["close", "volume"])
    expected = frame.loc[(frame.symbol == "SH600519") & frame.date.between(calendar[29], calendar[31]), ["date", "symbol", "close", "volume"]].reset_index(drop=True)
    assert_frame_equal(result, expected)
    assert result.attrs["price_basis"] == "front"
    assert result.attrs["volume_units"] == frame.attrs["volume_units"]
    assert pd.Timestamp(calendar[30]) not in result.date.tolist()
