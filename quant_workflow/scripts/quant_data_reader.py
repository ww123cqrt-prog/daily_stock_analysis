"""Read script inputs from the selected frozen QMT experiment, fully offline."""
from pathlib import Path
import pandas as pd
from quant_qmt_data.dataset import load_dataset
from quant_qmt_data.reader import normalize_symbol
from quant_workflow.scripts._paths import qmt_dataset_dir


def read_bars(symbols=None, *, start=None, end=None, fields=None, input_dir=None) -> pd.DataFrame:
    frame = load_dataset(Path(input_dir) if input_dir is not None else qmt_dataset_dir())
    selected = frame.attrs["universe"] if symbols is None else [normalize_symbol(s) for s in symbols]
    mask = frame.symbol.isin(selected)
    if start is not None:
        mask &= frame.date >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.date <= pd.Timestamp(end)
    columns = list(frame.columns) if fields is None else list(dict.fromkeys(["date", "symbol", *fields]))
    return frame.loc[mask, columns].sort_values(["symbol", "date"]).reset_index(drop=True)
