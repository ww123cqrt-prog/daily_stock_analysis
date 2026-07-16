from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "robust_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("robust_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_board_from_symbol_covers_main_growth_star_and_unknown() -> None:
    module = load_module()

    assert module.board_from_symbol("SH600000") == "main"
    assert module.board_from_symbol("SZ000001") == "main"
    assert module.board_from_symbol("SZ300001") == "growth"
    assert module.board_from_symbol("SH688001") == "star"
    assert module.board_from_symbol("BJ430001") == "other"


def test_normalize_feature_frame_operates_within_same_date_and_board() -> None:
    module = load_module()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-05"), "SH600000"),
            (pd.Timestamp("2026-01-05"), "SH600001"),
            (pd.Timestamp("2026-01-05"), "SZ300001"),
            (pd.Timestamp("2026-01-05"), "SZ300002"),
        ],
        names=["datetime", "instrument"],
    )
    features = pd.DataFrame(
        {"factor": [1.0, 3.0, 10.0, 14.0], "constant": [2.0] * 4}, index=index
    )

    normalized = module.normalize_feature_frame(features, add_board_features=True)

    assert np.allclose(normalized["factor"].values, [-1.0, 1.0, -1.0, 1.0])
    assert normalized["constant"].eq(0.0).all()
    assert normalized.loc[(pd.Timestamp("2026-01-05"), "SH600000"), "BOARD_MAIN"] == 1.0
    assert (
        normalized.loc[(pd.Timestamp("2026-01-05"), "SZ300001"), "BOARD_GROWTH"] == 1.0
    )
    assert normalized["BOARD_STAR"].eq(0.0).all()


def test_transform_prepared_frame_preserves_label_and_feature_column_groups() -> None:
    module = load_module()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-01-05"), "SH600000"),
            (pd.Timestamp("2026-01-05"), "SH600001"),
        ],
        names=["datetime", "instrument"],
    )
    columns = pd.MultiIndex.from_tuples([("feature", "F1"), ("label", "LABEL0")])
    frame = pd.DataFrame([[1.0, 0.1], [3.0, 0.2]], index=index, columns=columns)

    transformed = module.transform_prepared_frame(frame, col_set=["feature", "label"])

    assert np.allclose(transformed[("feature", "F1")].values, [-1.0, 1.0])
    assert transformed[("label", "LABEL0")].tolist() == [0.1, 0.2]
    assert ("feature", "BOARD_MAIN") in transformed.columns
