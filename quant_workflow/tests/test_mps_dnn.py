from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mps_dnn.py"
REQUIREMENTS_PATH = (
    Path(__file__).resolve().parents[1] / "requirements-mps-dnn.txt"
)


def load_module():
    spec = importlib.util.spec_from_file_location("mps_dnn", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_model_rejects_cpu_before_optional_torch_is_loaded() -> None:
    module = load_module()

    with pytest.raises(module.MPSDNNUnavailable, match="does not permit CPU"):
        module.MPSDNNModel(device="cpu")


def test_feature_and_label_filters_non_finite_rows_without_reordering_columns() -> None:
    module = load_module()
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2025-01-02"), "SH600000"),
            (pd.Timestamp("2025-01-02"), "SZ000001"),
            (pd.Timestamp("2025-01-03"), "SH600000"),
        ],
        names=["datetime", "instrument"],
    )
    columns = pd.MultiIndex.from_tuples([("feature", "A"), ("feature", "B"), ("label", "LABEL0")])
    frame = pd.DataFrame(
        [[1.0, 2.0, 0.01], [np.nan, 3.0, 0.02], [4.0, 5.0, np.inf]],
        index=index,
        columns=columns,
    )

    features, label = module._feature_and_label(frame)

    assert list(features.columns) == ["A", "B"]
    assert features.index.tolist() == [index[0]]
    assert label.index.tolist() == [index[0]]
    assert float(label.iloc[0]) == pytest.approx(0.01)


def test_formal_mps_environment_pins_every_direct_training_dependency() -> None:
    pinned = {
        line.split("==", 1)[0].lower(): line.split("==", 1)[1]
        for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    }

    assert set(pinned) == {
        "catboost",
        "cloudpickle",
        "lightgbm",
        "mlflow",
        "numpy",
        "pandas",
        "pyarrow",
        "pyqlib",
        "pyyaml",
        "scikit-learn",
        "torch",
        "xgboost",
    }
    assert all(
        version and not any(char in version for char in "<>=~*")
        for version in pinned.values()
    )
