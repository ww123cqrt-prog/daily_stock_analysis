from __future__ import annotations

import pytest

from scripts import build_mlflow_i18n_catalog as catalog_builder


def test_catalog_builder_rejects_installed_mlflow_version_mismatch() -> None:
    with pytest.raises(RuntimeError, match="expected MLflow 3.14.0"):
        catalog_builder.validate_mlflow_version("3.15.0")


def test_catalog_builder_accepts_recorded_mlflow_version() -> None:
    catalog_builder.validate_mlflow_version("3.14.0")
