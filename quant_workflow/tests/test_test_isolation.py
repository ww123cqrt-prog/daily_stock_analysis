"""The workflow suite must use the same private runtime as backend tests."""

import os
from pathlib import Path


def test_workflow_runtime_is_private(tmp_path):
    root = tmp_path.resolve()
    assert Path.cwd().resolve() == root
    for key in ("ENV_FILE", "DATABASE_PATH", "QUANT_WORKFLOW_REPORTS_ROOT",
                "QUANT_WORKFLOW_MLRUNS_DIR", "QUANT_DATA_ROOT", "DSA_STRATEGY_ASSETS_DB"):
        assert Path(os.environ[key]).resolve().is_relative_to(root), key
