from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "_robust_membership_integrity.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "robust_membership_integrity", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reader_rejects_membership_with_in_progress_checkpoint(tmp_path: Path) -> None:
    module = load_module()
    pit_dir = tmp_path / "pit"
    pit_dir.mkdir()
    (pit_dir / module.MEMBERSHIP_BUILD_IN_PROGRESS_FILENAME).write_text(
        json.dumps({"status": "running", "completed_date_count": 250}),
        encoding="utf-8",
    )

    with module.membership_read_lock(pit_dir) as snapshot:
        with pytest.raises(
            module.MembershipIntegrityError,
            match="membership build is still in progress",
        ) as exc_info:
            snapshot.ensure_transaction_clean()

    assert exc_info.value.reason_code == "membership_build_in_progress"
