from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "compare_native_qlib_benchmarks.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_native_qlib_benchmarks", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_provider(root: Path, *, missing_last: bool = False) -> Path:
    calendar = ("2026-07-22", "2026-07-23", "2026-07-24")
    (root / "calendars").mkdir(parents=True)
    (root / "calendars" / "day.txt").write_text(
        "\n".join(calendar) + "\n", encoding="utf-8"
    )
    for symbol_index, symbol in enumerate(
        ("SH000300", "SH000905", "SH000906"), start=1
    ):
        feature_dir = root / "features" / symbol.lower()
        feature_dir.mkdir(parents=True)
        for field_index, field in enumerate(
            ("open", "high", "low", "close", "volume", "factor"), start=1
        ):
            values = np.array(
                [
                    symbol_index * 100 + field_index + offset
                    for offset in range(len(calendar))
                ],
                dtype="<f4",
            )
            if missing_last and symbol == "SH000905" and field == "close":
                values[-1] = np.nan
            np.concatenate((np.array([0.0], dtype="<f4"), values)).tofile(
                feature_dir / f"{field}.day.bin"
            )
    return root


def test_full_benchmark_comparison_uses_dates_and_all_required_fields(
    tmp_path: Path,
) -> None:
    module = load_module()
    native = write_provider(tmp_path / "native")
    compatibility = write_provider(tmp_path / "compatibility")

    result = module.compare_benchmark_providers(native, compatibility)

    assert result["status"] == "pass"
    assert result["symbols"]["SH000300"]["coverage"] == {
        "start": "2026-07-22",
        "end": "2026-07-24",
        "observations": 3,
    }
    assert set(result["symbols"]["SH000906"]["fields"]) == set(
        module.REQUIRED_FIELDS
    )


def test_comparison_rejects_a_missing_benchmark_date(tmp_path: Path) -> None:
    module = load_module()
    native = write_provider(tmp_path / "native")
    compatibility = write_provider(
        tmp_path / "compatibility", missing_last=True
    )

    with pytest.raises(module.DataContractError, match="SH000905/close.*date"):
        module.compare_benchmark_providers(native, compatibility)


def test_comparison_rejects_a_value_difference(tmp_path: Path) -> None:
    module = load_module()
    native = write_provider(tmp_path / "native")
    compatibility = write_provider(tmp_path / "compatibility")
    target = compatibility / "features" / "sh000300" / "open.day.bin"
    values = np.fromfile(target, dtype="<f4")
    values[-1] += 1.0
    values.tofile(target)

    with pytest.raises(module.DataContractError, match="SH000300/open.*value"):
        module.compare_benchmark_providers(native, compatibility)
