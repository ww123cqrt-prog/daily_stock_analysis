from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "quant_data_cli.py"
    spec = importlib.util.spec_from_file_location("quant_data_cli", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_catalog(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "DATA_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "path_format": "relative_to_quant_data_root",
                "datasets": {
                    "legacy_raw": {
                        "dataset_id": "legacy_raw",
                        "data_kind": "daily_bars",
                        "role": "legacy",
                        "status": "legacy_unknown",
                        "source": {
                            "provider": "unknown",
                            "interface": "unknown",
                        },
                        "price_basis": "raw",
                        "path": "parquet/legacy_raw",
                        "adjustment": {"method": "none"},
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_list_and_describe_expose_human_and_json_contract(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)

    assert module.main(["--root", str(quant_data), "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["dataset_id"] == "legacy_raw"
    assert listed[0]["price_basis"] == "raw"

    assert module.main(["--root", str(quant_data), "describe", "legacy_raw"]) == 0
    output = capsys.readouterr().out
    assert "价格口径: raw" in output
    assert "复权方法: none" in output
    assert "可用状态: legacy_unknown" in output


def test_resolve_returns_nonzero_for_unusable_legacy_dataset(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)

    result = module.main(
        [
            "--root",
            str(quant_data),
            "resolve",
            "legacy_raw",
            "--require-basis",
            "qfq",
        ]
    )

    assert result == 2
    assert "不能使用" in capsys.readouterr().err


def test_update_reports_registered_ashare_v2_blockers(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    path = quant_data / "DATA_MANIFEST.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["datasets"]["ashare_daily_qfq"] = {
        "dataset_id": "ashare_daily_qfq",
        "data_kind": "daily_bars",
        "role": "canonical",
        "status": "blocked",
        "source": {"provider": "baostock", "interface": "adjust_factor"},
        "price_basis": "qfq",
        "path": "parquet/ashare_1d/ashare_daily_qfq",
        "adjustment": {"method": "local_factor"},
        "blocker_codes": ["raw_snapshot_missing", "factor_snapshot_missing"],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = module.main(
        ["--root", str(quant_data), "update", "ashare_daily_qfq"]
    )

    assert result == 2
    error = capsys.readouterr().err
    assert "A-share v2 update is blocked" in error
    assert "raw_snapshot_missing" in error


def test_update_ashare_v2_builds_and_publishes_a_verified_bundle(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    write_catalog(quant_data)
    raw = quant_data / "raw-snapshot"
    qfq = quant_data / "qfq-snapshot"
    fake = types.SimpleNamespace(
        build_snapshots_from_bundle=lambda **kwargs: (raw, qfq)
    )

    with patch.dict(sys.modules, {"download_baostock_v2": fake}):
        result = module.main(
            [
                "--root",
                str(quant_data),
                "update",
                "ashare_daily_qfq",
                "--bundle-dir",
                str(bundle),
            ]
        )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"qfq": str(qfq), "raw": str(raw)}
