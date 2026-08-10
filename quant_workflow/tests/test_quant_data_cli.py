from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


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


def _write_etf_release_snapshot(
    contract,
    quant_data: Path,
    *,
    snapshot_id: str,
    symbols: tuple[str, ...],
    retrieved_at: str,
) -> Path:
    snapshot = quant_data / "staging" / snapshot_id / snapshot_id
    for symbol in symbols:
        bars = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-08-05", "2026-08-06"]),
                "open": [1.0, 1.1],
                "high": [1.2, 1.3],
                "low": [0.9, 1.0],
                "close": [1.1, 1.2],
                "volume": [100.0, 110.0],
                "source_id": ["tencent", "tencent"],
            }
        )
        factors = pd.DataFrame(
            {
                "date": bars["date"],
                "factor": [1.0, 1.0],
                "offset": [0.0, 0.0],
            }
        )
        for component in ("raw", "qfq"):
            path = snapshot / component / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            bars.to_parquet(path, index=False)
        factor_path = snapshot / "factors" / f"{symbol}.parquet"
        factor_path.parent.mkdir(parents=True, exist_ok=True)
        factors.to_parquet(factor_path, index=False)

    pd.DataFrame(
        columns=["symbol", "effective_trade_date", "action_type"]
    ).to_parquet(snapshot / "corporate_actions.parquet", index=False)
    universe = pd.DataFrame(
        {
            "symbol": list(symbols),
            "code": [symbol[2:] for symbol in symbols],
            "market": [symbol[:2] for symbol in symbols],
            "name": [f"ETF {symbol[2:]}" for symbol in symbols],
        }
    )
    universe.to_parquet(snapshot / "ETF_UNIVERSE.parquet", index=False)
    universe_file_sha256 = contract.sha256_file(snapshot / "ETF_UNIVERSE.parquet")
    contract.atomic_write_json(
        snapshot / "UNIVERSE_MANIFEST.json",
        {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "symbol_count": len(symbols),
            "universe_sha256": "u" * 64,
            "universe_file_sha256": universe_file_sha256,
        },
    )
    contract.finalize_paired_price_snapshot(
        snapshot,
        dataset_id="cn_etf_daily",
        snapshot_id=snapshot_id,
        role="canonical",
        status="ready",
        source={
            "provider": "tencent",
            "interface": "test_fixture",
            "universe_path": "ETF_UNIVERSE.parquet",
            "universe_manifest_path": "UNIVERSE_MANIFEST.json",
            "universe_sha256": "u" * 64,
            "snapshot_retention": {
                "policy": "current_plus_one_previous",
                "max_complete_snapshots": 2,
            },
        },
        adjustment={
            "method": "provider_native_paired",
            "adjusted_at": retrieved_at,
            "adjustment_as_of": "2026-08-06",
            "adjusted_fields": ["open", "high", "low", "close"],
            "normalization_rule": (
                "latest_valid_trade_date_factor_equals_1_and_offset_equals_0"
            ),
            "algorithm_version": "provider-native-affine-paired-v1",
            "paired_raw_component_path": "raw",
            "volume_adjusted": False,
        },
        producer={"script": "test_quant_data_cli.py", "version": "1"},
        source_retrieved_at=retrieved_at,
    )
    return snapshot


def _build_etf_release_fixture(
    module,
    tmp_path: Path,
    *,
    current_symbols: tuple[str, ...] = (
        "SH512890",
        "SH513010",
        "SH510720",
        "SZ159915",
        "SH513100",
    ),
) -> tuple[Path, Path, Path]:
    contract = sys.modules["quant_data_contract"]
    quant_data = tmp_path / "quant_data"
    dataset_root = quant_data / "parquet" / "etf_1d" / "cn_etf_daily"
    backup_staging = _write_etf_release_snapshot(
        contract,
        quant_data,
        snapshot_id="cn-etf-backup",
        symbols=("SZ159915",),
        retrieved_at="2026-08-06T18:00:00+08:00",
    )
    backup = contract.publish_snapshot(
        backup_staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    current_staging = _write_etf_release_snapshot(
        contract,
        quant_data,
        snapshot_id="cn-etf-current",
        symbols=current_symbols,
        retrieved_at="2026-08-07T18:00:00+08:00",
    )
    current = contract.publish_snapshot(
        current_staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    contract.write_root_manifest(
        quant_data,
        {
            "cn_etf_daily": contract.catalog_entry_from_snapshot(
                current,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
    )
    return quant_data, current, backup


def write_pointer_state(module, root: Path, marker: str) -> dict[Path, bytes]:
    state: dict[Path, bytes] = {}
    for relative in module.ASHARE_UPDATE_POINTERS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = json.dumps({"marker": marker, "path": relative.as_posix()}).encode()
        path.write_bytes(value)
        state[path] = value
    return state


def fake_direct_provider_builder(
    root: Path,
    calls: list[dict[str, object]] | None = None,
    *,
    fail_cleaned: bool = False,
):
    def build_direct(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        mode = str(kwargs["cleaning_mode"])
        if fail_cleaned and mode == "cleaned":
            raise RuntimeError("cleaned direct provider build failed")
        descriptor = root / f"direct-{mode}" / "PROVIDER_DESCRIPTOR.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text("{}", encoding="utf-8")
        return descriptor

    return build_direct


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


def test_resolve_named_artifact_component(tmp_path: Path, capsys) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    artifact = tmp_path / "snapshot" / "daily"
    artifact.mkdir(parents=True)

    with patch.object(
        module,
        "resolve_dataset_artifact",
        return_value=types.SimpleNamespace(artifact_path=artifact),
    ) as resolver:
        result = module.main(
            [
                "--root",
                str(quant_data),
                "resolve",
                "pit_csi800",
                "--component",
                "daily",
            ]
        )

    assert result == 0
    assert capsys.readouterr().out.strip() == str(artifact)
    resolver.assert_called_once_with(
        "pit_csi800",
        "daily",
        quant_data_root=quant_data.resolve(),
    )


def test_bars_command_returns_json_records(tmp_path: Path, capsys) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-04"]), "close": [10.2]}
    )

    with patch.object(module, "read_bars", return_value=frame) as reader:
        result = module.main(
            [
                "--root",
                str(quant_data),
                "bars",
                "ashare_daily_qfq",
                "SH600000",
                "--basis",
                "qfq",
                "--fields",
                "close",
                "--start",
                "2026-08-04",
                "--end",
                "2026-08-04",
            ]
        )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == [
        {"date": "2026-08-04", "close": 10.2}
    ]
    reader.assert_called_once_with(
        "ashare_daily_qfq",
        "SH600000",
        quant_data_root=quant_data.resolve(),
        price_basis="qfq",
        fields=["close"],
        start="2026-08-04",
        end="2026-08-04",
    )


def test_export_universe_writes_snapshot_bound_update_input(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    output = tmp_path / "ashare-update-universe.json"
    payload = {
        "schema_version": 1,
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-001",
        "manifest_sha256": "a" * 64,
        "latest_trade_date": "2026-08-04",
        "symbol_count": 2,
        "all": ["SH600000", "SZ000001"],
    }

    with patch.object(module, "build_update_universe", return_value=payload) as builder:
        result = module.main(
            [
                "--root",
                str(quant_data),
                "export-universe",
                "ashare_daily_qfq",
                "--basis",
                "qfq",
                "--output",
                str(output),
            ]
        )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert json.loads(capsys.readouterr().out) == {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-001",
        "symbol_count": 2,
        "output": str(output.resolve()),
    }
    builder.assert_called_once_with(
        "ashare_daily_qfq",
        quant_data_root=quant_data.resolve(),
        price_basis="qfq",
    )


def test_release_validation_checks_current_backup_and_strategy_symbols(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, _backup = _build_etf_release_fixture(module, tmp_path)

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_passed"
    assert payload["dataset_id"] == "cn_etf_daily"
    assert payload["network_requests"] == 0
    assert payload["snapshot_retention"] == {
        "count": 2,
        "policy": "current_plus_one_previous",
    }
    assert payload["current"]["snapshot_id"] == "cn-etf-current"
    assert payload["backup"]["snapshot_id"] == "cn-etf-backup"
    assert payload["strategy_symbols"]["status"] == "passed"


def test_release_validation_allows_governed_resume_cache(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, _backup = _build_etf_release_fixture(module, tmp_path)
    cache = quant_data / "cache" / "cn_etf_daily" / "v1"
    cache.mkdir(parents=True)
    (cache / "progress.json").write_text('{"status":"incomplete"}', encoding="utf-8")

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "full_validation_passed"


def test_release_validation_fails_when_backup_is_missing(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, backup = _build_etf_release_fixture(module, tmp_path)
    shutil.rmtree(backup)

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_failed"
    assert "backup" in payload["error"]["message"]


def test_release_validation_fails_on_extra_snapshot_or_old_entry(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, current, _backup = _build_etf_release_fixture(module, tmp_path)
    snapshots = current.parent
    shutil.copytree(snapshots / "cn-etf-backup", snapshots / "cn-etf-extra")

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_failed"
    assert "exactly" in payload["error"]["message"]

    shutil.rmtree(snapshots / "cn-etf-extra")
    (quant_data / "parquet" / "etf_1d" / "etf_momentum_daily_qfq").mkdir(
        parents=True
    )
    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )
    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert "legacy" in payload["error"]["message"]


def test_release_validation_reports_missing_strategy_symbol(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, _backup = _build_etf_release_fixture(
        module,
        tmp_path,
        current_symbols=("SH513010", "SH510720", "SZ159915", "SH513100"),
    )

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert "SH512890" in payload["error"]["message"]


def test_release_validation_fails_on_backup_hash_damage(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, backup = _build_etf_release_fixture(module, tmp_path)
    damaged = pd.read_parquet(backup / "qfq" / "SZ159915.parquet")
    damaged.loc[0, "close"] = 1.15
    damaged.to_parquet(backup / "qfq" / "SZ159915.parquet", index=False)

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_failed"
    assert "hash mismatch" in payload["error"]["message"]


def test_release_validation_rejects_old_etf_catalog_identity(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, _backup = _build_etf_release_fixture(module, tmp_path)
    catalog_path = quant_data / "DATA_MANIFEST.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["datasets"]["cn_etf_daily_qfq"] = {
        "dataset_id": "cn_etf_daily_qfq",
        "data_kind": "daily_bars",
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "dataset_root": "parquet/etf_1d/cn_etf_daily_qfq",
    }
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    result = module.main(
        [
            "--root",
            str(quant_data),
            "validate",
            "cn_etf_daily",
            "--release",
        ]
    )

    assert result == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_failed"
    assert "legacy" in payload["error"]["message"]


def test_release_helpers_reject_symlinks_and_malformed_universe_counts(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data, current, _backup = _build_etf_release_fixture(module, tmp_path)
    linked = current / "linked-universe.parquet"
    linked.symlink_to(current / "ETF_UNIVERSE.parquet")
    with pytest.raises(module.DataContractError, match="unsafe"):
        module._release_path(current, linked.name, "test path")

    universe_manifest_path = current / "UNIVERSE_MANIFEST.json"
    universe_manifest = json.loads(
        universe_manifest_path.read_text(encoding="utf-8")
    )
    universe_manifest["selected_count"] = "not-an-integer"
    universe_manifest_path.write_text(json.dumps(universe_manifest), encoding="utf-8")
    manifest = json.loads(
        (current / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )
    qfq_index = pd.read_parquet(current / "QFQ_FILE_INDEX.parquet")

    with pytest.raises(module.DataContractError, match="symbol count"):
        module._validate_release_universe(current, manifest, qfq_index)


def test_release_validation_rejects_strategy_symbol_outside_qfq_component(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data, current, _backup = _build_etf_release_fixture(module, tmp_path)
    qfq_index_path = current / "QFQ_FILE_INDEX.parquet"
    qfq_index = pd.read_parquet(qfq_index_path)
    qfq_index.loc[
        qfq_index["symbol"] == "SH513100",
        "relative_path",
    ] = "raw/SH513100.parquet"
    qfq_index.to_parquet(qfq_index_path, index=False)
    manifest = json.loads(
        (current / "DATASET_MANIFEST.json").read_text(encoding="utf-8")
    )

    with pytest.raises(module.DataContractError, match="outside qfq component"):
        module._validate_release_snapshot(
            quant_data,
            current,
            manifest=manifest,
            required_symbols=module.ETF_RELEASE_REQUIRED_SYMBOLS,
        )


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
    builder_calls: list[dict[str, object]] = []
    build_direct = fake_direct_provider_builder(quant_data, builder_calls)

    with patch.dict(sys.modules, {"download_baostock_v2": fake}), patch.object(
        module, "build_direct_provider_descriptor", side_effect=build_direct
    ):
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
    assert payload == {
        "legacy_compatibility": {"status": "not_requested"},
        "qfq": str(qfq),
        "qlib_cleaned_provider": str(quant_data / "direct-cleaned"),
        "qlib_provider": str(quant_data / "direct-canonical"),
        "raw": str(raw),
    }
    assert [call["cleaning_mode"] for call in builder_calls] == [
        "canonical",
        "cleaned",
    ]
    status = json.loads(
        (
            quant_data
            / "runtime"
            / "status"
            / "quant_data_update_latest.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "completed"
    assert status["legacy_compatibility"] == {"status": "not_requested"}
    assert not (quant_data / module.ASHARE_UPDATE_TRANSACTION_PATH).exists()


def test_update_ashare_v2_dispatches_incremental_bundle_without_full_rebuild(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "BUNDLE_MANIFEST.json").write_text(
        json.dumps({"bundle_mode": "incremental"}),
        encoding="utf-8",
    )
    write_catalog(quant_data)
    raw_stage = quant_data / "staging" / "raw-incremental"
    qfq_stage = quant_data / "staging" / "qfq-incremental"
    raw_published = quant_data / "published-raw"
    qfq_published = quant_data / "published-qfq"
    incremental_calls: list[dict[str, object]] = []
    publish_calls: list[dict[str, object]] = []

    def build_incremental(**kwargs):
        incremental_calls.append(kwargs)
        return types.SimpleNamespace(
            raw_stage=raw_stage,
            qfq_stage=qfq_stage,
            base_latest_trade_date="2026-08-04",
            latest_trade_date="2026-08-07",
            downloaded_rows=16_548,
            reused_symbols=("SH600000",),
            factor_changed_symbols=("SZ000001",),
            qfq_recalculated_symbols=("SZ000001",),
        )

    def reject_full_rebuild(**_kwargs):
        raise AssertionError("incremental bundle must not use the full builder")

    def publish_group(snapshots, **kwargs):
        publish_calls.append({"snapshots": snapshots, **kwargs})
        return [raw_published, qfq_published]

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=reject_full_rebuild
        ),
        "incremental_update_ashare_v2": types.SimpleNamespace(
            build_incremental_price_snapshots=build_incremental
        ),
    }
    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "publish_snapshot_group",
        side_effect=publish_group,
        create=True,
    ), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data),
    ):
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
    assert incremental_calls == [
        {"bundle_dir": bundle, "quant_data_root": quant_data.resolve()}
    ]
    assert publish_calls == [
        {
            "snapshots": [
                (
                    raw_stage,
                    quant_data.resolve()
                    / "parquet/ashare_1d/ashare_daily_raw",
                ),
                (
                    qfq_stage,
                    quant_data.resolve()
                    / "parquet/ashare_1d/ashare_daily_qfq",
                ),
            ],
            "quant_data_root": quant_data.resolve(),
            "update_root_catalog": True,
            "publication_lock_held": True,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["update_mode"] == "incremental"
    assert payload["incremental"]["base_latest_trade_date"] == "2026-08-04"
    assert payload["incremental"]["latest_trade_date"] == "2026-08-07"
    assert payload["incremental"]["downloaded_rows"] == 16_548
    assert payload["incremental"]["factor_changed_symbols"] == 1


def test_update_ashare_v2_builds_direct_provider_descriptors_not_bin_snapshots(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    write_catalog(quant_data)
    raw = quant_data / "raw-snapshot"
    qfq = quant_data / "qfq-snapshot"
    calls: list[dict[str, object]] = []

    def build_direct(**kwargs):
        calls.append(kwargs)
        mode = str(kwargs["cleaning_mode"])
        descriptor = quant_data / f"direct-{mode}" / "PROVIDER_DESCRIPTOR.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text("{}", encoding="utf-8")
        return descriptor

    fake_download = types.SimpleNamespace(
        build_snapshots_from_bundle=lambda **_kwargs: (raw, qfq)
    )
    fake_legacy = types.SimpleNamespace(
        build_qlib_v2=lambda **_kwargs: quant_data / "legacy-qlib",
    )
    fake_legacy_cleaned = types.SimpleNamespace(
        build_cleaned_qlib_v2=lambda **_kwargs: quant_data / "legacy-cleaned",
    )
    with patch.dict(
        sys.modules,
        {
            "download_baostock_v2": fake_download,
            "build_qlib_v2": fake_legacy,
            "build_cleaned_qlib_v2": fake_legacy_cleaned,
        },
    ), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=build_direct,
        create=True,
    ):
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
    assert payload["qlib_provider"] == str(quant_data / "direct-canonical")
    assert payload["qlib_cleaned_provider"] == str(quant_data / "direct-cleaned")
    assert [call["cleaning_mode"] for call in calls] == ["canonical", "cleaned"]
    assert "qlib_v2" not in payload
    assert "qlib_cleaned_v2" not in payload


def test_update_cn_index_daily_uses_formal_parquet_updater(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    calls: list[dict[str, object]] = []

    def update_cn_index_snapshot(**kwargs):
        calls.append(kwargs)
        return {
            "status": "already_current",
            "dataset_id": "cn_index_daily",
            "latest_trade_date": "2026-08-04",
            "network_requests_for_symbols": [],
        }

    fake_module = types.SimpleNamespace(
        update_cn_index_snapshot=update_cn_index_snapshot
    )
    with patch.dict(sys.modules, {"update_cn_index_dataset": fake_module}):
        result = module._update(
            tmp_path / "quant_data",
            "cn_index_daily",
            "2026-08-04",
            None,
        )

    assert result == 0
    assert calls == [
        {
            "quant_data_root": (tmp_path / "quant_data").resolve(),
            "end_date": "2026-08-04",
        }
    ]
    assert json.loads(capsys.readouterr().out)["status"] == "already_current"


def test_update_cn_index_daily_invalid_through_is_a_bounded_cli_error(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    fake_module = types.SimpleNamespace(
        update_cn_index_snapshot=lambda **_kwargs: (_ for _ in ()).throw(
            ValueError("invalid --through date")
        )
    )

    with patch.dict(sys.modules, {"update_cn_index_dataset": fake_module}):
        result = module.main(
            [
                "--root",
                str(tmp_path / "quant_data"),
                "update",
                "cn_index_daily",
                "--through",
                "not-a-date",
            ]
        )

    captured = capsys.readouterr()
    assert result == 2
    assert "invalid --through date" in captured.err
    assert "Traceback" not in captured.err


def test_update_recovers_interrupted_transaction_before_new_publication(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    original = write_pointer_state(module, quant_data, "before-interruption")
    module._write_update_transaction(
        quant_data,
        module._capture_file_state(quant_data),
        dataset_id="ashare_daily_qfq",
        started_at="2026-07-30T10:00:00+08:00",
    )
    write_pointer_state(module, quant_data, "partially-published")
    observed_restored_state = False

    def build_snapshots(**_kwargs):
        nonlocal observed_restored_state
        observed_restored_state = all(
            path.read_bytes() == value for path, value in original.items()
        )
        return quant_data / "new-raw", quant_data / "new-qfq"

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
    }

    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data),
    ):
        assert (
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                tmp_path / "bundle",
            )
            == 0
        )

    assert observed_restored_state
    assert not (quant_data / module.ASHARE_UPDATE_TRANSACTION_PATH).exists()


def test_corrupt_interrupted_transaction_blocks_new_publication(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_pointer_state(module, quant_data, "stable")
    journal = quant_data / module.ASHARE_UPDATE_TRANSACTION_PATH
    journal.parent.mkdir(parents=True)
    journal.write_text("not-json", encoding="utf-8")
    builder_called = False

    def build_snapshots(**_kwargs):
        nonlocal builder_called
        builder_called = True
        return quant_data / "new-raw", quant_data / "new-qfq"

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: quant_data / "new-qlib"
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: quant_data / "new-cleaned"
        ),
    }

    with patch.dict(sys.modules, fake_modules):
        with pytest.raises(module.DataContractError, match="transaction journal"):
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                tmp_path / "bundle",
            )

    assert not builder_called
    assert journal.exists()


def test_update_refreshes_legacy_compatibility_only_when_explicitly_requested(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    write_catalog(quant_data)
    calls: list[dict[str, object]] = []

    def build_compatibility(**kwargs):
        calls.append(kwargs)
        return {"status": "published", "run_id": "compat-001"}

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=lambda **_kwargs: (
                quant_data / "raw",
                quant_data / "qfq",
            )
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=build_compatibility
        ),
    }

    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data),
    ):
        result = module.main(
            [
                "--root",
                str(quant_data),
                "update",
                "ashare_daily_qfq",
                "--bundle-dir",
                str(bundle),
                "--refresh-legacy-compatibility",
            ]
        )

    assert result == 0
    assert calls == [
        {
            "quant_data_root": quant_data.resolve(),
            "publication_lock_held": True,
        }
    ]
    payload = json.loads(capsys.readouterr().out)
    assert payload["legacy_compatibility"] == {
        "status": "published",
        "run_id": "compat-001",
    }


def test_update_rolls_back_all_native_pointers_when_cleaned_build_fails(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    write_catalog(quant_data)
    tracked = {
        quant_data / "DATA_MANIFEST.json": b'{"old":"manifest"}',
        quant_data
        / "parquet"
        / "ashare_1d"
        / "ashare_daily_raw"
        / "LATEST.json": b'{"old":"raw"}',
        quant_data
        / "parquet"
        / "ashare_1d"
        / "ashare_daily_qfq"
        / "LATEST.json": b'{"old":"qfq"}',
    }
    for path, value in tracked.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    tracked_paths = list(tracked)

    def build_snapshots(**_kwargs):
        for path in tracked_paths[0:3]:
            path.write_text('{"new":"snapshot"}', encoding="utf-8")
        return quant_data / "new-raw", quant_data / "new-qfq"

    def build_direct(**kwargs):
        if kwargs["cleaning_mode"] == "cleaned":
            raise RuntimeError("cleaned direct provider build failed")
        descriptor = quant_data / "direct-canonical" / "PROVIDER_DESCRIPTOR.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text("{}", encoding="utf-8")
        return descriptor

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: None
        ),
    }

    with patch.dict(sys.modules, fake_modules), patch.object(
        module, "build_direct_provider_descriptor", side_effect=build_direct
    ):
        try:
            module._update(quant_data, "ashare_daily_qfq", None, bundle)
        except module.DataContractError as exc:
            assert "rolled back" in str(exc)
        else:
            raise AssertionError("partial update must fail")

    for path, value in tracked.items():
        assert path.read_bytes() == value
    status = json.loads(
        (
            quant_data
            / "runtime"
            / "status"
            / "quant_data_update_latest.json"
        ).read_text(encoding="utf-8")
    )
    assert status["status"] == "rolled_back"
    assert status["failed_step"] == "direct_provider_cleaned"
    assert status["completed_steps"] == [
        "raw_qfq_v2",
        "direct_provider_canonical",
    ]
    assert not (quant_data / module.ASHARE_UPDATE_TRANSACTION_PATH).exists()


def test_compatibility_refresh_failure_does_not_roll_back_native_v2(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    bundle = tmp_path / "bundle"
    write_catalog(quant_data)
    pointers = {
        quant_data / "parquet/ashare_1d/ashare_daily_raw/LATEST.json": "raw-new",
        quant_data / "parquet/ashare_1d/ashare_daily_qfq/LATEST.json": "qfq-new",
    }

    def publish_pointer(path: Path, value: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
        return path.parent / value

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=lambda **_kwargs: (
                publish_pointer(next(iter(pointers)), "raw-new"),
                publish_pointer(list(pointers)[1], "qfq-new"),
            )
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("compatibility failed")
            )
        ),
    }

    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data),
    ):
        assert (
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                bundle,
                refresh_legacy_compatibility=True,
            )
            == 0
        )

    for path, value in pointers.items():
        assert path.read_text(encoding="utf-8") == value
    output = json.loads(capsys.readouterr().out)
    assert output["legacy_compatibility"] == {
        "status": "legacy_refresh_failed",
        "error_type": "RuntimeError",
    }
    status = json.loads(
        (quant_data / "runtime/status/quant_data_update_latest.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "completed_with_legacy_warning"
    assert status["completed_steps"] == [
        "raw_qfq_v2",
        "direct_provider_canonical",
        "direct_provider_cleaned",
    ]


def test_rollback_status_write_failure_does_not_mask_update_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    original_manifest = (quant_data / "DATA_MANIFEST.json").read_bytes()
    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=lambda **_kwargs: (
                quant_data / "new-raw",
                quant_data / "new-qfq",
            )
        ),
    }
    monkeypatch.setattr(
        module,
        "_write_update_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("status unavailable")),
    )

    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data, fail_cleaned=True),
    ):
        with pytest.raises(module.DataContractError, match="was rolled back") as exc_info:
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                tmp_path / "bundle",
                refresh_legacy_compatibility=True,
            )

    assert "status" in str(exc_info.value)
    assert (quant_data / "DATA_MANIFEST.json").read_bytes() == original_manifest


def test_pointer_rollback_failure_reports_update_and_rollback_error_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("bundle build failed")
            )
        ),
        "build_qlib_v2": types.SimpleNamespace(build_qlib_v2=lambda **_kwargs: None),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: None
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: None
        ),
    }
    monkeypatch.setattr(
        module,
        "_restore_file_state",
        lambda _state: (_ for _ in ()).throw(OSError("pointer rollback failed")),
    )

    with patch.dict(sys.modules, fake_modules):
        with pytest.raises(module.DataContractError) as exc_info:
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                tmp_path / "bundle",
            )

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "OSError" in message


def test_atomic_restore_deletion_fsyncs_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    pointer = tmp_path / "quant_data" / "LATEST.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text('{"snapshot_id":"new"}', encoding="utf-8")
    fsync_calls: list[int] = []
    monkeypatch.setattr(module.os, "fsync", fsync_calls.append)

    module._atomic_restore_bytes(pointer, None)

    assert not pointer.exists()
    assert len(fsync_calls) == (0 if module.os.name == "nt" else 1)


def test_update_dispatches_common_etf_updater(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    published = tmp_path / "quant_data" / "published-qfq"
    calls: list[dict[str, object]] = []
    fake_module = types.SimpleNamespace(
        update_cn_etf_snapshot=lambda **kwargs: calls.append(kwargs) or published
    )

    with patch.dict(sys.modules, {"download_cn_etf_data": fake_module}):
        assert (
            module._update(
                tmp_path / "quant_data",
                "cn_etf_daily",
                "2026-08-06",
                None,
            )
            == 0
        )

    assert calls == [
        {
            "quant_data_dir": tmp_path / "quant_data",
            "end_date": "2026-08-06",
        }
    ]
    assert capsys.readouterr().out.strip() == str(published)


def test_completed_status_write_failure_does_not_report_publication_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=lambda **_kwargs: (
                quant_data / "new-raw",
                quant_data / "new-qfq",
            )
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: {
                "status": "published"
            }
        ),
    }
    monkeypatch.setattr(
        module,
        "_write_update_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("status unavailable")),
    )

    with patch.dict(sys.modules, fake_modules), patch.object(
        module,
        "build_direct_provider_descriptor",
        side_effect=fake_direct_provider_builder(quant_data),
    ):
        result = module._update(
            quant_data,
            "ashare_daily_qfq",
            None,
            tmp_path / "bundle",
        )

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["status_record"]["status"] == "write_failed"
    assert output["status_record"]["error_type"] == "OSError"


def test_update_refuses_when_another_ashare_update_holds_the_lock(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    lock_path = quant_data / "runtime" / "locks" / "ashare_v2_update.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+b") as handle:
        module.lock_file(handle, exclusive=True, blocking=False)
        try:
            with pytest.raises(module.DataContractError, match="already in progress"):
                module._update(
                    quant_data,
                    "ashare_daily_qfq",
                    None,
                    tmp_path / "bundle",
                )
        finally:
            module.unlock_file(handle)


def test_update_holds_dataset_publication_lock_across_all_publish_steps(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    observed_steps: list[str] = []

    def assert_publication_lock(step: str, **kwargs):
        if kwargs:
            assert kwargs["publication_lock_held"] is True
        lock_path = (
            quant_data / "runtime" / "locks" / "dataset_publication.lock"
        )
        with lock_path.open("a+b") as handle:
            with pytest.raises(BlockingIOError):
                module.lock_file(handle, exclusive=True, blocking=False)
        observed_steps.append(step)

    def build_snapshots(**kwargs):
        assert_publication_lock("raw_qfq_v2", **kwargs)
        return quant_data / "new-raw", quant_data / "new-qfq"

    def build_direct(**_kwargs):
        assert_publication_lock("direct_provider")
        mode = str(_kwargs["cleaning_mode"])
        descriptor = quant_data / f"direct-{mode}" / "PROVIDER_DESCRIPTOR.json"
        descriptor.parent.mkdir(parents=True, exist_ok=True)
        descriptor.write_text("{}", encoding="utf-8")
        return descriptor

    def build_compatibility(**kwargs):
        assert_publication_lock("compatibility", **kwargs)
        return {"status": "published"}

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=build_compatibility
        ),
    }

    with patch.dict(sys.modules, fake_modules), patch.object(
        module, "build_direct_provider_descriptor", side_effect=build_direct
    ):
        assert (
            module._update(
                quant_data,
                "ashare_daily_qfq",
                None,
                tmp_path / "bundle",
                refresh_legacy_compatibility=True,
            )
            == 0
        )

    assert observed_steps == [
        "raw_qfq_v2",
        "direct_provider",
        "direct_provider",
        "compatibility",
    ]


def test_human_description_does_not_call_compatibility_output_audit_only() -> None:
    module = load_module()
    text = module._human_description(
        {
            "dataset_id": "ashare_merged_qfq_compatibility",
            "role": "canonical",
            "status": "ready",
            "lifecycle": "compatibility_output",
            "price_basis": "qfq",
            "path": "parquet/ashare_1d/merged_baostock_jqdata",
            "snapshot_available": False,
            "source": {"provider": "quant_workflow", "interface": "v2_to_v1"},
            "adjustment": {"method": "local_factor"},
        }
    )

    assert "正式兼容输出" in text
    assert "只能审计" not in text


def test_human_description_reports_native_upstream_and_benchmark_lineage() -> None:
    module = load_module()
    text = module._human_description(
        {
            "dataset_id": "qlib_ashare_v2",
            "role": "canonical",
            "status": "ready",
            "price_basis": "qfq",
            "snapshot_available": True,
            "snapshot_id": "qlib-v2-001",
            "snapshot_path": "qlib/cn_data_ashare_v2/snapshots/qlib-v2-001",
            "source": {
                "provider": "quant_workflow",
                "interface": "v2_snapshot_to_qlib",
            },
            "adjustment": {"method": "upstream_qfq_snapshot"},
            "upstream_snapshots": [
                {
                    "dataset_id": "ashare_daily_qfq",
                    "snapshot_id": "qfq-001",
                    "content_sha256": "a" * 64,
                }
            ],
            "benchmark_complete": True,
            "benchmarks": {
                "calendar_sha256": "b" * 64,
                "source": {
                    "provider": "tencent",
                    "interface": "qt.gtimg.cn historical daily",
                    "evidence_sha256": "c" * 64,
                },
                "symbols": {
                    "SH000300": {
                        "coverage": {"start": "2005-04-08", "end": "2026-07-24"}
                    },
                    "SH000905": {
                        "coverage": {"start": "2007-01-15", "end": "2026-07-24"}
                    },
                    "SH000906": {
                        "coverage": {"start": "2007-01-15", "end": "2026-07-24"}
                    },
                },
            },
            "producer": {"script": "build_qlib_v2.py", "version": "4"},
        }
    )

    assert "上游数据集: ashare_daily_qfq" in text
    assert "上游快照: qfq-001" in text
    assert f"上游内容哈希: {'a' * 64}" in text
    assert "指数完整: 是" in text
    assert "指数来源: tencent / qt.gtimg.cn historical daily" in text
    assert f"交易日历哈希: {'b' * 64}" in text
    for symbol in ("SH000300", "SH000905", "SH000906"):
        assert symbol in text
