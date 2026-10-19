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
                "policy": "immutable_history",
                "max_complete_snapshots": 3,
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
        "policy": "immutable_history",
        "max_complete_snapshots": 3,
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


def test_release_validation_allows_single_complete_snapshot(
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

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_passed"
    assert payload["snapshot_retention"] == {
        "count": 1,
        "policy": "immutable_history",
        "max_complete_snapshots": 3,
    }
    assert payload["backup"] is None


def test_release_validation_accepts_extra_complete_snapshot_or_rejects_old_entry(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data, _current, _backup = _build_etf_release_fixture(module, tmp_path)
    contract = sys.modules["quant_data_contract"]
    dataset_root = quant_data / "parquet" / "etf_1d" / "cn_etf_daily"
    extra_staging = _write_etf_release_snapshot(
        contract,
        quant_data,
        snapshot_id="cn-etf-extra",
        symbols=(
            "SH512890",
            "SH513010",
            "SH510720",
            "SZ159915",
            "SH513100",
        ),
        retrieved_at="2026-08-08T18:00:00+08:00",
    )
    extra = contract.publish_snapshot(
        extra_staging,
        dataset_root=dataset_root,
        quant_data_root=quant_data,
    )
    contract.write_root_manifest(
        quant_data,
        {
            "cn_etf_daily": contract.catalog_entry_from_snapshot(
                extra,
                quant_data_root=quant_data,
                dataset_root=dataset_root,
            )
        },
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

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "full_validation_passed"
    assert payload["snapshot_retention"] == {
        "count": 3,
        "policy": "immutable_history",
        "max_complete_snapshots": 3,
    }
    assert payload["validated_snapshot_ids"] == [
        "cn-etf-backup",
        "cn-etf-current",
        "cn-etf-extra",
    ]

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


def test_expand_cn_etf_universe_cli_uses_the_separate_governed_entrypoint(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    calls: list[dict[str, object]] = []
    published = tmp_path / "quant_data" / "expanded-snapshot"

    def expand_cn_etf_universe(**kwargs):
        calls.append(kwargs)
        return published

    class FakeCnEtfDownloadError(RuntimeError):
        pass

    fake_module = types.SimpleNamespace(
        CnEtfDownloadError=FakeCnEtfDownloadError,
        expand_cn_etf_universe=expand_cn_etf_universe
    )
    with patch.dict(sys.modules, {"download_cn_etf_data": fake_module}):
        result = module.main(
            [
                "--root",
                str(tmp_path / "quant_data"),
                "expand-cn-etf-universe",
                "--target-size",
                "200",
                "--through",
                "2026-08-14",
            ]
        )

    assert result == 0
    assert calls == [
        {
            "quant_data_dir": (tmp_path / "quant_data").resolve(),
            "target_size": 200,
            "end_date": "2026-08-14",
        }
    ]
    assert capsys.readouterr().out.strip() == str(published)


def test_expand_cn_etf_universe_cli_rejects_unregistered_target_size(
    tmp_path: Path,
) -> None:
    module = load_module()

    with pytest.raises(SystemExit):
        module.main(
            [
                "--root",
                str(tmp_path / "quant_data"),
                "expand-cn-etf-universe",
                "--target-size",
                "150",
                "--through",
                "2026-08-14",
            ]
        )


def test_expand_cn_etf_universe_cli_handles_download_error_without_traceback(
    tmp_path: Path, capsys
) -> None:
    module = load_module()

    class FakeCnEtfDownloadError(RuntimeError):
        pass

    def expand_cn_etf_universe(**_kwargs):
        raise FakeCnEtfDownloadError("provider rejected the bounded request")

    fake_module = types.SimpleNamespace(
        CnEtfDownloadError=FakeCnEtfDownloadError,
        expand_cn_etf_universe=expand_cn_etf_universe,
    )
    with patch.dict(sys.modules, {"download_cn_etf_data": fake_module}):
        result = module.main(
            [
                "--root",
                str(tmp_path / "quant_data"),
                "expand-cn-etf-universe",
                "--through",
                "2026-08-14",
            ]
        )

    captured = capsys.readouterr()
    assert result == 2
    assert "provider rejected the bounded request" in captured.err
    assert "Traceback" not in captured.err
















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






def test_update_refuses_when_another_ashare_update_holds_the_lock(
    tmp_path: Path,
) -> None:
    from quant_qmt_data._file_lock import lock_file as provider_lock_file, unlock_file as provider_unlock_file
    module = load_module()
    quant_data = tmp_path / "quant_data"
    write_catalog(quant_data)
    lock_path = quant_data / "runtime" / "locks" / "ashare_v2_update.lock"
    lock_path.parent.mkdir(parents=True)
    with lock_path.open("a+b") as handle:
        provider_lock_file(handle, exclusive=True, blocking=False)
        try:
            with pytest.raises(module.DataContractError, match="already in progress"):
                module._update(
                    quant_data,
                    "ashare_daily_qfq",
                    None,
                    tmp_path / "bundle",
                )
        finally:
            provider_unlock_file(handle)




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


def test_compare_snapshots_command_writes_machine_readable_report(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    output = tmp_path / "comparison.json"
    payload = {
        "status": "increment_comparison_passed",
        "dataset_id": "ashare_daily_raw",
        "checked_symbols": 5516,
    }

    with patch.object(
        module, "compare_snapshot_increment", return_value=payload, create=True
    ) as compare:
        result = module.main(
            [
                "--root",
                str(quant_data),
                "compare-snapshots",
                "ashare_daily_raw",
                "--base-snapshot",
                "raw-base",
                "--candidate-snapshot",
                "raw-candidate",
                "--output",
                str(output),
            ]
        )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    compare.assert_called_once_with(
        "ashare_daily_raw",
        quant_data_root=quant_data.resolve(),
        base_snapshot_id="raw-base",
        candidate_snapshot_id="raw-candidate",
    )


def test_sample_increment_command_uses_comparison_report_for_category_coverage(
    tmp_path: Path, capsys
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    quant_data.mkdir()
    bundle = tmp_path / "bundle"
    comparison = tmp_path / "comparison.json"
    comparison_payload = {
        "status": "increment_comparison_passed",
        "factor_changed_symbols": ["SH600000"],
    }
    comparison.write_text(json.dumps(comparison_payload), encoding="utf-8")
    payload = {
        "status": "increment_sample_passed",
        "dataset_id": "ashare_daily_qfq",
        "sampled_symbol_count": 100,
    }

    with patch.object(
        module, "sample_increment_against_bundle", return_value=payload, create=True
    ) as sample:
        result = module.main(
            [
                "--root",
                str(quant_data),
                "sample-increment",
                "ashare_daily_qfq",
                "--base-snapshot",
                "qfq-base",
                "--candidate-snapshot",
                "qfq-candidate",
                "--bundle-dir",
                str(bundle),
                "--sample-size",
                "100",
                "--seed",
                "daily-check",
                "--comparison-report",
                str(comparison),
            ]
        )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == payload
    sample.assert_called_once_with(
        "ashare_daily_qfq",
        quant_data_root=quant_data.resolve(),
        base_snapshot_id="qfq-base",
        candidate_snapshot_id="qfq-candidate",
        bundle_dir=bundle.resolve(),
        sample_size=100,
        seed="daily-check",
        comparison_result=comparison_payload,
    )


@pytest.mark.parametrize("dataset", ["ashare_daily_raw", "ashare_daily_qfq"])
def test_stock_update_delegates_existing_bundle_without_automatic_qlib(tmp_path, monkeypatch, capsys, dataset):
    import quant_qmt_data as provider
    cli = load_module()
    calls = []
    def publish(**kwargs):
        calls.append(kwargs)
        return tmp_path / "raw", tmp_path / "qfq"
    monkeypatch.setattr(provider, "publish_ashare_bundle", publish)
    monkeypatch.setattr(cli, "build_direct_provider_descriptor", lambda **_: pytest.fail("update rebuilt Qlib"))
    assert cli.main(["--root", str(tmp_path), "update", dataset,
                     "--bundle-dir", str(tmp_path / "bundle")]) == 0
    assert calls == [{"quant_data_dir": tmp_path, "bundle_dir": tmp_path / "bundle"}]
    assert json.loads(capsys.readouterr().out)["backend"] == "quant_qmt_data"
    assert not (tmp_path / cli.ASHARE_UPDATE_TRANSACTION_PATH).exists()


@pytest.mark.parametrize("domain", ["contract", "bridge", "bundle", "etf"])
def test_updater_domain_failure_is_nonzero_without_a_second_publisher(tmp_path, monkeypatch, capsys, domain):
    import quant_qmt_data as provider
    from quant_qmt_data.download_baostock_v2 import BaostockV2Error
    from quant_qmt_data.download_cn_etf_data import CnEtfDownloadError
    cli = load_module()
    error = {"contract": provider.DataContractError, "bridge": provider.MiniQMTError,
             "bundle": BaostockV2Error, "etf": CnEtfDownloadError}[domain]
    def fail(**kwargs):
        raise error("bounded supplier or input failure")
    monkeypatch.setattr(provider, "update_ashare_snapshot", fail)
    monkeypatch.setattr(provider, "update_cn_etf_snapshot", fail)
    dataset = "cn_etf_daily" if domain == "etf" else "ashare_daily_qfq"
    assert cli.main(["--root", str(tmp_path), "update", dataset]) == 2
    assert "bounded supplier or input failure" in capsys.readouterr().err
    assert not (tmp_path / "DATA_MANIFEST.json").exists()


@pytest.mark.parametrize("state,expected", [("no_gap", 0), ("prepared", 0), ("published", 0),
                                            ("partial", 2), ("blocked", 2), ("failed", 2)])
def test_etf_cli_uses_this_provider_operation_result(tmp_path, monkeypatch, capsys, state, expected):
    import quant_qmt_data as provider
    cli = load_module()
    def update(**kwargs):
        kwargs["_result"].update(status=state, failed_symbols=["SZ159915"] if expected else [])
        return tmp_path / "snapshot"
    monkeypatch.setattr(provider, "update_cn_etf_snapshot", update)
    assert cli.main(["--root", str(tmp_path), "update", "cn_etf_daily"]) == expected
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == state and result["backend"] == "quant_qmt_data"
    assert bool(result["failed_symbols"]) == bool(expected)


def test_old_compatibility_refresh_is_not_silently_ignored(tmp_path, capsys):
    cli = load_module()
    assert cli.main(["--root", str(tmp_path), "update", "ashare_daily_qfq",
                     "--refresh-legacy-compatibility"]) == 2
    assert "does not rebuild legacy" in capsys.readouterr().err
