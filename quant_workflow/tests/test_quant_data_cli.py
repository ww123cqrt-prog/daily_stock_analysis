from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

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


def write_pointer_state(module, root: Path, marker: str) -> dict[Path, bytes]:
    state: dict[Path, bytes] = {}
    for relative in module.ASHARE_UPDATE_POINTERS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        value = json.dumps({"marker": marker, "path": relative.as_posix()}).encode()
        path.write_bytes(value)
        state[path] = value
    return state


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
    qlib_v2 = quant_data / "qlib-v2-snapshot"
    builder_calls: dict[str, dict[str, object]] = {}

    def build_qlib(**kwargs):
        builder_calls["canonical"] = kwargs
        return qlib_v2

    qlib_builder = types.SimpleNamespace(
        build_qlib_v2=build_qlib
    )
    cleaned_v2 = quant_data / "qlib-cleaned-v2-snapshot"

    def build_cleaned(**kwargs):
        builder_calls["cleaned"] = kwargs
        return cleaned_v2

    cleaned_builder = types.SimpleNamespace(
        build_cleaned_qlib_v2=build_cleaned
    )

    with patch.dict(
        sys.modules,
        {
            "download_baostock_v2": fake,
            "build_qlib_v2": qlib_builder,
            "build_cleaned_qlib_v2": cleaned_builder,
        },
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
        "qlib_cleaned_v2": str(cleaned_v2),
        "qlib_v2": str(qlib_v2),
        "raw": str(raw),
    }
    assert (
        builder_calls["canonical"]["validated_cache"]
        is builder_calls["cleaned"]["validated_cache"]
    )
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
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: quant_data / "new-qlib"
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: quant_data / "new-cleaned"
        ),
    }

    with patch.dict(sys.modules, fake_modules):
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
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: quant_data / "qlib"
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: quant_data / "cleaned"
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=build_compatibility
        ),
    }

    with patch.dict(sys.modules, fake_modules):
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
        quant_data
        / "qlib"
        / "cn_data_ashare_v2"
        / "LATEST.json": b'{"old":"qlib"}',
        quant_data
        / "qlib"
        / "cn_data_ashare_cleaned_v2"
        / "LATEST.json": b'{"old":"cleaned"}',
    }
    for path, value in tracked.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    tracked_paths = list(tracked)

    def build_snapshots(**_kwargs):
        for path in tracked_paths[0:3]:
            path.write_text('{"new":"snapshot"}', encoding="utf-8")
        return quant_data / "new-raw", quant_data / "new-qfq"

    def build_qlib(**_kwargs):
        tracked_paths[3].write_text('{"new":"qlib"}', encoding="utf-8")
        return quant_data / "new-qlib"

    def fail_cleaned(**_kwargs):
        tracked_paths[4].write_text('{"new":"cleaned"}', encoding="utf-8")
        raise RuntimeError("cleaned build failed")

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
        "build_qlib_v2": types.SimpleNamespace(build_qlib_v2=build_qlib),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=fail_cleaned
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: None
        ),
    }

    with patch.dict(sys.modules, fake_modules):
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
    assert status["failed_step"] == "qlib_cleaned_v2"
    assert status["completed_steps"] == ["raw_qfq_v2", "qlib_v2"]
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
        quant_data / "qlib/cn_data_ashare_v2/LATEST.json": "qlib-new",
        quant_data / "qlib/cn_data_ashare_cleaned_v2/LATEST.json": "cleaned-new",
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
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: publish_pointer(
                list(pointers)[2], "qlib-new"
            )
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: publish_pointer(
                list(pointers)[3], "cleaned-new"
            )
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("compatibility failed")
            )
        ),
    }

    with patch.dict(sys.modules, fake_modules):
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
        "qlib_v2",
        "qlib_cleaned_v2",
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
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: quant_data / "new-qlib"
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("cleaned build failed")
            )
        ),
    }
    monkeypatch.setattr(
        module,
        "_write_update_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("status unavailable")),
    )

    with patch.dict(sys.modules, fake_modules):
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
        "build_qlib_v2": types.SimpleNamespace(
            build_qlib_v2=lambda **_kwargs: quant_data / "new-qlib"
        ),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **_kwargs: quant_data / "new-cleaned"
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

    with patch.dict(sys.modules, fake_modules):
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

    def build_qlib(**kwargs):
        assert_publication_lock("qlib_v2", **kwargs)
        return quant_data / "new-qlib"

    def build_compatibility(**kwargs):
        assert_publication_lock("compatibility", **kwargs)
        return {"status": "published"}

    fake_modules = {
        "download_baostock_v2": types.SimpleNamespace(
            build_snapshots_from_bundle=build_snapshots
        ),
        "build_qlib_v2": types.SimpleNamespace(build_qlib_v2=build_qlib),
        "build_cleaned_qlib_v2": types.SimpleNamespace(
            build_cleaned_qlib_v2=lambda **kwargs: (
                assert_publication_lock("qlib_cleaned_v2", **kwargs)
                or quant_data / "new-cleaned"
            )
        ),
        "publish_v2_compatibility": types.SimpleNamespace(
            rebuild_and_publish_v1_compatibility=build_compatibility
        ),
    }

    with patch.dict(sys.modules, fake_modules):
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
        "qlib_v2",
        "qlib_cleaned_v2",
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
