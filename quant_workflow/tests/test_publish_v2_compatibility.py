import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "publish_v2_compatibility.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("publish_v2_compatibility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_legacy_frame_preserves_v1_schema_and_unit_factor() -> None:
    module = load_module()
    source = pd.DataFrame(
        {
            "date": pd.Series(["2026-07-23", "2026-07-24"], dtype="datetime64[us]"),
            "open": [10.0, 10.1],
            "high": [10.3, 10.4],
            "low": [9.9, 10.0],
            "close": [10.2, 10.3],
            "preclose": [9.8, 10.2],
            "volume": [1000, 1200],
            "amount": [10_100.0, 12_300.0],
            "turn": [0.5, 0.6],
            "pctChg": [4.0, 0.98],
            "tradestatus": [1, 1],
            "isST": [0, 0],
            "symbol": ["SH600000", "SH600000"],
            "source_id": ["baostock_raw_x_foreAdjustFactor"] * 2,
            "factor": [0.5, 1.0],
        }
    )

    result = module.to_v1_frame(source)

    assert list(result.columns) == module.V1_PARQUET_COLUMNS
    assert str(result["date"].dtype) == "datetime64[ns]"
    assert result["factor"].tolist() == [1.0, 1.0]
    assert result["close"].tolist() == [10.2, 10.3]


def test_build_v1_parquet_output_uses_six_digit_file_names(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24"]),
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [1000],
            "amount": [10_100.0],
            "turn": [0.5],
            "pctChg": [1.0],
            "factor": [0.75],
        }
    ).to_parquet(source / "SH600000.parquet", index=False)

    summary = module.build_v1_parquet_output(
        source,
        target,
        upstream={"dataset_id": "ashare_daily_qfq", "snapshot_id": "snap-1"},
    )

    output = target / "600000.parquet"
    assert output.is_file()
    assert not (target / "SH600000.parquet").exists()
    assert summary["symbol_count"] == 1
    assert pd.read_parquet(output)["factor"].tolist() == [1.0]
    assert (target / ".metadata" / "COMPATIBILITY_MANIFEST.json").is_file()


def test_build_v1_parquet_output_rejects_symbol_collision(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "source"
    source.mkdir()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-24"]),
            "open": [10.0],
            "high": [10.2],
            "low": [9.9],
            "close": [10.1],
            "volume": [1000],
        }
    )
    frame.to_parquet(source / "SH600000.parquet", index=False)
    frame.to_parquet(source / "SZ600000.parquet", index=False)

    with pytest.raises(ValueError, match="duplicate V1 symbol"):
        module.build_v1_parquet_output(
            source,
            tmp_path / "target",
            upstream={"dataset_id": "ashare_daily_qfq", "snapshot_id": "snap-1"},
        )


def _artifact_tree(root: Path, value: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in ("merged_parquet", "merged_qlib", "cleaned_parquet", "cleaned_qlib"):
        path = root / name
        path.mkdir(parents=True)
        (path / "marker.txt").write_text(value, encoding="utf-8")
        paths[name] = path
    return paths


def test_first_publish_archives_v1_as_old_and_promotes_candidates(tmp_path: Path) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {name: path.with_name(path.name + "_old") for name, path in targets.items()}

    result = module.publish_v1_contract(
        candidates=candidates,
        targets=targets,
        old_targets=old_targets,
        backup_root=tmp_path / "backup",
        after_publish=lambda: None,
    )

    assert result["status"] == "published"
    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v2"
        assert (old_targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1"


def test_repeat_publish_preserves_old_and_moves_displaced_current_to_backup(
    tmp_path: Path,
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2-next")
    targets = _artifact_tree(tmp_path / "targets", "v2-current")
    old_targets = _artifact_tree(tmp_path / "old", "v1-old")
    backup_root = tmp_path / "backup"

    result = module.publish_v1_contract(
        candidates=candidates,
        targets=targets,
        old_targets=old_targets,
        backup_root=backup_root,
        after_publish=lambda: None,
    )

    assert result["status"] == "published"
    assert result["archived_as_old"] == []
    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v2-next"
        assert (old_targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1-old"
        assert (backup_root / name / "marker.txt").read_text(encoding="utf-8") == "v2-current"


def test_publish_refuses_when_another_compatibility_publish_holds_the_lock(
    tmp_path: Path,
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {
        name: path.with_name(path.name + "_old") for name, path in targets.items()
    }
    lock_path = tmp_path / "compatibility.lock"
    with lock_path.open("a+b") as handle:
        module.lock_file(handle, exclusive=True, blocking=False)
        try:
            with pytest.raises(RuntimeError, match="already in progress"):
                module.publish_v1_contract(
                    candidates=candidates,
                    targets=targets,
                    old_targets=old_targets,
                    backup_root=tmp_path / "backup",
                    after_publish=lambda: None,
                    lock_path=lock_path,
                )
        finally:
            module.unlock_file(handle)


def test_publish_refuses_when_previous_rollback_failure_is_unresolved(
    tmp_path: Path,
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {
        name: path.with_name(path.name + "_old") for name, path in targets.items()
    }
    lock_path = tmp_path / "locks" / "compatibility.lock"
    lock_path.parent.mkdir(parents=True)
    (lock_path.parent / "v1_compatibility_rollback_failed.json").write_text(
        '{"status":"rollback_failed"}', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="unresolved rollback failure"):
        module.publish_v1_contract(
            candidates=candidates,
            targets=targets,
            old_targets=old_targets,
            backup_root=tmp_path / "backup",
            after_publish=lambda: None,
            lock_path=lock_path,
        )

    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1"
        assert (candidates[name] / "marker.txt").read_text(encoding="utf-8") == "v2"


def test_publish_rolls_back_all_directories_when_manifest_sync_fails(
    tmp_path: Path,
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {name: path.with_name(path.name + "_old") for name, path in targets.items()}

    def fail() -> None:
        raise RuntimeError("sync failed")

    with pytest.raises(RuntimeError, match="sync failed"):
        module.publish_v1_contract(
            candidates=candidates,
            targets=targets,
            old_targets=old_targets,
            backup_root=tmp_path / "backup",
            after_publish=fail,
        )

    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1"
        assert (candidates[name] / "marker.txt").read_text(encoding="utf-8") == "v2"
        assert not old_targets[name].exists()


def test_publish_retries_a_transient_rollback_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {
        name: path.with_name(path.name + "_old") for name, path in targets.items()
    }
    original_replace = module.Path.replace
    failed_once = False

    def flaky_replace(path: Path, target: Path):
        nonlocal failed_once
        if (
            path == targets["cleaned_qlib"]
            and Path(target) == candidates["cleaned_qlib"]
            and not failed_once
        ):
            failed_once = True
            raise OSError("transient rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(module.Path, "replace", flaky_replace)

    with pytest.raises(RuntimeError, match="sync failed"):
        module.publish_v1_contract(
            candidates=candidates,
            targets=targets,
            old_targets=old_targets,
            backup_root=tmp_path / "backup",
            after_publish=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
        )

    assert failed_once
    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1"
        assert (candidates[name] / "marker.txt").read_text(encoding="utf-8") == "v2"
        assert not old_targets[name].exists()


def test_rollback_marker_write_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    targets = _artifact_tree(tmp_path / "targets", "v1")
    old_targets = {
        name: path.with_name(path.name + "_old") for name, path in targets.items()
    }
    original_replace = module.Path.replace

    def fail_rollback(path: Path, target: Path):
        if (
            path == targets["cleaned_qlib"]
            and Path(target) == candidates["cleaned_qlib"]
        ):
            raise OSError("persistent rollback failure")
        return original_replace(path, target)

    monkeypatch.setattr(module.Path, "replace", fail_rollback)
    monkeypatch.setattr(
        module,
        "_atomic_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            PermissionError("marker is not writable")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        module.publish_v1_contract(
            candidates=candidates,
            targets=targets,
            old_targets=old_targets,
            backup_root=tmp_path / "backup",
            after_publish=lambda: (_ for _ in ()).throw(RuntimeError("sync failed")),
        )

    assert "PermissionError" in str(exc_info.value)
    assert "RuntimeError" in str(exc_info.value)
    assert "PermissionError" in capsys.readouterr().err


def test_publish_summary_failure_rolls_back_directories_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    run_root = quant_data / "runtime" / "v2_compatibility" / "runs" / "run-001"
    candidates = _artifact_tree(run_root / "staging", "v2")
    targets = _artifact_tree(tmp_path / "official", "v1")
    old_targets = {
        name: path.with_name(path.name + "_old") for name, path in targets.items()
    }
    stock_lists = quant_data / "stock_lists.json"
    stock_lists.parent.mkdir(parents=True, exist_ok=True)
    stock_lists.write_text("{}", encoding="utf-8")
    manifest_path = quant_data / "DATA_MANIFEST.json"
    manifest_path.write_text('{"state":"old"}', encoding="utf-8")
    source_manifest = {
        "snapshot_id": "qfq-001",
        "content_sha256": "a" * 64,
    }
    candidate = {
        "run_id": "run-001",
        "run_root": str(run_root),
        "source_snapshot": source_manifest,
        "stock_lists_sha256": module._sha256_file(stock_lists),
        "candidates": {name: str(path) for name, path in candidates.items()},
    }
    monkeypatch.setattr(
        module,
        "resolve_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=source_manifest),
    )
    monkeypatch.setattr(
        module,
        "validate_compatibility_candidates",
        lambda *_args, **_kwargs: {"status": "pass"},
    )
    monkeypatch.setattr(module, "_official_paths", lambda _root: (targets, old_targets))

    def sync_manifest(**_kwargs):
        manifest_path.write_text('{"state":"new"}', encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "sync_data_manifest",
        SimpleNamespace(sync_data_manifest=sync_manifest),
    )
    original_atomic_json = module._atomic_json

    def fail_summary(path: Path, payload) -> None:
        if path.name == "publish_summary.json":
            raise OSError("summary write failed")
        original_atomic_json(path, payload)

    monkeypatch.setattr(module, "_atomic_json", fail_summary)

    with pytest.raises(OSError, match="summary write failed"):
        module._publish_candidate_result(candidate, quant_data_root=quant_data)

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {"state": "old"}
    for name in targets:
        assert (targets[name] / "marker.txt").read_text(encoding="utf-8") == "v1"
        assert (candidates[name] / "marker.txt").read_text(encoding="utf-8") == "v2"


def test_publish_candidate_reports_publish_and_file_restore_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    run_root = quant_data / "runtime" / "v2_compatibility" / "runs" / "run-001"
    stock_lists = quant_data / "stock_lists.json"
    stock_lists.parent.mkdir(parents=True, exist_ok=True)
    stock_lists.write_text("{}", encoding="utf-8")
    source_manifest = {
        "snapshot_id": "qfq-001",
        "content_sha256": "a" * 64,
    }
    candidate = {
        "run_id": "run-001",
        "run_root": str(run_root),
        "source_snapshot": source_manifest,
        "stock_lists_sha256": module._sha256_file(stock_lists),
        "candidates": {},
    }
    publish_error = RuntimeError("publication failed")
    restore_error = OSError("tracked file restore failed")

    monkeypatch.setattr(
        module,
        "resolve_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=source_manifest),
    )
    monkeypatch.setattr(
        module,
        "validate_compatibility_candidates",
        lambda *_args, **_kwargs: {"status": "pass"},
    )

    def fail_publish(**_kwargs):
        raise publish_error

    def fail_restore(_state):
        raise restore_error

    monkeypatch.setattr(module, "publish_v1_contract", fail_publish)
    monkeypatch.setattr(module, "_restore_file_state", fail_restore)

    with pytest.raises(RuntimeError) as exc_info:
        module._publish_candidate_result(candidate, quant_data_root=quant_data)

    message = str(exc_info.value)
    assert "RuntimeError" in message
    assert "OSError" in message
    assert exc_info.value.__cause__ is restore_error
    assert restore_error.__context__ is publish_error


def test_restore_file_state_fsyncs_restored_file_and_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    path = tmp_path / "DATA_MANIFEST.json"
    path.write_bytes(b"new")
    fsync_calls: list[int] = []
    monkeypatch.setattr(module.os, "fsync", fsync_calls.append)

    module._restore_file_state({path: b"old"})

    assert path.read_bytes() == b"old"
    assert len(fsync_calls) == 2


def test_restore_file_state_attempts_every_file_and_reports_all_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    paths = [
        tmp_path / "DATA_MANIFEST.json",
        tmp_path / "README.md",
        tmp_path / "parquet" / "ashare_1d" / "README.md",
        tmp_path / "qlib" / "README.md",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new")

    real_replace = module.os.replace
    failed_paths = {paths[0], paths[2]}
    attempted_paths: list[Path] = []
    raised_errors: list[OSError] = []

    def fail_selected_restores(source, target) -> None:
        target_path = Path(target)
        attempted_paths.append(target_path)
        if target_path in failed_paths:
            error = OSError(f"cannot restore {target_path}")
            raised_errors.append(error)
            raise error
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", fail_selected_restores)

    with pytest.raises(OSError) as exc_info:
        module._restore_file_state({path: b"old" for path in paths})

    assert attempted_paths == paths
    assert paths[0].read_bytes() == b"new"
    assert paths[1].read_bytes() == b"old"
    assert paths[2].read_bytes() == b"new"
    assert paths[3].read_bytes() == b"old"
    message = str(exc_info.value)
    assert str(paths[0]) in message
    assert str(paths[2]) in message
    assert exc_info.value.__cause__ is raised_errors[0]


def test_restore_file_state_aggregates_delete_failures_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    paths = [
        tmp_path / "DATA_MANIFEST.json",
        tmp_path / "README.md",
        tmp_path / "parquet" / "ashare_1d" / "README.md",
        tmp_path / "qlib" / "README.md",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"new")

    real_unlink = Path.unlink
    target_paths = set(paths)
    failed_paths = {paths[0], paths[2]}
    attempted_deletions: list[Path] = []
    raised_errors: list[OSError] = []

    def fail_selected_deletions(self: Path, missing_ok: bool = False) -> None:
        if self in target_paths:
            attempted_deletions.append(self)
            if self in failed_paths:
                error = OSError(f"cannot delete {self}")
                raised_errors.append(error)
                raise error
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_selected_deletions)

    with pytest.raises(OSError) as exc_info:
        module._restore_file_state(
            {
                paths[0]: None,
                paths[1]: b"old",
                paths[2]: None,
                paths[3]: None,
            }
        )

    assert attempted_deletions == [paths[0], paths[2], paths[3]]
    assert paths[0].read_bytes() == b"new"
    assert paths[1].read_bytes() == b"old"
    assert paths[2].read_bytes() == b"new"
    assert not paths[3].exists()
    message = str(exc_info.value)
    assert str(paths[0]) in message
    assert str(paths[2]) in message
    assert exc_info.value.__cause__ is raised_errors[0]


def test_restore_file_state_preserves_primary_error_when_cleanup_also_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    path = tmp_path / "DATA_MANIFEST.json"
    path.write_bytes(b"new")
    primary_error = OSError("replace failed")
    cleanup_error = OSError("temporary cleanup failed")
    real_unlink = Path.unlink

    def fail_replace(_source, _target) -> None:
        raise primary_error

    def fail_temporary_cleanup(self: Path, missing_ok: bool = False) -> None:
        if self.name.startswith(f".{path.name}.") and self.name.endswith(".restore"):
            raise cleanup_error
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_temporary_cleanup)

    with pytest.raises(OSError) as exc_info:
        module._restore_file_state({path: b"old"})

    message = str(exc_info.value)
    assert "replace failed" in message
    assert "temporary cleanup failed" in message
    assert exc_info.value.__cause__ is primary_error


def test_cli_converts_runtime_error_to_controlled_failure(capsys) -> None:
    module = load_module()

    def fail(**_kwargs):
        raise RuntimeError("lock conflict")

    module.rebuild_and_publish_v1_compatibility = fail

    assert module.main(["--publish"]) == 2
    assert "RuntimeError: lock conflict" in capsys.readouterr().err


def _write_benchmark_features(provider: Path, symbols: tuple[str, ...]) -> None:
    calendar = [
        line
        for line in (provider / "calendars" / "day.txt").read_text().splitlines()
        if line
    ]
    for symbol in symbols:
        feature_dir = provider / "features" / symbol.lower()
        feature_dir.mkdir(parents=True)
        for field in ("open", "high", "low", "close", "volume", "factor"):
            values = (
                np.ones(len(calendar), dtype="<f4")
                if field != "volume"
                else np.full(len(calendar), 1000.0, dtype="<f4")
            )
            with (feature_dir / f"{field}.day.bin").open("wb") as stream:
                np.array([0], dtype="<f4").tofile(stream)
                values.tofile(stream)


def test_build_candidates_runs_fresh_scan_and_builds_all_v1_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    snapshot = quant_data / "source-snapshot"
    bars = snapshot / "bars"
    bars.mkdir(parents=True)
    common = {
        "date": pd.to_datetime(["2026-07-23", "2026-07-24"]),
        "open": [10.0, 10.2],
        "high": [10.2, 10.4],
        "low": [9.9, 10.1],
        "volume": [1000.0, 1200.0],
        "factor": [0.5, 1.0],
    }
    pd.DataFrame({**common, "close": [10.1, 10.3]}).to_parquet(
        bars / "SH600000.parquet", index=False
    )
    pd.DataFrame({**common, "high": [10.2, 20.4], "close": [10.1, 20.3]}).to_parquet(
        bars / "SZ000001.parquet", index=False
    )
    source_manifest = {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-001",
        "content_sha256": "a" * 64,
        "price_basis": "qfq",
        "latest_trade_date": "2026-07-24",
        "symbol_count": 2,
        "row_count": 4,
    }
    monkeypatch.setattr(
        module,
        "resolve_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(
            snapshot_path=snapshot,
            manifest=source_manifest,
        ),
    )
    (quant_data / "stock_lists.json").parent.mkdir(parents=True, exist_ok=True)
    (quant_data / "stock_lists.json").write_text(
        json.dumps({"csi300": ["600000.XSHG"]}), encoding="utf-8"
    )

    def fake_add_benchmark(provider: Path, **_kwargs):
        _write_benchmark_features(provider, tuple(module.BENCHMARK_SYMBOLS))
        return {"status": "completed", "source": "test"}

    monkeypatch.setattr(module, "add_benchmark", fake_add_benchmark)

    result = module.build_compatibility_candidates(
        quant_data_root=quant_data,
        run_id="test-run",
    )

    assert set(result["candidates"]) == set(module.ARTIFACT_NAMES)
    for path in result["candidates"].values():
        assert Path(path).is_dir()
    merged = Path(result["candidates"]["merged_parquet"])
    cleaned = Path(result["candidates"]["cleaned_parquet"])
    assert str(pd.read_parquet(merged / "600000.parquet")["date"].dtype) == "datetime64[ns]"
    assert pd.read_parquet(merged / "600000.parquet")["factor"].tolist() == [1.0, 1.0]
    assert pd.isna(pd.read_parquet(cleaned / "000001.parquet").iloc[-1]["close"])
    events = pd.read_csv(Path(result["reports_dir"]) / "data_anomaly_events_latest.csv")
    assert "abs_return_gt_50pct" in set(events["issue_type"])
    for name in ("merged_qlib", "cleaned_qlib"):
        provider = Path(result["candidates"][name])
        assert (provider / "instruments" / "csi300.txt").is_file()
        assert not (provider / "instruments" / "csi800.txt").exists()
    assert result["validation"]["status"] == "pass"
    assert result["stock_lists_sha256"] == module._sha256_file(
        quant_data / "stock_lists.json"
    )


def test_publish_candidate_rejects_changed_stock_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    stock_lists = quant_data / "stock_lists.json"
    stock_lists.parent.mkdir(parents=True)
    stock_lists.write_text('{"csi300":["600000.XSHG"]}', encoding="utf-8")
    source = {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-001",
        "content_sha256": "a" * 64,
        "price_basis": "qfq",
    }
    monkeypatch.setattr(
        module,
        "resolve_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=source),
    )
    candidate = {
        "run_id": "candidate-001",
        "run_root": str(
            quant_data
            / "runtime"
            / "v2_compatibility"
            / "runs"
            / "candidate-001"
        ),
        "source_snapshot": source,
        "stock_lists_sha256": "b" * 64,
        "candidates": {},
    }

    with pytest.raises(ValueError, match="stock list changed"):
        module._publish_candidate_result(candidate, quant_data_root=quant_data)


def test_publish_candidate_refuses_when_dataset_publication_is_locked(
    tmp_path: Path,
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"

    with module.dataset_publication_lock(quant_data):
        with pytest.raises(module.DataContractError, match="already in progress"):
            module._publish_candidate_result({}, quant_data_root=quant_data)


def test_publish_candidate_rejects_stale_qfq_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    quant_data = tmp_path / "quant_data"
    current = {
        "dataset_id": "ashare_daily_qfq",
        "snapshot_id": "qfq-current",
        "content_sha256": "a" * 64,
        "price_basis": "qfq",
    }
    monkeypatch.setattr(
        module,
        "resolve_dataset",
        lambda *_args, **_kwargs: SimpleNamespace(manifest=current),
    )
    candidate = {
        "source_snapshot": {
            **current,
            "snapshot_id": "qfq-old",
        }
    }

    with pytest.raises(ValueError, match="source snapshot changed"):
        module._publish_candidate_result(candidate, quant_data_root=quant_data)


def test_atomic_parquet_fsyncs_file_and_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    calls: list[int] = []
    real_fsync = os.fsync

    def record_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(module.os, "fsync", record_fsync)
    target = tmp_path / "data.parquet"
    module._atomic_parquet(pd.DataFrame({"value": [1]}), target)

    assert target.is_file()
    assert len(calls) >= 2


def test_validate_candidates_rejects_non_unit_qlib_factor(tmp_path: Path) -> None:
    module = load_module()
    candidates = _artifact_tree(tmp_path / "candidates", "v2")
    provider = candidates["merged_qlib"]
    (provider / "calendars").mkdir(exist_ok=True)
    (provider / "calendars" / "day.txt").write_text("2026-07-24\n")
    (provider / "instruments").mkdir(exist_ok=True)
    (provider / "instruments" / "all.txt").write_text(
        "SH600000\t2026-07-24\t2026-07-24\n"
    )
    (provider / "instruments" / "csi300.txt").write_text(
        "SH600000\t2026-07-24\t2026-07-24\n"
    )
    feature = provider / "features" / "sh600000"
    feature.mkdir(parents=True)
    with (feature / "factor.day.bin").open("wb") as stream:
        np.array([0.0, 0.5], dtype="<f4").tofile(stream)

    with pytest.raises(ValueError, match="unit factor"):
        module.validate_qlib_output(provider, require_benchmarks=False)
