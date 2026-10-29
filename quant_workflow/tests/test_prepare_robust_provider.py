from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_robust_provider.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_robust_provider", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_manifest(root: Path, version: int = 1) -> Path:
    path = root / "DATA_MANIFEST.json"
    path.write_text(json.dumps({"version": version}) + "\n", encoding="utf-8")
    return path


def test_normalize_intervals_merges_adjacent_csi300_and_csi500_membership() -> None:
    module = load_module()
    intervals = pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000", "SZ000001"],
            "index_code": ["000300.XSHG", "000905.XSHG", "000300.XSHG"],
            "start_date": ["2023-01-02", "2023-03-01", "2023-01-02"],
            "end_date": ["2023-02-28", "2023-06-30", "2023-06-30"],
        }
    )
    calendar = pd.bdate_range("2023-01-02", "2023-06-30")

    normalized = module.normalize_union_intervals(intervals, calendar)

    sh = normalized[normalized["symbol"] == "SH600000"]
    assert len(sh) == 1
    assert sh.iloc[0]["start_date"] == pd.Timestamp("2023-01-02")
    assert sh.iloc[0]["end_date"] == pd.Timestamp("2023-06-30")


def test_prepare_runtime_provider_keeps_canonical_provider_untouched(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "instruments").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    original = canonical / "instruments" / "all.txt"
    original.write_text("SH600000\t2023-01-02\t2023-01-03\n", encoding="utf-8")
    feature = canonical / "features" / "sample.bin"
    feature.write_bytes(b"before")
    manifest = write_manifest(tmp_path)
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "index_code": ["000300.XSHG"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)

    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )

    provider = Path(result["provider_uri"])
    assert provider != canonical
    assert not (provider / "calendars").is_symlink()
    assert not (provider / "features").is_symlink()
    assert (provider / "instruments" / "csi800.txt").read_text(encoding="utf-8") == (
        "SH600000\t2023-01-02\t2023-01-03\n"
    )
    assert original.read_text(encoding="utf-8") == "SH600000\t2023-01-02\t2023-01-03\n"
    feature.write_bytes(b"after")
    (canonical / "calendars" / "day.txt").write_text("2023-01-02\n", encoding="utf-8")
    assert (provider / "features" / "sample.bin").read_bytes() == b"before"
    assert (provider / "calendars" / "day.txt").read_text(
        encoding="utf-8"
    ) == "2023-01-02\n2023-01-03\n"


def test_prepare_runtime_provider_uses_qlib_descriptor_without_copying_market_data(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "direct-provider"
    canonical.mkdir()
    descriptor = canonical / "PROVIDER_DESCRIPTOR.json"
    descriptor.write_text("{}\n", encoding="utf-8")
    manifest = write_manifest(tmp_path)
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "index_code": ["000300.XSHG"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    context = type(
        "Context",
        (),
        {
            "descriptor": {
                "cleaning": {"mode": "cleaned"},
                "dataset": {"snapshot_id": "qfq-001"},
            },
            "calendar": pd.DatetimeIndex(
                ["2023-01-02", "2023-01-03"]
            ),
            "membership": pd.read_parquet(intervals),
            "file_index": pd.DataFrame(
                {
                    "symbol": ["SH600000"],
                    "first_date": ["2023-01-02"],
                    "last_date": ["2023-01-03"],
                }
            ),
        },
    )()
    module._read_provider_context = lambda *_args, **_kwargs: context

    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )

    assert Path(result["provider_uri"]) == canonical.resolve()
    assert result["provider_backend"] == "quant_data"
    assert not (tmp_path / "runtime" / "providers").exists()
    latest = json.loads(
        (tmp_path / "runtime" / "latest_provider.json").read_text(encoding="utf-8")
    )
    assert latest["provider_uri"] == str(canonical.resolve())


def test_prepare_runtime_provider_is_content_addressed_and_reusable(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    manifest = write_manifest(tmp_path)
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "index_code": ["000300.XSHG"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)

    first = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )
    second = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )

    assert first["provider_uri"] == second["provider_uri"]
    assert second["reused"] is True
    latest = json.loads(
        (tmp_path / "runtime" / "latest_provider.json").read_text(encoding="utf-8")
    )
    assert latest["provider_uri"] == first["provider_uri"]

    manifest.write_text('{"version": 2}\n', encoding="utf-8")
    third = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )
    assert third["provider_uri"] != first["provider_uri"]


def test_reuse_check_rejects_modified_feature_even_after_restoring_read_only_mode(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)
    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )
    provider = Path(result["provider_uri"])
    feature = provider / "features" / "sample.bin"
    feature.chmod(0o644)
    feature.write_bytes(b"tampered")
    feature.chmod(0o444)

    assert module._snapshot_is_reusable(provider, result["provider_key"]) is False


@pytest.mark.parametrize("writable_mode", [0o644, 0o464, 0o446], ids=["owner", "group", "other"])
def test_reuse_check_rejects_owner_group_or_other_writable_snapshot(
    tmp_path: Path, writable_mode: int
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)
    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )
    provider = Path(result["provider_uri"])
    feature = provider / "features" / "sample.bin"
    assert module._snapshot_is_reusable(provider, result["provider_key"]) is True
    feature.chmod(writable_mode)
    write_bits = feature.stat().st_mode & 0o222
    if writable_mode != 0o644 and not write_bits:
        pytest.skip("filesystem chmod does not expose the requested write bit")
    assert write_bits, "filesystem did not make the snapshot writable"

    assert module._snapshot_is_reusable(provider, result["provider_key"]) is False


def test_runtime_identity_cache_reuses_unchanged_hashes_and_detects_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    cache_type = getattr(module, "RuntimeProviderIdentityCache", None)
    assert cache_type is not None, "runtime provider identity cache is missing"
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)
    prepared = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=manifest,
    )
    provider = Path(prepared["provider_uri"])
    original_sha256 = module.file_sha256
    hashed: list[Path] = []

    def counted_sha256(path: Path) -> str:
        hashed.append(Path(path))
        return original_sha256(path)

    monkeypatch.setattr(module, "file_sha256", counted_sha256)
    cache = cache_type()

    first = cache(provider, prepared["provider_key"])
    first_hash_count = len(hashed)
    second = cache(provider, prepared["provider_key"])

    assert first == second
    assert first_hash_count > 0
    assert len(hashed) == first_hash_count

    feature = provider / "features" / "sample.bin"
    feature.chmod(0o644)
    feature.write_bytes(b"tampered")
    feature.chmod(0o444)

    assert cache(provider, prepared["provider_key"]) is None
    assert len(hashed) == first_hash_count + 1


def test_runtime_identity_cache_scans_provider_tree_once_per_check(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "features" / "sample.bin").write_bytes(b"original")
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    prepared = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=write_manifest(tmp_path),
    )
    provider = Path(prepared["provider_uri"])
    cache = module.RuntimeProviderIdentityCache()
    assert cache(provider, prepared["provider_key"]) is not None

    original_rglob = Path.rglob
    original_stat = Path.stat
    scan_count = 0
    follow_stat_count = 0

    def counted_rglob(path: Path, pattern: str):
        nonlocal scan_count
        if path == provider:
            scan_count += 1
        return original_rglob(path, pattern)

    def counted_stat(path: Path, *args, **kwargs):
        nonlocal follow_stat_count
        if (
            kwargs.get("follow_symlinks", True)
            and (path == provider or path.is_relative_to(provider))
        ):
            follow_stat_count += 1
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "rglob", counted_rglob)
    monkeypatch.setattr(Path, "stat", counted_stat)

    assert cache(provider, prepared["provider_key"]) is not None
    assert scan_count == 1
    assert follow_stat_count == 1


def test_main_can_write_status_to_reports_without_moving_provider_data(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    report_dir = tmp_path / "reports"
    manifest = write_manifest(tmp_path)

    return_code = module.main(
        [
            "--canonical-provider",
            str(canonical),
            "--intervals",
            str(intervals),
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--data-manifest",
            str(manifest),
            "--status-out-dir",
            str(report_dir),
        ]
    )

    assert return_code == 0
    assert (
        json.loads((report_dir / "provider_latest.json").read_text(encoding="utf-8"))[
            "status"
        ]
        == "completed"
    )


def test_prepare_runtime_provider_rejects_future_or_overlapping_intervals(
    tmp_path: Path,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    manifest = write_manifest(tmp_path)
    pd.DataFrame(
        {
            "symbol": ["SH600000", "SH600000"],
            "index_code": ["000300.XSHG", "000905.XSHG"],
            "start_date": ["2023-01-02", "2023-01-03"],
            "end_date": ["2023-01-03", "2023-01-04"],
        }
    ).to_parquet(intervals, index=False)

    try:
        module.prepare_runtime_provider(
            canonical_provider=canonical,
            intervals_path=intervals,
            runtime_root=tmp_path / "runtime",
            data_manifest_path=manifest,
        )
    except ValueError as exc:
        assert "calendar" in str(exc) or "overlap" in str(exc)
    else:
        raise AssertionError("invalid PIT intervals were accepted")


def test_read_only_failure_does_not_leave_reusable_writable_provider(
    tmp_path: Path, monkeypatch
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "features" / "sample.bin").write_bytes(b"data")
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)
    runtime = tmp_path / "runtime"
    original = module._make_tree_read_only

    monkeypatch.setattr(
        module,
        "_make_tree_read_only",
        lambda root: (_ for _ in ()).throw(OSError("chmod failed")),
    )
    try:
        module.prepare_runtime_provider(
            canonical_provider=canonical,
            intervals_path=intervals,
            runtime_root=runtime,
            data_manifest_path=manifest,
        )
    except OSError as exc:
        assert "chmod" in str(exc)
    else:
        raise AssertionError("read-only failure was ignored")

    assert not any((runtime / "providers").glob("[!.]*"))
    monkeypatch.setattr(module, "_make_tree_read_only", original)
    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=runtime,
        data_manifest_path=manifest,
    )
    assert result["reused"] is False


def test_provider_snapshot_rejects_symlinks_without_touching_link_target(
    tmp_path: Path,
    make_symlink,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    external = tmp_path / "external.bin"
    external.write_bytes(b"external")
    external.chmod(0o600)
    external_mode = external.stat().st_mode
    make_symlink(canonical / "features" / "linked.bin", external)
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)

    try:
        module.prepare_runtime_provider(
            canonical_provider=canonical,
            intervals_path=intervals,
            runtime_root=tmp_path / "runtime",
            data_manifest_path=manifest,
        )
    except ValueError as exc:
        assert "symlink" in str(exc).lower()
    else:
        raise AssertionError("canonical provider symlink was accepted")

    assert external.read_bytes() == b"external"
    assert external.stat().st_mode == external_mode


def test_reuse_check_rejects_provider_root_symlink(
    tmp_path: Path, make_symlink
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    (canonical / "features" / "sample.bin").write_bytes(b"canonical")
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=tmp_path / "runtime",
        data_manifest_path=write_manifest(tmp_path),
    )
    target = Path(result["provider_uri"])
    key = result["provider_key"]
    assert not target.is_symlink()
    assert module._snapshot_is_reusable(target, key) is True

    link = tmp_path / "provider-link"
    make_symlink(link, target, target_is_directory=True)

    assert module._snapshot_is_reusable(link, key) is False


def test_invalid_cached_root_symlink_is_unlinked_without_modifying_target(
    tmp_path: Path,
    make_symlink,
) -> None:
    module = load_module()
    canonical = tmp_path / "canonical"
    (canonical / "calendars").mkdir(parents=True)
    (canonical / "features").mkdir()
    feature = canonical / "features" / "sample.bin"
    feature.write_bytes(b"canonical")
    feature.chmod(0o600)
    feature_mode = feature.stat().st_mode
    (canonical / "calendars" / "day.txt").write_text(
        "2023-01-02\n2023-01-03\n", encoding="utf-8"
    )
    intervals = tmp_path / "membership_intervals.parquet"
    pd.DataFrame(
        {
            "symbol": ["SH600000"],
            "start_date": ["2023-01-02"],
            "end_date": ["2023-01-03"],
        }
    ).to_parquet(intervals, index=False)
    manifest = write_manifest(tmp_path)
    runtime = tmp_path / "runtime"
    key = module._provider_key(
        canonical.resolve(), intervals.resolve(), manifest.resolve()
    )
    cached = runtime / "providers" / key[:16]
    cached.parent.mkdir(parents=True)
    make_symlink(cached, canonical, target_is_directory=True)

    result = module.prepare_runtime_provider(
        canonical_provider=canonical,
        intervals_path=intervals,
        runtime_root=runtime,
        data_manifest_path=manifest,
    )

    assert Path(result["provider_uri"]).is_dir()
    assert not Path(result["provider_uri"]).is_symlink()
    assert feature.read_bytes() == b"canonical"
    assert feature.stat().st_mode == feature_mode
