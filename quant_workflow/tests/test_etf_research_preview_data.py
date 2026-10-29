from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "etf_research_preview_data.py"
)
LOCKED_SYMBOLS = ("SZ159915", "SH513100", "SH512890")


def load_module():
    spec = importlib.util.spec_from_file_location(
        "etf_research_preview_data", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_hash(snapshot: Path, manifest: dict[str, object]) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        candidate
        for candidate in snapshot.rglob("*")
        if candidate.is_file() and candidate.name != "DATASET_MANIFEST.json"
    ):
        digest.update(path.relative_to(snapshot).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    stable = dict(manifest)
    stable.pop("content_sha256", None)
    digest.update(_json_bytes(stable))
    return digest.hexdigest()


def _bar_frame(symbol: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": [symbol, symbol, symbol],
            "date": pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]),
            "open": [1.0, 1.1, 1.2],
            "high": [1.1, 1.2, 1.3],
            "low": [0.9, 1.0, 1.1],
            "close": [1.05, 1.15, 1.25],
            "volume": [1000.0, 1100.0, 1200.0],
            "adjustment": ["qfq", "qfq", "qfq"],
            "source_id": ["fixture.qfq", "fixture.qfq", "fixture.qfq"],
        }
    )


def _refresh_publication(root: Path, snapshot: Path, manifest: dict[str, object]) -> None:
    manifest["content_sha256"] = _content_hash(snapshot, manifest)
    manifest_path = snapshot / "DATASET_MANIFEST.json"
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path)
    snapshot_id = str(manifest["snapshot_id"])
    pointer = {
        "schema_version": 2,
        "dataset_id": "cn_etf_daily",
        "snapshot_id": snapshot_id,
        "snapshot_path": (
            "parquet/etf_1d/cn_etf_daily/snapshots/" + snapshot_id
        ),
        "manifest_sha256": manifest_sha,
        "published_at": "2026-08-07T13:59:39+08:00",
    }
    _write_json(root / "parquet/etf_1d/cn_etf_daily/LATEST.json", pointer)
    catalog_entry = {
        "dataset_id": "cn_etf_daily",
        "dataset_root": "parquet/etf_1d/cn_etf_daily",
        "latest_pointer": "parquet/etf_1d/cn_etf_daily/LATEST.json",
        "snapshot_id": snapshot_id,
        "snapshot_manifest_sha256": manifest_sha,
        "data_kind": "paired_daily_bars",
        "role": "canonical",
        "status": "ready",
        "price_basis": "qfq",
        "content_sha256": manifest["content_sha256"],
    }
    _write_json(
        root / "DATA_MANIFEST.json",
        {
            "schema_version": 2,
            "path_format": "relative_to_quant_data_root",
            "datasets": {"cn_etf_daily": catalog_entry},
        },
    )


def build_preview_snapshot(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    root = tmp_path / "quant_data"
    snapshot_id = "cn-etf-preview-fixture"
    snapshot = (
        root
        / "parquet/etf_1d/cn_etf_daily/snapshots"
        / snapshot_id
    )
    qfq_dir = snapshot / "qfq"
    qfq_dir.mkdir(parents=True)
    index_rows: list[dict[str, object]] = []
    for symbol in LOCKED_SYMBOLS:
        path = qfq_dir / f"{symbol}.parquet"
        frame = _bar_frame(symbol)
        frame.to_parquet(path, index=False)
        index_rows.append(
            {
                "symbol": symbol,
                "relative_path": f"qfq/{symbol}.parquet",
                "row_count": len(frame),
                "first_date": "2026-01-05",
                "last_date": "2026-01-07",
                "source_ids": '["fixture.qfq"]',
                "sha256": _sha256(path),
            }
        )
    pd.DataFrame(index_rows).to_parquet(
        snapshot / "QFQ_FILE_INDEX.parquet", index=False
    )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "dataset_id": "cn_etf_daily",
        "snapshot_id": snapshot_id,
        "data_kind": "paired_daily_bars",
        "role": "canonical",
        "status": "ready",
        "source": {"provider": "fixture"},
        "price_basis": "qfq",
        "source_retrieved_at": "2026-08-07T13:25:13+08:00",
        "adjustment": {
            "method": "provider_native_paired",
            "paired_raw_component_path": "raw",
            "factor_path": "factors",
        },
        "producer": {"script": "fixture", "version": "2"},
        "date_range": {"start": "2026-01-05", "end": "2026-01-07"},
        "latest_trade_date": "2026-01-07",
        "symbol_count": 3,
        "row_count": 9,
        "file_index_path": "QFQ_FILE_INDEX.parquet",
        "components": {
            "qfq": {
                "path": "qfq",
                "file_index_path": "QFQ_FILE_INDEX.parquet",
                "price_basis": "qfq",
                "symbol_count": 3,
                "row_count": 9,
                "date_range": {"start": "2026-01-05", "end": "2026-01-07"},
            },
            "raw": {"path": "raw"},
            "factors": {"path": "factors"},
        },
        "content_sha256": "",
    }
    _refresh_publication(root, snapshot, manifest)
    return root, snapshot, manifest


def test_resolver_returns_locked_qfq_bars_as_research_preview(tmp_path: Path) -> None:
    module = load_module()
    root, snapshot, manifest = build_preview_snapshot(tmp_path)

    result = module.resolve_etf_research_preview(root)

    assert result.status == "research_preview"
    assert result.dataset_id == "cn_etf_daily"
    assert result.snapshot_id == manifest["snapshot_id"]
    assert result.snapshot_path == snapshot.resolve()
    assert tuple(result.bars) == LOCKED_SYMBOLS
    assert all(len(frame) == 3 for frame in result.bars.values())
    assert result.formally_accepted is False
    assert result.usable_for_trading is False


def test_resolver_rejects_any_other_required_symbol_contract(tmp_path: Path) -> None:
    module = load_module()
    root, _, _ = build_preview_snapshot(tmp_path)

    with pytest.raises(ValueError, match="locked three-ETF contract"):
        module.resolve_etf_research_preview(
            root,
            required_symbols=("SZ159915", "SH513100"),
        )


def test_resolver_rejects_catalog_pointer_identity_mismatch(tmp_path: Path) -> None:
    module = load_module()
    root, _, _ = build_preview_snapshot(tmp_path)
    catalog_path = root / "DATA_MANIFEST.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["datasets"]["cn_etf_daily"]["snapshot_id"] = "another-snapshot"
    _write_json(catalog_path, catalog)

    with pytest.raises(module.DataContractError, match="catalog and latest pointer"):
        module.resolve_etf_research_preview(root)


def test_resolver_checks_full_snapshot_content_hash(tmp_path: Path) -> None:
    module = load_module()
    root, snapshot, _ = build_preview_snapshot(tmp_path)
    (snapshot / "unexpected.txt").write_text("mutation", encoding="utf-8")

    with pytest.raises(module.DataContractError, match="content hash mismatch"):
        module.resolve_etf_research_preview(root)


def test_resolver_checks_required_etf_hash_against_qfq_index(tmp_path: Path) -> None:
    module = load_module()
    root, snapshot, manifest = build_preview_snapshot(tmp_path)
    path = snapshot / "qfq/SZ159915.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "close"] = 9.99
    changed.to_parquet(path, index=False)
    _refresh_publication(root, snapshot, manifest)

    with pytest.raises(module.DataContractError, match="ETF file hash mismatch"):
        module.resolve_etf_research_preview(root)


def test_resolver_rejects_symlinked_required_etf_file(
    tmp_path: Path, make_symlink
) -> None:
    module = load_module()
    root, snapshot, manifest = build_preview_snapshot(tmp_path)
    target = snapshot / "qfq/SZ159915.parquet"
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    make_symlink(target, outside)
    _refresh_publication(root, snapshot, manifest)

    with pytest.raises(module.DataContractError, match="symlink"):
        module.resolve_etf_research_preview(root)


def test_resolver_rejects_missing_locked_symbol(tmp_path: Path) -> None:
    module = load_module()
    root, snapshot, manifest = build_preview_snapshot(tmp_path)
    index_path = snapshot / "QFQ_FILE_INDEX.parquet"
    index = pd.read_parquet(index_path)
    index = index.loc[index["symbol"] != "SH512890"].reset_index(drop=True)
    index.to_parquet(index_path, index=False)
    manifest["symbol_count"] = 2
    manifest["row_count"] = 6
    manifest["components"]["qfq"]["symbol_count"] = 2
    manifest["components"]["qfq"]["row_count"] = 6
    _refresh_publication(root, snapshot, manifest)

    with pytest.raises(module.DataContractError, match="missing required ETF"):
        module.resolve_etf_research_preview(root)
