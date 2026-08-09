from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_workflow.scripts import quant_data_contract as contract
from quant_workflow.scripts import update_cn_index_dataset as updater


SYMBOLS = list(updater.REQUIRED_SYMBOLS)


def _write_current_snapshot(root: Path) -> Path:
    staging = root / "staging" / "index-initial" / "cn-index-initial"
    bars = staging / "bars"
    bars.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
    for offset, symbol in enumerate(SYMBOLS):
        close = [100.0 + offset, 101.0 + offset, 102.0 + offset]
        pd.DataFrame(
            {
                "date": dates,
                "open": [value - 0.5 for value in close],
                "high": [value + 0.5 for value in close],
                "low": [value - 1.0 for value in close],
                "close": close,
                "volume": [1000.0, 1100.0, 1200.0],
                "source_id": "fixture.initial",
            }
        ).to_parquet(bars / f"{symbol}.parquet", index=False)
    contract.finalize_price_snapshot(
        staging,
        dataset_id="cn_index_daily",
        snapshot_id="cn-index-initial",
        role="canonical",
        status="ready",
        source={"provider": "fixture", "interface": "initial"},
        price_basis="derived",
        adjustment={"method": "none"},
        producer={"script": "test_update_cn_index_dataset.py", "version": "1"},
        source_retrieved_at="2026-01-06T18:00:00+08:00",
    )
    dataset_root = root / "parquet" / "index_1d" / "cn_index_daily"
    published = contract.publish_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=root,
    )
    contract.write_root_manifest(
        root,
        {
            "cn_index_daily": contract.catalog_entry_from_snapshot(
                published,
                quant_data_root=root,
                dataset_root=dataset_root,
            )
        },
    )
    return published


def _new_rows() -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(["2026-01-06", "2026-01-07"])
    result: dict[str, pd.DataFrame] = {}
    for offset, symbol in enumerate(SYMBOLS):
        close = [102.0 + offset, 103.0 + offset]
        result[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": [value - 0.5 for value in close],
                "high": [value + 0.5 for value in close],
                "low": [value - 1.0 for value in close],
                "close": close,
                "volume": [1200.0, 1300.0],
                "source_id": "fixture.incremental",
            }
        )
    return result


def _rows_through_friday() -> dict[str, pd.DataFrame]:
    dates = pd.to_datetime(["2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09"])
    result: dict[str, pd.DataFrame] = {}
    for offset, symbol in enumerate(SYMBOLS):
        close = [102.0 + offset, 103.0 + offset, 104.0 + offset, 105.0 + offset]
        result[symbol] = pd.DataFrame(
            {
                "date": dates,
                "open": [value - 0.5 for value in close],
                "high": [value + 0.5 for value in close],
                "low": [value - 1.0 for value in close],
                "close": close,
                "volume": [1200.0, 1300.0, 1400.0, 1500.0],
                "source_id": "fixture.incremental",
            }
        )
    return result


def test_update_publishes_one_formal_index_snapshot_without_stock_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    old = _write_current_snapshot(root)
    calls: list[tuple[str, ...]] = []

    def fake_eastmoney(_start: str, _end: str, **kwargs):
        calls.append(tuple(kwargs["symbols"]))
        return _new_rows()

    result = updater.update_cn_index_snapshot(
        quant_data_root=root,
        end_date="2026-01-07",
        eastmoney_download=fake_eastmoney,
        tencent_download=lambda *_args, **_kwargs: pytest.fail(
            "Tencent should not run after a valid Eastmoney response"
        ),
    )

    assert result["status"] == "published"
    assert calls == [tuple(SYMBOLS)]
    assert result["latest_trade_date"] == "2026-01-07"
    assert result["stock_symbols_downloaded"] == 0
    assert result["index_symbols_downloaded"] == len(SYMBOLS)
    latest = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    )
    assert latest.snapshot_path != old
    assert latest.manifest["latest_trade_date"] == "2026-01-07"
    assert len(pd.read_parquet(latest.snapshot_path / "bars" / "SH000300.parquet")) == 4
    assert not (root / "qlib").exists()


def test_failed_index_update_keeps_previous_latest_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    old = _write_current_snapshot(root)

    def incomplete(_start: str, _end: str, **_kwargs):
        rows = _new_rows()
        rows.pop(SYMBOLS[-1])
        return rows

    with pytest.raises(updater.DataContractError, match="symbol set|missing"):
        updater.update_cn_index_snapshot(
            quant_data_root=root,
            end_date="2026-01-07",
            eastmoney_download=incomplete,
            tencent_download=incomplete,
        )

    latest = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    )
    assert latest.snapshot_path == old
    assert latest.manifest["latest_trade_date"] == "2026-01-06"


def test_current_index_snapshot_does_not_download_again(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    _write_current_snapshot(root)

    def must_not_download(*_args, **_kwargs):
        pytest.fail("an up-to-date formal index snapshot must not download")

    result = updater.update_cn_index_snapshot(
        quant_data_root=root,
        end_date="2026-01-06",
        eastmoney_download=must_not_download,
        tencent_download=must_not_download,
    )

    assert result["status"] == "already_current"
    assert result["network_requests_for_symbols"] == []


def test_normalize_frame_accepts_unnamed_date_index() -> None:
    frame = pd.DataFrame(
        {
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [100.5],
            "volume": [1000.0],
        },
        index=["2026-01-06"],
    )

    normalized = updater._normalize_frame(frame, "SH000300")

    assert normalized["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-06"]


def test_weekend_through_uses_previous_exchange_session_and_does_not_republish(
    tmp_path: Path,
) -> None:
    root = tmp_path / "quant_data"
    _write_current_snapshot(root)

    first = updater.update_cn_index_snapshot(
        quant_data_root=root,
        end_date="2026-01-10",
        eastmoney_download=lambda *_args, **_kwargs: _rows_through_friday(),
        tencent_download=lambda *_args, **_kwargs: pytest.fail(
            "Tencent should not run after a valid Eastmoney response"
        ),
    )
    assert first["status"] == "published"
    assert first["latest_trade_date"] == "2026-01-09"

    latest_before = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    ).snapshot_path

    def must_not_download(*_args, **_kwargs):
        pytest.fail("a Saturday request must reuse the complete Friday snapshot")

    second = updater.update_cn_index_snapshot(
        quant_data_root=root,
        end_date="2026-01-10",
        eastmoney_download=must_not_download,
        tencent_download=must_not_download,
    )
    latest_after = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    ).snapshot_path

    assert second["status"] == "already_current"
    assert second["latest_trade_date"] == "2026-01-09"
    assert latest_after == latest_before


def test_weekday_source_that_is_still_stale_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    old = _write_current_snapshot(root)
    stale = {
        symbol: frame.iloc[[0]].copy()
        for symbol, frame in _new_rows().items()
    }

    with pytest.raises(updater.DataContractError, match="both formal index sources failed"):
        updater.update_cn_index_snapshot(
            quant_data_root=root,
            end_date="2026-01-07",
            eastmoney_download=lambda *_args, **_kwargs: stale,
            tencent_download=lambda *_args, **_kwargs: stale,
        )

    latest = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    )
    assert latest.snapshot_path == old


def test_stale_primary_source_falls_back_to_current_secondary(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    _write_current_snapshot(root)
    stale = {
        symbol: frame.iloc[[0]].copy()
        for symbol, frame in _new_rows().items()
    }
    calls: list[str] = []

    def stale_eastmoney(*_args, **_kwargs):
        calls.append("eastmoney")
        return stale

    def current_tencent(*_args, **_kwargs):
        calls.append("tencent")
        return _new_rows()

    result = updater.update_cn_index_snapshot(
        quant_data_root=root,
        end_date="2026-01-07",
        eastmoney_download=stale_eastmoney,
        tencent_download=current_tencent,
    )

    assert result["status"] == "published"
    assert result["source"]["provider"] == "tencent"
    assert calls == ["eastmoney", "tencent"]


def test_update_rejects_source_rows_after_requested_session(tmp_path: Path) -> None:
    root = tmp_path / "quant_data"
    old = _write_current_snapshot(root)
    future = _new_rows()
    for offset, symbol in enumerate(SYMBOLS):
        extra = pd.DataFrame(
            {
                "date": [pd.Timestamp("2026-01-08")],
                "open": [103.5 + offset],
                "high": [104.5 + offset],
                "low": [103.0 + offset],
                "close": [104.0 + offset],
                "volume": [1400.0],
                "source_id": ["fixture.future"],
            }
        )
        future[symbol] = pd.concat([future[symbol], extra], ignore_index=True)

    with pytest.raises(updater.DataContractError, match="both formal index sources failed"):
        updater.update_cn_index_snapshot(
            quant_data_root=root,
            end_date="2026-01-07",
            eastmoney_download=lambda *_args, **_kwargs: future,
            tencent_download=lambda *_args, **_kwargs: future,
        )

    latest = contract.resolve_dataset(
        "cn_index_daily", quant_data_root=root, require_basis="derived"
    )
    assert latest.snapshot_path == old
