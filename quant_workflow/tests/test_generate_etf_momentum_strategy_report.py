from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "generate_etf_momentum_strategy_report.py"
)


def test_canonical_etf_strategy_report_generator_exists() -> None:
    assert SCRIPT_PATH.exists()


def load_module():
    spec = importlib.util.spec_from_file_location(
        "generate_etf_momentum_strategy_report", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def price_frame(symbol: str, closes: list[float], *, start: str = "2020-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "symbol": symbol,
            "date": dates,
            "open": closes,
            "close": closes,
        }
    )


def sample_prices(rows: int = 180) -> dict[str, pd.DataFrame]:
    first = [100.0 + index * 0.08 + (index % 9) * 0.12 for index in range(rows)]
    second = [100.0 + index * 0.05 + ((index + 4) % 13) * 0.10 for index in range(rows)]
    return {
        "SZ159915": price_frame("SZ159915", first),
        "SH513100": price_frame("SH513100", second),
    }


def load_config() -> dict[str, object]:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "etf_momentum_robustness.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_select_price_source_prefers_complete_discontinuity_free_qfq_data() -> None:
    module = load_module()
    raw = sample_prices()
    adjusted = {symbol: frame.iloc[1:].reset_index(drop=True) for symbol, frame in raw.items()}

    selected, quality = module.select_price_source(raw, adjusted, minimum_coverage=0.98)

    assert quality["status"] == "verified_qfq"
    assert quality["selected_source"] == "akshare.fund_etf_hist_em_qfq"
    assert quality["coverage_ratio"] >= 0.98
    assert quality["price_discontinuities"]["passed"] is True
    assert len(selected["SZ159915"]) == len(adjusted["SZ159915"])


def test_select_price_source_records_tencent_qfq_and_resolved_split() -> None:
    module = load_module()
    raw = sample_prices()
    adjusted = {symbol: frame.copy() for symbol, frame in raw.items()}
    manifest = {
        "corporate_actions": [
            {
                "symbol": "SH513100",
                "type": "share_split",
                "split_date": "2022-01-13",
                "ratio": "1:5",
            }
        ]
    }

    _, quality = module.select_price_source(
        raw,
        adjusted,
        adjusted_source=module.TENCENT_ADJUSTED_SOURCE,
        adjusted_manifest=manifest,
    )

    assert quality["status"] == "verified_qfq"
    assert quality["selected_source"] == module.TENCENT_ADJUSTED_SOURCE
    assert quality["resolved_corporate_actions"] == manifest["corporate_actions"]
    assert "unresolved_price_discontinuity" not in quality["evidence_gaps"]


def test_resolve_canonical_etf_bars_uses_v2_catalog_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    snapshot = tmp_path / "snapshot"
    bars = snapshot / "bars"
    bars.mkdir(parents=True)
    calls: list[tuple[str, str]] = []

    def fake_resolve(dataset_id, *, quant_data_root, require_basis):
        calls.append((dataset_id, require_basis))
        return SimpleNamespace(
            snapshot_path=snapshot,
            manifest={
                "dataset_id": dataset_id,
                "snapshot_id": "snapshot-001",
                "price_basis": "qfq",
            },
        )

    monkeypatch.setattr(module, "resolve_dataset", fake_resolve)

    resolved_bars, manifest = module.resolve_canonical_etf_bars(tmp_path)

    assert calls == [("etf_momentum_daily_qfq", "qfq")]
    assert resolved_bars == bars
    assert manifest["snapshot_id"] == "snapshot-001"


def test_run_uses_paired_canonical_raw_and_qfq_without_legacy_data(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    raw_snapshot = tmp_path / "raw-snapshot"
    qfq_snapshot = tmp_path / "qfq-snapshot"
    raw_bars = raw_snapshot / "bars"
    qfq_bars = qfq_snapshot / "bars"
    raw_bars.mkdir(parents=True)
    qfq_bars.mkdir(parents=True)
    for symbol, frame in sample_prices(180).items():
        frame.to_parquet(raw_bars / f"{symbol}.parquet", index=False)
        frame.to_parquet(qfq_bars / f"{symbol}.parquet", index=False)

    raw_manifest = {
        "dataset_id": "etf_momentum_daily_raw",
        "snapshot_id": "raw-001",
        "price_basis": "raw",
        "source": {"provider": "tencent"},
    }
    qfq_manifest = {
        "dataset_id": "etf_momentum_daily_qfq",
        "snapshot_id": "qfq-001",
        "price_basis": "qfq",
        "source": {"provider": "tencent"},
        "upstream_snapshots": [
            {"dataset_id": "etf_momentum_daily_raw", "snapshot_id": "raw-001"}
        ],
    }
    (raw_snapshot / "DATASET_MANIFEST.json").write_text(
        json.dumps(raw_manifest), encoding="utf-8"
    )
    (qfq_snapshot / "DATASET_MANIFEST.json").write_text(
        json.dumps(qfq_manifest), encoding="utf-8"
    )

    def fake_resolve(dataset_id, *, quant_data_root, require_basis):
        assert quant_data_root == tmp_path / "quant-data"
        if dataset_id == "etf_momentum_daily_raw":
            assert require_basis == "raw"
            return SimpleNamespace(snapshot_path=raw_snapshot, manifest=raw_manifest)
        assert dataset_id == "etf_momentum_daily_qfq"
        assert require_basis == "qfq"
        return SimpleNamespace(snapshot_path=qfq_snapshot, manifest=qfq_manifest)

    monkeypatch.setattr(module, "resolve_dataset", fake_resolve)
    reports_dir = tmp_path / "reports"

    module.run(
        data_dir=tmp_path / "missing-legacy-data",
        adjusted_data_dir=tmp_path / "missing-eastmoney-data",
        tencent_adjusted_data_dir=tmp_path / "missing-tencent-data",
        report_dir=reports_dir,
        config_path=(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "etf_momentum_robustness.yaml"
        ),
        generated_at="2026-07-26T01:00:00+08:00",
        quant_data_dir=tmp_path / "quant-data",
    )

    report = json.loads(
        (reports_dir / "strategy_report_latest.json").read_text(encoding="utf-8")
    )
    assert report["data_quality"]["status"] == "verified_qfq"
    assert report["data_quality"]["selected_source"] == module.TENCENT_ADJUSTED_SOURCE
    assert report["data_quality"]["raw_source"] == "tencent.catalog_raw"
    assert report["reproducibility"]["data_manifest_hash"]


def test_resolve_canonical_etf_pair_rejects_unpaired_qfq_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    module = load_module()
    raw_snapshot = tmp_path / "raw-snapshot"
    qfq_snapshot = tmp_path / "qfq-snapshot"
    (raw_snapshot / "bars").mkdir(parents=True)
    (qfq_snapshot / "bars").mkdir(parents=True)

    def fake_resolve(dataset_id, *, quant_data_root, require_basis):
        if dataset_id == "etf_momentum_daily_raw":
            return SimpleNamespace(
                snapshot_path=raw_snapshot,
                manifest={
                    "dataset_id": dataset_id,
                    "snapshot_id": "raw-001",
                    "price_basis": require_basis,
                    "source": {"provider": "tencent"},
                },
            )
        return SimpleNamespace(
            snapshot_path=qfq_snapshot,
            manifest={
                "dataset_id": dataset_id,
                "snapshot_id": "qfq-001",
                "price_basis": require_basis,
                "source": {"provider": "tencent"},
                "upstream_snapshots": [
                    {
                        "dataset_id": "etf_momentum_daily_raw",
                        "snapshot_id": "different-raw-snapshot",
                    }
                ],
            },
        )

    monkeypatch.setattr(module, "resolve_dataset", fake_resolve)

    with pytest.raises(module.DataContractError, match="paired raw snapshot"):
        module.resolve_canonical_etf_pair(tmp_path)


def test_select_price_source_keeps_raw_data_unverified_when_qfq_is_missing() -> None:
    module = load_module()
    raw = sample_prices()

    selected, quality = module.select_price_source(raw, None, minimum_coverage=0.98)

    assert quality["status"] == "unverified_data"
    assert quality["selected_source"] == "akshare.fund_etf_hist_sina"
    assert "independent_source_cross_check_not_done" in quality["evidence_gaps"]
    assert selected["SH513100"].equals(raw["SH513100"])


def test_build_strategy_report_contains_complete_sections_and_chart_series() -> None:
    module = load_module()
    data = sample_prices(320)
    _, quality = module.select_price_source(data, None, minimum_coverage=0.98)

    report = module.build_strategy_report(
        data,
        config=load_config(),
        data_quality=quality,
        generated_at="2026-07-25T21:00:00+08:00",
        data_manifest_hash="manifest-hash",
        git_commit="deadbeef",
        git_dirty=True,
    )

    assert report["status"] == "manual_review_required"
    assert set(report) >= {
        "summary",
        "strategy",
        "latest_signal",
        "performance",
        "stability",
        "risk",
        "chart_data",
        "data_quality",
        "reproducibility",
    }
    assert report["summary"]["data_status"] == "unverified_data"
    assert report["latest_signal"]["reliability"] == "unverified_data"
    assert report["latest_signal"]["signal"] in {"SZ159915", "SH513100", "CASH"}
    assert report["chart_data"]["wealth"]
    assert report["chart_data"]["drawdown"]
    assert report["chart_data"]["quarterly_returns"]
    assert report["chart_data"]["yearly_returns"]
    assert report["chart_data"]["monthly_allocations"]
    assert len(report["stability"]["reference_checks"]) == 7
    assert set(report["stability"]["quarter_consistency"]) >= {
        "strict_complete_positive_ratio",
        "all_available_positive_ratio",
        "earlier_comparison_positive_ratio",
    }


def test_verified_qfq_report_does_not_mark_raw_split_as_price_anomaly() -> None:
    module = load_module()
    data = sample_prices(320)
    _, quality = module.select_price_source(
        data,
        data,
        adjusted_source=module.TENCENT_ADJUSTED_SOURCE,
        adjusted_manifest={"corporate_actions": []},
    )
    quality["raw_price_discontinuities"] = {
        "passed": False,
        "events": [{"date": "2022-01-14"}],
    }

    report = module.build_strategy_report(
        data,
        config=load_config(),
        data_quality=quality,
        generated_at="2026-07-25T21:00:00+08:00",
        data_manifest_hash="manifest-hash",
        git_commit="deadbeef",
        git_dirty=True,
    )

    assert report["summary"]["data_status"] == "verified_qfq"
    assert report["chart_data"]["anomaly_dates"] == []


def test_drawdown_episodes_report_trough_and_recovery() -> None:
    module = load_module()
    curve = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="B"),
            "wealth": [1.0, 1.1, 0.8, 0.9, 1.1, 1.2],
        }
    )

    episodes = module.drawdown_episodes(curve)

    assert episodes[0]["start_date"] == "2025-01-02"
    assert episodes[0]["trough_date"] == "2025-01-03"
    assert episodes[0]["recovery_date"] == "2025-01-07"
    assert episodes[0]["max_drawdown"] == pytest.approx(0.8 / 1.1 - 1.0)


def test_holding_episode_summary_separates_cash_and_invested_holds() -> None:
    module = load_module()
    periods = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=6, freq="B"),
            "next_date": pd.date_range("2025-01-02", periods=6, freq="B"),
            "position": ["CASH", "SZ159915", "SZ159915", "CASH", "SH513100", "SH513100"],
            "net_return": [0.0, 0.02, -0.01, 0.0, 0.03, 0.01],
        }
    )

    summary = module.holding_episode_summary(periods)

    assert summary["episode_count"] == 4
    assert summary["invested_episode_count"] == 2
    assert summary["cash_episode_count"] == 2
    assert summary["invested_win_rate"] == pytest.approx(1.0)


def test_run_writes_canonical_latest_and_run_scoped_artifacts(tmp_path: Path) -> None:
    module = load_module()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for symbol, frame in sample_prices(180).items():
        frame.to_parquet(raw_dir / f"{symbol}.parquet", index=False)
    (raw_dir / "SOURCE_MANIFEST.json").write_text(
        '{"source":"akshare.fund_etf_hist_sina"}\n', encoding="utf-8"
    )
    reports_dir = tmp_path / "reports"

    run_dir = module.run(
        data_dir=raw_dir,
        adjusted_data_dir=tmp_path / "missing-qfq",
        report_dir=reports_dir,
        config_path=(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "etf_momentum_robustness.yaml"
        ),
        generated_at="2026-07-25T21:00:00+08:00",
    )

    assert (run_dir / "strategy_report.json").exists()
    assert (run_dir / "strategy_report.md").exists()
    assert (run_dir / "signals.parquet").exists()
    assert (reports_dir / "strategy_report_latest.json").exists()
    assert (reports_dir / "strategy_report_latest.md").exists()


@pytest.mark.parametrize(
    ("module_name", "filename", "entrypoint"),
    [
        ("backtest_etf_momentum_rotation", "backtest_etf_momentum_rotation.py", "run"),
        (
            "evaluate_etf_momentum_robustness",
            "evaluate_etf_momentum_robustness.py",
            "evaluate",
        ),
    ],
)
def test_legacy_report_entrypoints_delegate_to_canonical_generator(
    monkeypatch, tmp_path: Path, module_name: str, filename: str, entrypoint: str
) -> None:
    calls: list[dict[str, object]] = []
    expected = tmp_path / "canonical-run"

    def fake_run(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setitem(
        sys.modules,
        "generate_etf_momentum_strategy_report",
        SimpleNamespace(run=fake_run),
    )
    path = Path(__file__).resolve().parents[1] / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    legacy = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(legacy)

    kwargs = {"data_dir": tmp_path / "data", "report_dir": tmp_path / "reports"}
    if entrypoint == "evaluate":
        kwargs["config_path"] = tmp_path / "config.yaml"
    result = getattr(legacy, entrypoint)(**kwargs)

    assert result == expected
    assert len(calls) == 1
    assert calls[0]["data_dir"] == tmp_path / "data"
    assert calls[0]["report_dir"] == tmp_path / "reports"
