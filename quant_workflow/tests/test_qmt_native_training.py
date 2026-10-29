from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
import yaml


@pytest.fixture
def frozen_input(tmp_path):
    from quant_qmt_data.dataset import save_dataset

    calendar = pd.bdate_range("2024-01-02", periods=130)
    symbols = ["SH600519", "SZ000001", "SH510300", "BJ920149"]
    rows = []
    for offset, symbol in enumerate(symbols):
        for position, day in enumerate(calendar):
            if symbol == "SH600519" and position == 30:
                continue
            close = 20 + offset * 5 + position * (0.02 + offset * 0.005) + np.sin(position / (3 + offset))
            volume = 10000.0 + position * 10 + offset + 200 * np.sin(position / (2 + offset))
            rows.append(dict(
                date=day, symbol=symbol, open=close * 0.995, high=close * 1.01,
                low=close * 0.99, close=close, volume=volume,
                amount=close * volume,
                preclose=close * 0.998, suspend_flag=0,
            ))
    frame = pd.DataFrame(rows)
    frame.attrs = dict(
        provider="qmt", request=dict(symbols=symbols, start=str(calendar[0].date()),
            end=str(calendar[-1].date()), adjustment="front", volume_unit="base"),
        price_basis="front", volume_units={s: "units" if s == "SH510300" else "shares" for s in symbols},
        missing_symbols=[], read_at="2026-10-28T22:00:00+08:00",
        completed_through=str(calendar[-1].date()), contains_partial=False,
        qmt_data_dir=str(tmp_path / "qmt-datadir"), sdk_version="fixture",
        actual_start=str(calendar[0].date()), actual_end=str(calendar[-1].date()),
    )
    directory = save_dataset(frame, tmp_path / "frozen", calendar=calendar, universe=symbols)
    return directory, frame, calendar, symbols


@pytest.fixture
def source_config(tmp_path, frozen_input):
    _, _, calendar, _ = frozen_input
    template = Path(__file__).resolve().parents[1] / "configs" / "baseline_qmt_native.yaml"
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    kwargs = payload["task"]["dataset"]["kwargs"]
    kwargs["segments"] = {
        "train": [str(calendar[20].date()), str(calendar[79].date())],
        "valid": [str(calendar[80].date()), str(calendar[104].date())],
        "test": [str(calendar[105].date()), str(calendar[129].date())],
    }
    kwargs["handler"]["kwargs"].update(
        start_time=str(calendar[20].date()), end_time=str(calendar[-1].date()),
        fit_start_time=str(calendar[20].date()), fit_end_time=str(calendar[79].date()),
    )
    payload["task"]["model"]["kwargs"].update(num_boost_round=8, early_stopping_rounds=3)
    source = tmp_path / "smoke.yaml"
    source.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return source


def test_frozen_provider_preserves_calendar_selection_and_gaps(frozen_input):
    from qlib.data import D
    from quant_workflow.qmt_native_provider import init_qlib_from_quant_data

    directory, frame, calendar, symbols = frozen_input
    init_qlib_from_quant_data(input_dir=directory, kernels=1)
    assert list(pd.to_datetime(D.calendar())) == list(calendar)
    assert set(D.list_instruments(D.instruments("all"), as_list=True)) == set(symbols)
    features = D.features(["SH600519"], ["$close", "$volume", "$factor"],
        start_time=calendar[29], end_time=calendar[31], disk_cache=0)
    stock = features.xs("SH600519", level="instrument")
    assert stock.loc[calendar[30]].isna().all()
    original = frame.loc[(frame.symbol == "SH600519") & (frame.date == calendar[31])].iloc[0]
    assert stock.loc[calendar[31], "$close"] == pytest.approx(original.close)
    assert stock.loc[calendar[31], "$volume"] == pytest.approx(original.volume)
    assert stock.loc[calendar[31], "$factor"] == 1.0
    assert {p.name for p in directory.iterdir()} == {"bars.parquet", "dataset.json"}


def test_frozen_membership_intervals_are_used_without_sdk(tmp_path, frozen_input):
    from quant_qmt_data.dataset import save_dataset
    from quant_workflow.qmt_native_provider import init_qlib_from_quant_data
    from qlib.data import D

    _, frame, calendar, symbols = frozen_input
    intervals = [{"symbol": s, "start_date": str(calendar[10].date()), "end_date": str(calendar[60].date())}
                 for s in symbols]
    directory = save_dataset(frame, tmp_path / "historical", calendar=calendar,
        universe=symbols, universe_kind="historical", membership_intervals=intervals)
    init_qlib_from_quant_data(input_dir=directory, kernels=1)
    instruments = D.list_instruments(D.instruments("all"))
    assert instruments["SH600519"] == [(calendar[10], calendar[60])]
    with pytest.raises(ValueError, match="CSI800|csi800|selected|historical"):
        D.list_instruments(D.instruments("csi800"))


def test_provider_identity_is_bound_to_input_and_public_lineage_hides_paths(
    monkeypatch, frozen_input,
):
    from quant_workflow.scripts import _mlflow_run_provenance as provenance
    from quant_workflow.scripts import _paths

    directory, frame, _, _ = frozen_input
    monkeypatch.setenv("QMT_DATASET_DIR", str(directory))
    identity = provenance.provider_identity(directory)
    metadata = json.loads((directory / "dataset.json").read_text(encoding="utf-8"))
    assert identity == dict(backend="qmt_native", experiment_id=metadata["experiment_id"],
        bars_sha256=metadata["bars_sha256"], inputs_sha256=metadata["inputs_sha256"],
        price_basis=frame.attrs["price_basis"],
        input_dir=str(directory))
    public = provenance.normalize_provider_lineage(identity)
    assert public == {k: v for k, v in identity.items() if k != "input_dir"}
    assert provenance.provider_lineages_match(public, identity)
    assert _paths.qlib_workflow_provider_uri("qmt_native", expected_provider_identity=public) == directory
    with (directory / "bars.parquet").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises((ValueError, provenance.RunProvenanceError), match="SHA256|hash|integrity"):
        provenance.revalidate_provider_identity(directory, identity)


@pytest.mark.parametrize("future_sessions", [2, 3])
def test_runtime_config_purges_actual_label_window_and_fits_effective_train(
    tmp_path, frozen_input, source_config, future_sessions,
):
    from quant_workflow.scripts import train_baseline_lgb as training

    directory, _, calendar, _ = frozen_input
    payload = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    payload["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = [
        [f"Ref($close, -{future_sessions}) / Ref($close, -1) - 1"], ["LABEL0"]
    ]
    source_config.write_text(yaml.safe_dump(payload), encoding="utf-8")
    source_bytes = source_config.read_bytes()
    runtime = training.write_runtime_config_with_provider(source_config, directory,
        "qmt_native", out_dir=tmp_path / "runtime", extend_dates_to_provider_latest=False)
    configured = yaml.safe_load(runtime.read_text(encoding="utf-8"))
    assert source_config.read_bytes() == source_bytes
    for component in ("calendar_provider", "instrument_provider", "feature_provider"):
        assert configured["qlib_init"][component]["kwargs"]["input_dir"] == str(directory)
    kwargs = configured["task"]["dataset"]["kwargs"]
    for name, end_index in (("train", 79), ("valid", 104), ("test", 129)):
        assert kwargs["segments"][name][1] == str(calendar[end_index - future_sessions].date())
    assert kwargs["handler"]["kwargs"]["fit_end_time"] == kwargs["segments"]["train"][1]
    semantics = configured["task"]["time_semantics"]
    assert semantics["label_future_sessions"] == future_sessions
    assert semantics["label_last_observation"]["train"] == str(calendar[79].date())
    assert semantics["validation_used_for_model_selection"] is True


def test_default_config_keeps_real_holdout_out_of_training():
    template = Path(__file__).resolve().parents[1] / "configs" / "baseline_qmt_native.yaml"
    task = yaml.safe_load(template.read_text(encoding="utf-8"))["task"]
    segments = task["dataset"]["kwargs"]["segments"]
    assert segments == {
        "train": ["2021-01-01", "2022-12-31"],
        "valid": ["2023-01-01", "2023-12-31"],
        "test": ["2024-01-01", "2024-12-31"],
    }
    assert [record["class"] for record in task["record"]] == ["SignalRecord"]


def test_independent_research_modules_import_without_legacy_provider():
    import importlib
    modules = (
        "generate_csi300", "merge_to_qlib", "apply_data_cleaning",
        "compare_native_qlib_benchmarks", "add_index_benchmark", "repair_qlib_provider",
        "report_market_regime", "report_cash_control_backtest",
        "report_catdesk_signal_validation", "report_china_astock_indicators",
        "report_data_anomalies", "validate_latest_snapshot", "run_robust_shadow_day",
        "prepare_robust_provider", "report_data_quality", "run_robust_strategy",
        "run_robust_development",
    )
    failures = []
    for name in modules:
        try:
            importlib.import_module(f"quant_workflow.scripts.{name}")
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")
    assert not failures, "\n".join(failures)




def test_training_features_labels_and_preprocessor_fit_ignore_future_values(
    tmp_path, frozen_input, source_config,
):
    import qlib
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.utils import init_instance_by_config
    from quant_qmt_data.dataset import save_dataset
    from quant_workflow.scripts import train_baseline_lgb as training

    directory, frame, calendar, symbols = frozen_input
    prepared = []
    statistics = []
    for offset in (0, 1):
        if offset:
            modified = frame.copy()
            future = modified.date >= calendar[80]
            modified.loc[future, ["open", "high", "low", "close", "preclose"]] *= 7
            modified.loc[future, "volume"] *= 5
            directory = save_dataset(modified, tmp_path / "changed-future",
                calendar=calendar, universe=symbols)
        runtime = training.write_runtime_config_with_provider(source_config, directory,
            "qmt_native", out_dir=tmp_path / f"runtime-{offset}", extend_dates_to_provider_latest=False)
        config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        qlib.init(**config["qlib_init"])
        dataset = init_instance_by_config(config["task"]["dataset"])
        prepared.append(dataset.prepare("train", data_key=DataHandlerLP.DK_L))
        normalizer = dataset.handler.infer_processors[0]
        statistics.append((normalizer.mean_train.copy(), normalizer.std_train.copy()))
        if not offset:
            raw = dataset.handler.fetch(selector=slice(calendar[80], calendar[80]),
                col_set="feature", data_key=DataHandlerLP.DK_R)
            history = frame.loc[(frame.symbol == "SH600519") & frame.date.between(calendar[76], calendar[80]), "close"]
            assert raw.loc[(calendar[80], "SH600519"), "MA5"] == pytest.approx(history.mean() / history.iloc[-1])
    assert_frame_equal(prepared[0], prepared[1])
    np.testing.assert_allclose(statistics[0][0], statistics[1][0], equal_nan=True)
    np.testing.assert_allclose(statistics[0][1], statistics[1][1], equal_nan=True)


def test_ohlcv_configuration_excludes_vwap(frozen_input, source_config):
    from quant_workflow.qmt_native_provider import QMTNativeOHLCV

    expressions, _ = QMTNativeOHLCV.get_feature_config()
    assert expressions
    assert not any("vwap" in expression.lower() for expression in expressions)
    assert "$amount" not in " ".join(expressions)


def test_real_fit_predict_records_mlflow_and_uses_isolated_output(
    tmp_path, monkeypatch, frozen_input, source_config,
):
    import mlflow
    from quant_workflow.scripts import train_baseline_lgb as training
    from quant_workflow.scripts._mlflow_run_provenance import provider_identity

    directory, _, calendar, symbols = frozen_input
    store = tmp_path / "mlruns"
    monkeypatch.setattr(training, "MLRUNS_DIR", store)
    monkeypatch.setattr(training, "MLFLOW_TRACKING_URI", store.resolve().as_uri())
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.resolve().as_uri())
    runtime = training.write_runtime_config_with_provider(source_config, directory,
        "qmt_native", out_dir=tmp_path / "runtime", extend_dates_to_provider_latest=False)
    run_id = training.run_qrun(str(runtime), experiment_name="qmt_native_smoke")
    info = training.load_run_metrics(run_id, "qmt_native_smoke")
    pred = pd.read_pickle(store / info["experiment_id"] / run_id / "artifacts" / "pred.pkl")
    assert not pred.empty and np.isfinite(pred.to_numpy()).all()
    assert set(pred.index.names) == {"datetime", "instrument"}
    assert set(pred.index.get_level_values("instrument")) == set(symbols)
    assert pred.index.get_level_values("datetime").max() == calendar[127]
    training.publish_training_result(info, provider="qmt_native", provider_uri=directory,
        expected_provider_identity=provider_identity(directory), source_config=source_config,
        runtime_config=runtime, publish_latest=False, report_dir=tmp_path / "reports",
        result_output=tmp_path / "smoke-result.json")
    assert not (tmp_path / "reports" / "latest_training_result.json").exists()
    result = json.loads((tmp_path / "smoke-result.json").read_text(encoding="utf-8"))
    assert result["time_semantics"]["label_future_sessions"] == 2
    tags = mlflow.tracking.MlflowClient(tracking_uri=store.resolve().as_uri()).get_run(run_id).data.tags
    assert tags["dsa.quant_data.backend"] == "qmt_native"
