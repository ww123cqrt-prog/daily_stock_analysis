from __future__ import annotations

import importlib.util
import json
import threading
from pathlib import Path

import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "robust_shadow_tracking.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("robust_shadow_tracking", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def observation(
    day: str,
    portfolio_return: float = 0.001,
    benchmark_return: float = 0.0002,
    success: bool = True,
):
    return {
        "date": day,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "task_success": success,
    }


def test_shadow_observations_are_unique_chronological_and_forward_only() -> None:
    module = load_module()
    state = module.new_shadow_state(
        confirmation_run_id="confirmation-one",
        frozen_config_hash="a" * 64,
        start_after="2026-07-15",
    )

    state = module.append_observation(state, observation("2026-07-16"))
    same = module.append_observation(state, observation("2026-07-16"))

    assert len(same["observations"]) == 1
    assert same["status"] == "shadow_observation"
    try:
        module.append_observation(state, observation("2026-07-14"))
    except ValueError as exc:
        assert "after" in str(exc)
    else:
        raise AssertionError("pre-confirmation observation was accepted")


def test_60_successful_days_with_positive_excess_qualify() -> None:
    module = load_module()
    state = module.new_shadow_state(
        confirmation_run_id="confirmation-one",
        frozen_config_hash="a" * 64,
        start_after="2026-07-15",
    )
    for day in pd.bdate_range("2026-07-16", periods=60):
        state = module.append_observation(state, observation(str(day.date())))

    assert state["status"] == "robust_strategy_qualified"
    assert state["verdict"]["passed"] is True
    assert state["metrics"]["trading_days"] == 60
    assert state["metrics"]["net_excess_return"] > 0
    assert state["metrics"]["task_success_ratio"] == 1.0


def test_60_days_fail_when_task_success_ratio_is_below_95_percent() -> None:
    module = load_module()
    state = module.new_shadow_state(
        confirmation_run_id="confirmation-one",
        frozen_config_hash="a" * 64,
        start_after="2026-07-15",
    )
    for index, day in enumerate(pd.bdate_range("2026-07-16", periods=60)):
        state = module.append_observation(
            state, observation(str(day.date()), success=index >= 4)
        )

    assert state["status"] == "development_failed"
    assert state["verdict"]["passed"] is False
    assert "task_success_ratio" in state["verdict"]["failed_gates"]


def test_drawdown_gate_uses_compounded_paper_portfolio() -> None:
    module = load_module()
    state = module.new_shadow_state(
        confirmation_run_id="confirmation-one",
        frozen_config_hash="a" * 64,
        start_after="2026-07-15",
    )
    days = pd.bdate_range("2026-07-16", periods=60)
    for index, day in enumerate(days):
        value = -0.15 if index == 30 else 0.004
        state = module.append_observation(
            state,
            observation(str(day.date()), portfolio_return=value, benchmark_return=0.0),
        )

    assert state["metrics"]["max_drawdown"] < -0.12
    assert state["status"] == "development_failed"
    assert "maximum_drawdown" in state["verdict"]["failed_gates"]


def test_shadow_signals_are_written_atomically_and_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    module = load_module()
    path = tmp_path / "shadow_signal_predictions.parquet"
    prediction = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2026-07-16", "2026-07-16"]),
            "instrument": ["SH600000", "SZ000001"],
            "score": [0.75, 0.25],
        }
    )

    first = module.append_signal_predictions(path, prediction, as_of_date="2026-07-16")
    second = module.append_signal_predictions(path, prediction, as_of_date="2026-07-16")

    assert first["appended_rows"] == 2
    assert second["appended_rows"] == 0
    saved = (
        pd.read_parquet(path)
        .sort_values(["datetime", "instrument"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        saved, prediction.sort_values(["datetime", "instrument"]).reset_index(drop=True)
    )

    conflicting = prediction.copy()
    conflicting.loc[0, "score"] = 0.8
    with pytest.raises(ValueError, match="different score"):
        module.append_signal_predictions(path, conflicting, as_of_date="2026-07-16")


def test_shadow_signals_reject_backfill_and_non_finite_scores(tmp_path: Path) -> None:
    module = load_module()
    path = tmp_path / "shadow_signal_predictions.parquet"

    with pytest.raises(ValueError, match="as_of_date"):
        module.append_signal_predictions(
            path,
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-07-15"]),
                    "instrument": ["SH600000"],
                    "score": [0.5],
                }
            ),
            as_of_date="2026-07-16",
        )

    with pytest.raises(ValueError, match="finite"):
        module.append_signal_predictions(
            path,
            pd.DataFrame(
                {
                    "datetime": pd.to_datetime(["2026-07-16"]),
                    "instrument": ["SH600000"],
                    "score": [float("nan")],
                }
            ),
            as_of_date="2026-07-16",
        )


def test_concurrent_signal_appends_do_not_lose_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    path = tmp_path / "shadow_signal_predictions.parquet"
    first_at_write = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []
    original_write = module._atomic_write_parquet

    def delayed_write(frame: pd.DataFrame, target: Path) -> None:
        dates = set(pd.to_datetime(frame["datetime"]).dt.strftime("%Y-%m-%d"))
        if dates == {"2026-07-16"}:
            first_at_write.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first writer")
        original_write(frame, target)

    monkeypatch.setattr(module, "_atomic_write_parquet", delayed_write)

    def append(day: str, done: threading.Event | None = None) -> None:
        try:
            module.append_signal_predictions(
                path,
                pd.DataFrame(
                    {
                        "datetime": pd.to_datetime([day]),
                        "instrument": ["SH600000"],
                        "score": [0.5],
                    }
                ),
                as_of_date=day,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=append, args=("2026-07-16",))
    second = threading.Thread(target=append, args=("2026-07-17", second_done))
    first.start()
    assert first_at_write.wait(timeout=5)
    second.start()
    second_finished_while_first_was_writing = second_done.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_finished_while_first_was_writing is False
    assert errors == []
    saved = pd.read_parquet(path)
    assert set(saved["datetime"].dt.strftime("%Y-%m-%d")) == {
        "2026-07-16",
        "2026-07-17",
    }


def test_concurrent_state_updates_do_not_lose_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_module()
    state_path = tmp_path / "shadow_state_latest.json"
    first_at_write = threading.Event()
    release_first = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []
    original_write = module._atomic_write

    def delayed_write(path: Path, content: str) -> None:
        payload = json.loads(content)
        dates = [item["date"] for item in payload.get("observations", [])]
        if dates == ["2026-07-16"]:
            first_at_write.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first state writer")
        original_write(path, content)

    monkeypatch.setattr(module, "_atomic_write", delayed_write)

    def update(day: str, done: threading.Event | None = None) -> None:
        try:
            module.update_shadow_state_file(
                state_path,
                confirmation_run_id="confirmation-one",
                frozen_config_hash="a" * 64,
                start_after="2026-07-15",
                observation=observation(day),
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=update, args=("2026-07-16",))
    second = threading.Thread(target=update, args=("2026-07-17", second_done))
    first.start()
    assert first_at_write.wait(timeout=5)
    second.start()
    second_finished_while_first_was_writing = second_done.wait(timeout=0.2)
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_finished_while_first_was_writing is False
    assert errors == []
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert [item["date"] for item in saved["observations"]] == [
        "2026-07-16",
        "2026-07-17",
    ]
