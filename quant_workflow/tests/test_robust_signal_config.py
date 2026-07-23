from pathlib import Path

import yaml


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "robust_signal_research.yaml"


def test_signal_research_preregisters_exactly_eight_unique_schemes() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    signals = payload["signals"]

    assert len(signals) == 8
    assert len({item["id"] for item in signals}) == 8
    assert {item["horizon_days"] for item in signals} == {5, 10}
    assert payload["evaluation"]["market"] == "csi800"
    assert payload["evaluation"]["primary_topk"] == 50


def test_signal_research_locks_2026_out_of_screening_and_validation() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    quarters = payload["periods"]["screening_quarters"] + payload["periods"][
        "validation_quarters"
    ]

    assert not any(str(quarter).startswith("2026") for quarter in quarters)
    assert payload["safety"]["locked_confirmation_start"] == "2026-01-01"
