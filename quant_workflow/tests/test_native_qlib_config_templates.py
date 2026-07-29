from pathlib import Path

import yaml


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES = (
    "alpha158_lgb_baseline.yaml",
    "baseline_lgb_alpha158.yaml",
    "allA_lgb_alpha158.yaml",
    "bear_market_test.yaml",
    "pca_alpha158.yaml",
)


def test_qlib_templates_require_runtime_native_snapshot_resolution():
    for name in TEMPLATES:
        payload = yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))
        assert payload["qlib_init"]["provider_uri"] == (
            "RESOLVE_QLIB_ASHARE_V2_AT_RUNTIME"
        )


def _production_sources() -> list[Path]:
    return [
        *REPO_ROOT.joinpath("quant_workflow", "scripts").glob("*.py"),
        *REPO_ROOT.joinpath("src").rglob("*.py"),
        *REPO_ROOT.joinpath("api").rglob("*.py"),
        *REPO_ROOT.joinpath("apps", "dsa-web", "src").rglob("*.tsx"),
    ]


def _unexpected_token_users(token: str, allowed: set[str]) -> list[str]:
    offenders: list[str] = []
    for path in _production_sources():
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "__tests__" in path.parts or relative in allowed:
            continue
        if token in path.read_text(encoding="utf-8"):
            offenders.append(relative)
    return offenders


def test_legacy_qlib_paths_are_confined_to_explicit_compatibility_roles():
    allowed = {
        "quant_workflow/scripts/_paths.py",
        "quant_workflow/scripts/publish_v2_compatibility.py",
        "quant_workflow/scripts/report_data_hygiene.py",
        "quant_workflow/scripts/sync_data_manifest.py",
    }

    assert _unexpected_token_users("cn_data_baostock_jqdata", allowed) == []


def test_legacy_merged_parquet_paths_are_confined_to_legacy_or_frozen_roles():
    allowed = {
        "quant_workflow/scripts/apply_data_cleaning.py",
        "quant_workflow/scripts/merge_to_qlib.py",
        "quant_workflow/scripts/preflight_robust_data.py",
        "quant_workflow/scripts/publish_v2_compatibility.py",
        "quant_workflow/scripts/report_china_astock_indicators.py",
        "quant_workflow/scripts/report_data_hygiene.py",
        "quant_workflow/scripts/sync_data_manifest.py",
    }

    assert _unexpected_token_users("merged_baostock_jqdata", allowed) == []
