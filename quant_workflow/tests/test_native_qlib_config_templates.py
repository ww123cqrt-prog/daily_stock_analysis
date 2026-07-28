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


def test_legacy_provider_paths_are_confined_to_explicit_compatibility_roles():
    allowed = {
        "apps/dsa-web/src/pages/ResearchPage.tsx",
        "quant_workflow/scripts/_paths.py",
        "quant_workflow/scripts/apply_data_cleaning.py",
        "quant_workflow/scripts/build_qlib_v2.py",
        "quant_workflow/scripts/merge_to_qlib.py",
        "quant_workflow/scripts/preflight_robust_data.py",
        "quant_workflow/scripts/publish_v2_compatibility.py",
        "quant_workflow/scripts/report_china_astock_indicators.py",
        "quant_workflow/scripts/report_data_hygiene.py",
        "quant_workflow/scripts/sync_data_manifest.py",
    }
    tokens = ("cn_data_baostock_jqdata", "merged_baostock_jqdata")
    offenders: list[str] = []
    candidates = [
        *REPO_ROOT.joinpath("quant_workflow", "scripts").glob("*.py"),
        *REPO_ROOT.joinpath("src").rglob("*.py"),
        *REPO_ROOT.joinpath("apps", "dsa-web", "src").rglob("*.tsx"),
    ]
    for path in candidates:
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "__tests__" in path.parts:
            continue
        if relative in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in tokens):
            offenders.append(relative)

    assert offenders == []
