# -*- coding: utf-8 -*-
"""Documentation coverage for the local quant workflow UI."""

from pathlib import Path
import re


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_env_example_documents_quant_workflow_path_overrides() -> None:
    env_example = (ROOT_DIR / ".env.example").read_text(encoding="utf-8")

    assert "本地量化工作流" in env_example
    for key in (
        "DSA_ROOT",
        "WORKING_PROJECTS_ROOT",
        "QUANT_WORKFLOW_ROOT",
        "QUANT_WORKFLOW_REPORTS_ROOT",
        "QLIB_PYTHON",
    ):
        assert f"# {key}=" in env_example


def test_changelog_mentions_web_quant_workflow_entry() -> None:
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert "/workflow" in unreleased
    assert "量化任务中心" in unreleased
    assert "Qlib" in unreleased


def test_web_package_exposes_local_workbench_smoke_entry() -> None:
    package_json = (ROOT_DIR / "apps" / "dsa-web" / "package.json").read_text(encoding="utf-8")
    smoke_spec = ROOT_DIR / "apps" / "dsa-web" / "e2e" / "local-workbench.spec.ts"
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert '"test:smoke:local"' in package_json
    assert "DSA_WEB_E2E_BASE_URL" in package_json
    assert smoke_spec.exists()
    assert "dsa_e2e_" in smoke_spec.read_text(encoding="utf-8")
    assert "本地工作台浏览器 smoke" in unreleased


def test_web_package_exposes_fake_analysis_smoke_entry() -> None:
    package_json = (ROOT_DIR / "apps" / "dsa-web" / "package.json").read_text(encoding="utf-8")
    playwright_config = (ROOT_DIR / "apps" / "dsa-web" / "playwright.config.ts").read_text(encoding="utf-8")
    vite_config = (ROOT_DIR / "apps" / "dsa-web" / "vite.config.ts").read_text(encoding="utf-8")
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert '"test:smoke:fake-analysis"' in package_json
    assert "DSA_WEB_SMOKE_MANAGED=1" in package_json
    assert "DSA_WEB_E2E_FAKE_ANALYSIS=1" in package_json
    assert "DSA_E2E_FAKE_ANALYSIS=1" in package_json
    assert "DSA_WEB_SMOKE_BACKEND_PORT" in package_json
    assert "DSA_WEB_SMOKE_MANAGED" in playwright_config
    assert "DSA_WEB_DEV_API_TARGET" in vite_config
    assert "test:smoke:fake-analysis" in unreleased
    assert "嵌套按钮" in unreleased


def test_mimo_llm_checklist_does_not_embed_real_token_shape() -> None:
    checklist = (ROOT_DIR / "docs" / "mimo-llm-checklist.md").read_text(encoding="utf-8")

    assert "https://token-plan-cn.xiaomimimo.com/v1" in checklist
    assert not re.search(r"tp-[A-Za-z0-9]{8,}", checklist)
