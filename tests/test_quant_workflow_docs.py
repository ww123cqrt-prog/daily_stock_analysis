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
        "QUANT_DATA_ROOT",
        "QLIB_PYTHON",
        "MLFLOW_URL",
    ):
        assert f"# {key}=" in env_example


def test_changelog_mentions_web_quant_workflow_entry() -> None:
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert "/workflow" in unreleased
    assert "量化任务中心" in unreleased
    assert "Qlib" in unreleased


def test_local_quant_workbench_manual_covers_self_service_checklist() -> None:
    manual_path = ROOT_DIR / "docs" / "local-quant-workbench.md"

    manual = manual_path.read_text(encoding="utf-8")

    for phrase in (
        "本地统一量化研究与回测工作台",
        "http://127.0.0.1:8010/",
        "/workflow",
        "/research",
        "/backtest",
        "quant_workflow",
        "Qlib",
        "china-astock-quant",
        "唯一 Python 虚拟环境",
        "QUANT_DATA_ROOT",
        "MLFLOW_URL",
        "策略优化预演",
        "Qlib 预测摘要",
        "qlib_predictions_latest",
        "实验复现与产物",
        "pred.pkl",
        "runtime config",
        "quant_workflow 脚本覆盖盘点",
        "add_index_benchmark.py",
        "generate_csi300.py",
        "supervise_baostock.py",
        "provider 维护脚本",
        "MLflow",
        "不自动下单",
        "不读取账号密码",
        "pytest tests/test_quant_jobs_api.py",
        "npm run lint",
        "npm run build",
        "npm run test:smoke:fake-analysis",
        "npm run test:smoke:local",
        "apps/dsa-web/test-results/managed-smoke",
    ):
        assert phrase in manual


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
    assert "DSA_WEB_SMOKE_REPORTS_ROOT" in playwright_config
    assert "QUANT_WORKFLOW_REPORTS_ROOT" in playwright_config
    assert "managedWebServerEnv.QUANT_WORKFLOW_REPORTS_ROOT = managedReportsRoot" in playwright_config
    assert "env: managedWebServerEnv" in playwright_config
    assert "managed-smoke" in playwright_config
    assert "DSA_WEB_DEV_API_TARGET" in vite_config
    assert "test:smoke:fake-analysis" in unreleased
    assert "嵌套按钮" in unreleased


def test_mimo_llm_checklist_does_not_embed_real_token_shape() -> None:
    checklist = (ROOT_DIR / "docs" / "mimo-llm-checklist.md").read_text(encoding="utf-8")

    assert "https://token-plan-cn.xiaomimimo.com/v1" in checklist
    assert not re.search(r"tp-[A-Za-z0-9]{8,}", checklist)
