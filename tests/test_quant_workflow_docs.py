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
        "QUANT_WORKFLOW_PYTHON",
        "MLFLOW_URL",
    ):
        assert f"# {key}=" in env_example

    assert "reports/mlruns 的正式位置是仓内" in env_example
    assert "不再自动按 marker 兜底读取" in env_example
    assert "迁移时复制旧 reports/mlruns 到仓内" in env_example


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
        "china-astock 技术指标报告",
        "report_china_astock_indicators.py",
        "Qlib 预测摘要",
        "qlib_predictions_latest",
        "Qlib 研究增强",
        "report_qlib_research.py",
        "qlib_research_model_explainability_latest",
        "实验复现与产物",
        "pred.pkl",
        "runtime config",
        "quant_workflow 脚本覆盖盘点",
        "旧同级 reports/mlruns 不再自动读取或写入",
        "迁移旧历史应复制到仓内目录",
        "report_data_hygiene.py",
        "data_hygiene_inventory_latest",
        "data_hygiene_cleanup_plan_latest",
        "不包含真实删除动作",
        "report_workbench_capabilities.py",
        "workbench_capability_inventory_latest",
        "add_index_benchmark.py",
        "generate_csi300.py",
        "repair_qlib_provider.py",
        "缓存缺失时直接失败，不走 AkShare fallback",
        "supervise_baostock.py",
        "provider 维护脚本",
        "MLflow",
        "CHILD_ENV_EXACT_NAMES",
        "CHILD_ENV_PREFIXES",
        "不自动下单",
        "不读取账号密码",
        "pytest tests/test_quant_jobs_api.py",
        "npm run lint",
        "npm run build",
        "npm run test:smoke:fake-analysis",
        "npm run test:smoke:local",
        "apps/dsa-web/test-results/managed-smoke",
        "git status --short",
        "未跟踪正式文件",
        "鲁棒策略生命周期",
        "历史时点 CSI800",
        "blocked_data",
        "research_candidate_ready",
        "shadow_observation",
        "robust_strategy_qualified",
        "60 个交易日",
    ):
        assert phrase in manual


def test_verification_script_requires_robust_strategy_entrypoints() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )

    for filename in (
        "build_pit_csi800.py",
        "download_robust_reference.py",
        "preflight_robust_data.py",
        "prepare_robust_provider.py",
        "run_robust_development.py",
        "run_robust_strategy.py",
        "run_robust_confirmation.py",
        "run_robust_shadow_day.py",
        "robust_shadow_tracking.py",
    ):
        assert filename in script


def test_mlflow_workbench_docs_keep_user_and_backend_ports_distinct() -> None:
    install_doc = (ROOT_DIR / "docs" / "install-local-quant-workbench.md").read_text(
        encoding="utf-8"
    )
    manual = (ROOT_DIR / "docs" / "local-quant-workbench.md").read_text(
        encoding="utf-8"
    )

    assert "8010` 是 DSA 日常网页入口" in install_doc
    assert "6006` 是 MLflow 后台数据服务" in install_doc
    assert "普通实验和 Run 页面仍可能残留少量英文" in manual
    assert "upstreamCommit" in manual


def test_quant_workflow_runtime_data_doc_marks_embedded_runtime_canonical() -> None:
    runtime_doc = (ROOT_DIR / "quant_workflow" / "RUNTIME_DATA.md").read_text(
        encoding="utf-8"
    )

    assert "canonical local runtime home" in runtime_doc
    assert "reports/" in runtime_doc
    assert "mlruns/" in runtime_doc
    assert "../quant_workflow/reports/" in runtime_doc
    assert "rollback/deprecated copy" in runtime_doc
    assert "no longer auto-detects sibling history marker files" in runtime_doc
    assert "QUANT_WORKFLOW_REPORTS_ROOT" in runtime_doc
    assert "/Users/" not in runtime_doc
    assert "JQ_PASSWORD" not in runtime_doc


def test_verification_script_opencode_review_covers_untracked_files() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )

    assert "current working tree" in script
    assert "git diff HEAD" in script
    assert "git status --short" in script
    assert "untracked files" in script
    assert "newly added but untracked files" in script
    assert "HEAD~2..HEAD" in script


def test_verification_script_quick_covers_schedule_mode_regression() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )
    quick_body = script.split("run_quick() {", 1)[1].split("\n}\n", 1)[0]

    assert "tests/test_main_schedule_mode.py" in quick_body
    assert "tests/test_safe_urls.py" in quick_body
    assert "run_py_compile" in quick_body
    assert "check_quant_workflow_scripts" in quick_body
    assert "REQUIRED_QUANT_WORKFLOW_SCRIPTS" in script
    assert "report_china_astock_indicators.py" in script
    assert "report_workbench_capabilities.py" in script
    assert "report_qlib_research.py" in script
    assert "train_baseline_lgb.py" in script
    assert "run_candidate_pipeline.py" in script
    assert "Missing quant_workflow script(s)" in script
    assert "run_dsa_backtest is intentionally a DSA-native job" in script
    assert "Missing DSA backtest entrypoint" in script
    assert "python -m py_compile" not in script
    assert '"${PYTHON_BIN}" -m py_compile' in script
    assert "git diff --name-only HEAD -- '*.py'" in script
    assert "git ls-files --others --exclude-standard -- '*.py'" in script
    assert "No changed Python files to compile." in script
    assert "run_quick_browser_hygiene" in quick_body
    assert "verify-data-hygiene.spec.ts" in script
    assert "[SKIPPED] Local DSA service is not reachable" in script
    assert "./scripts/ci_gate.sh all" in script
    assert 'PATH="$(dirname "${PYTHON_BIN}"):${PATH}"' in script


def test_verification_script_opencode_review_fails_without_bridge() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )

    assert "opencode-review cannot run" in script
    assert "Install cli-agent-interop or set AGENT_BRIDGE" in script
    assert "before claiming complete verification" in script
    assert "OPENCODE_REVIEW_TIMEOUT_SECONDS" in script
    assert "opencode-review timed out" in script
    assert "SystemExit(124)" in script
    opencode_body = script.split("run_opencode_review() {", 1)[1].split("\n}\n", 1)[0]
    assert "return 2" in opencode_body
    assert "return 0" not in opencode_body


def test_managed_browser_smoke_cannot_reuse_the_live_dsa_service() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )
    package_json = (ROOT_DIR / "apps" / "dsa-web" / "package.json").read_text(
        encoding="utf-8"
    )
    playwright_config = (
        ROOT_DIR / "apps" / "dsa-web" / "playwright.config.ts"
    ).read_text(encoding="utf-8")
    fake_body = script.split("run_browser_fake() {", 1)[1].split("\n}\n", 1)[0]

    assert 'DSA_WEB_E2E_BASE_URL="http://127.0.0.1:4173"' in fake_body
    assert 'DSA_WEB_SMOKE_BACKEND_PORT="${managed_backend_port}"' in fake_body
    assert (
        'DSA_WEB_DEV_API_TARGET="http://127.0.0.1:${managed_backend_port}"' in fake_body
    )
    assert (
        "DSA_WEB_SMOKE_BACKEND_PORT=${DSA_WEB_SMOKE_BACKEND_PORT:-18011}"
        in package_json
    )
    assert "reuseExistingServer: false" in playwright_config


def test_web_package_exposes_local_workbench_smoke_entry() -> None:
    package_json = (ROOT_DIR / "apps" / "dsa-web" / "package.json").read_text(
        encoding="utf-8"
    )
    smoke_spec = ROOT_DIR / "apps" / "dsa-web" / "e2e" / "local-workbench.spec.ts"
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert '"test:smoke:local"' in package_json
    assert "DSA_WEB_E2E_BASE_URL" in package_json
    assert smoke_spec.exists()
    assert "dsa_e2e_" in smoke_spec.read_text(encoding="utf-8")
    assert "本地工作台浏览器 smoke" in unreleased


def test_web_package_exposes_fake_analysis_smoke_entry() -> None:
    package_json = (ROOT_DIR / "apps" / "dsa-web" / "package.json").read_text(
        encoding="utf-8"
    )
    playwright_config = (
        ROOT_DIR / "apps" / "dsa-web" / "playwright.config.ts"
    ).read_text(encoding="utf-8")
    vite_config = (ROOT_DIR / "apps" / "dsa-web" / "vite.config.ts").read_text(
        encoding="utf-8"
    )
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
    assert (
        "managedWebServerEnv.QUANT_WORKFLOW_REPORTS_ROOT = managedReportsRoot"
        in playwright_config
    )
    assert "env: managedWebServerEnv" in playwright_config
    assert "managed-smoke" in playwright_config
    assert "DSA_WEB_DEV_API_TARGET" in vite_config
    assert "test:smoke:fake-analysis" in unreleased
    assert "嵌套按钮" in unreleased


def test_mimo_llm_checklist_does_not_embed_real_token_shape() -> None:
    checklist = (ROOT_DIR / "docs" / "mimo-llm-checklist.md").read_text(
        encoding="utf-8"
    )

    assert "https://token-plan-cn.xiaomimimo.com/v1" in checklist
    assert not re.search(r"tp-[A-Za-z0-9]{8,}", checklist)
