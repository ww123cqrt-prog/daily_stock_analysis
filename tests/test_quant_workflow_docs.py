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


def test_changelog_mentions_consolidated_quant_research_entry() -> None:
    changelog = (ROOT_DIR / "docs" / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = changelog.split("## [3.21.0]", 1)[0]

    assert "首页、数据、量化研究、个股分析、持仓和设置" in unreleased
    assert "/research" in unreleased
    assert "/api/v1/data/status" in unreleased
    assert "quant_data" in unreleased


def test_quant_data_docs_mark_ashare_v2_datasets_as_canonical_ready() -> None:
    contract = (
        ROOT_DIR / "quant_workflow" / "docs" / "quant-data-contract.md"
    ).read_text(encoding="utf-8")
    readme = (ROOT_DIR / "quant_workflow" / "README.md").read_text(
        encoding="utf-8"
    )

    for dataset_id in (
        "ashare_daily_raw",
        "ashare_daily_qfq",
        "qlib_ashare_v2",
    ):
        assert dataset_id in contract
        assert dataset_id in readme
    assert "A 股全市场因子快照和新 Qlib provider 尚未完成迁移" not in contract
    assert "不等于 A 股 v2 迁移完成" not in contract
    assert "until full raw/factor evidence is published" not in readme


def test_local_quant_workbench_manual_covers_consolidated_entry_contract() -> None:
    manual_path = ROOT_DIR / "docs" / "local-quant-workbench.md"

    manual = manual_path.read_text(encoding="utf-8")

    for phrase in (
        "本地统一量化研究与回测工作台",
        "| 首页 | `/` |",
        "| 数据 | `/data` |",
        "| 量化研究 | `/research` |",
        "| 个股分析 | `/stock-analysis` |",
        "| 持仓 | `/portfolio` |",
        "| 设置 | `/settings` |",
        "九点猫研",
        "MLflow",
        "`quant_data` 数据服务",
        "不下载、不清洗、不发布数据",
        "不提供下载、构建或发布按钮",
        "选股研究",
        "交易策略研究",
        "CSI800 选股信号稳定性检验",
        "/workflow",
        "/backtest",
        "/other",
        "旧地址继续兼容",
        "不自动下单",
        "不读取账号密码",
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
        "report_robust_strategy_closure.py",
        "robust_signal_dataset.py",
        "robust_signal_evaluation.py",
        "robust_signal_models.py",
        "run_robust_signal_research.py",
        "launch_robust_signal_research.py",
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
    assert "保留 MLflow 3.14.0 官方首页" in manual
    assert "部分页面仍可能残留少量英文" in manual
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


def test_verification_script_sets_quant_workflow_script_import_path() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )
    quant_body = script.split("run_quant_workflow() {", 1)[1].split("\n}\n", 1)[0]

    assert 'PYTHONPATH="${QUANT_WORKFLOW_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"' in quant_body
    assert '"${PYTHON_BIN}" -m pytest tests -q' in quant_body


def test_backend_verification_does_not_collect_embedded_quant_workflow_tests() -> None:
    script = (ROOT_DIR / "scripts" / "verify-local-quant-workbench.sh").read_text(
        encoding="utf-8"
    )
    ci_gate = (ROOT_DIR / "scripts" / "ci_gate.sh").read_text(encoding="utf-8")
    backend_body = script.split("run_backend() {", 1)[1].split("\n}\n", 1)[0]
    offline_body = ci_gate.split("offline_test_suite() {", 1)[1].split(
        "\n}", 1
    )[0]

    assert '"${PYTHON_BIN}" -m pytest tests -q' in backend_body
    assert 'python -m pytest tests "${pytest_args[@]}"' in offline_body


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
