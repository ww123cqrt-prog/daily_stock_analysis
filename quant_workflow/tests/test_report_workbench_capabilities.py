import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "report_workbench_capabilities.py"
    spec = importlib.util.spec_from_file_location("report_workbench_capabilities", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ReportWorkbenchCapabilitiesTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_generates_inventory_without_running_workflow_tasks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            projects = root / "WorkingProjects"
            workflow = projects / "quant_workflow"
            dsa = projects / "daily_stock_analysis"
            astock = projects / "china-astock-quant"
            reports = workflow / "reports"

            (dsa / "apps/dsa-web/src/pages/__tests__").mkdir(parents=True)
            (dsa / "api/v1/endpoints").mkdir(parents=True)
            (dsa / "tests").mkdir(parents=True)
            (dsa / "src/services").mkdir(parents=True)
            (workflow / "scripts").mkdir(parents=True)
            (workflow / "tests").mkdir(parents=True)
            (astock / "strategies").mkdir(parents=True)
            (astock / "backtest").mkdir(parents=True)
            (astock / "trading").mkdir(parents=True)

            (dsa / "apps/dsa-web/src/App.tsx").write_text(
                "const HomePage = lazy(() => import('./pages/HomePage'));\n"
                "<Route path=\"/\" element={<HomePage />} />\n"
                "const WorkflowPage = lazy(() => import('./pages/WorkflowPage'));\n"
                "<Route path=\"/workflow\" element={<WorkflowPage />} />\n"
                "const StocksPage = lazy(() => import('./pages/StocksPage'));\n"
                "<Route path=\"/stocks\" element={<StocksPage />} />\n",
                encoding="utf-8",
            )
            (dsa / "apps/dsa-web/src/pages/HomePage.tsx").write_text("export default function HomePage() { return null }\n", encoding="utf-8")
            (dsa / "apps/dsa-web/src/pages/WorkflowPage.tsx").write_text(
                "const WORKFLOW_STEPS = [{ jobs: [{ type: 'check_qlib_data' }] }]\n",
                encoding="utf-8",
            )
            (dsa / "apps/dsa-web/src/pages/__tests__/HomePage.test.tsx").write_text("test('home', () => {})\n", encoding="utf-8")
            (dsa / "apps/dsa-web/src/pages/__tests__/WorkflowPage.test.tsx").write_text("test('workflow', () => {})\n", encoding="utf-8")
            (dsa / "api/app.py").write_text(
                "app.include_router(quant_jobs.router, prefix=\"/api/v1/quant\")\n"
                "app.include_router(stocks.router, prefix=\"/api/v1/stocks\")\n",
                encoding="utf-8",
            )
            (dsa / "api/v1/endpoints/quant_jobs.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "api/v1/endpoints/stocks.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "tests/test_quant_jobs_api.py").write_text("def test_quant(): pass\n", encoding="utf-8")
            (dsa / "src/services/quant_job_service.py").write_text(
                '"check_qlib_data": {"label": "检查 Qlib 数据", "allowed_params": set()},\n',
                encoding="utf-8",
            )

            (workflow / "scripts/check_qlib_data.py").write_text("print('ok')\n", encoding="utf-8")
            (workflow / "scripts/supervise_baostock.py").write_text("print('supervise')\n", encoding="utf-8")
            (workflow / "tests/test_check_qlib_data.py").write_text("def test_check(): pass\n", encoding="utf-8")
            (workflow / "tests/test_supervise_baostock.py").write_text("def test_supervise(): pass\n", encoding="utf-8")

            (astock / "strategies/_indicators.py").write_text("def rsi(): pass\n", encoding="utf-8")
            (astock / "strategies/dual_ma.py").write_text("class DualMA: pass\n", encoding="utf-8")
            (astock / "backtest/engine.py").write_text("class BacktestEngine: pass\n", encoding="utf-8")
            (astock / "trading/easytrader_broker.py").write_text("class Broker: pass\n", encoding="utf-8")

            paths = self.module.generate_report(
                projects_root=projects,
                reports_root=reports,
                timestamp="20260706_120000",
                generated_at="2026-07-06T12:00:00+08:00",
            )

            data = json.loads(paths.json_path.read_text(encoding="utf-8"))
            self.assertTrue(data["no_download_conversion_cleaning_training_started"])
            self.assertEqual(data["artifact_kind"], "workbench_capability_inventory")

            route_by_path = {item["path"]: item for item in data["frontend_routes"]}
            self.assertEqual(route_by_path["/"]["status"], "available")
            self.assertEqual(route_by_path["/stocks"]["status"], "missing_test")

            api_by_prefix = {item["prefix"]: item for item in data["api_routers"]}
            self.assertEqual(api_by_prefix["/api/v1/quant"]["status"], "available")
            self.assertEqual(api_by_prefix["/api/v1/stocks"]["status"], "missing_test")

            script_by_name = {item["script"]: item for item in data["quant_workflow_scripts"]}
            self.assertEqual(script_by_name["check_qlib_data.py"]["status"], "available")
            self.assertEqual(script_by_name["supervise_baostock.py"]["status"], "script_only_by_design")

            job_by_type = {item["job_type"]: item for item in data["workflow_jobs"]}
            self.assertEqual(job_by_type["check_qlib_data"]["status"], "available")

            reuse_by_id = {item["id"]: item for item in data["china_astock_reuse"]}
            self.assertEqual(reuse_by_id["indicators"]["status"], "available")
            self.assertEqual(reuse_by_id["sample_strategies"]["status"], "partial")
            self.assertEqual(reuse_by_id["paper_trading"]["status"], "not_planned")

            self.assertTrue(paths.md_path.exists())
            self.assertTrue((reports / "workbench_capability_inventory_latest.json").exists())
            md = paths.md_path.read_text(encoding="utf-8")
            self.assertIn("DSA 本地量化工作台能力盘点", md)
            self.assertIn("不启动下载、转换、清洗、训练任务", md)
            self.assertIn("supervise_baostock.py", md)

    def test_handles_aggregated_api_router_and_filters_parser_noise(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            projects = root / "WorkingProjects"
            workflow = projects / "quant_workflow"
            dsa = projects / "daily_stock_analysis"
            reports = workflow / "reports"

            (dsa / "api/v1/endpoints").mkdir(parents=True)
            (dsa / "api/v1").mkdir(parents=True, exist_ok=True)
            (dsa / "tests").mkdir(parents=True)
            (dsa / "src/services").mkdir(parents=True)
            (dsa / "apps/dsa-web/src/pages").mkdir(parents=True)
            (workflow / "scripts").mkdir(parents=True)
            (workflow / "tests/__pycache__").mkdir(parents=True)

            (dsa / "api/v1/router.py").write_text(
                "router = APIRouter(prefix=\"/api/v1\")\n"
                "router.include_router(quant_jobs.router, prefix=\"/quant\")\n"
                "router.include_router(stocks.router, prefix=\"/stocks\")\n",
                encoding="utf-8",
            )
            (dsa / "api/v1/endpoints/quant_jobs.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "api/v1/endpoints/stocks.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "tests/test_quant_jobs_api.py").write_text("def test_quant(): pass\n", encoding="utf-8")
            (dsa / "apps/dsa-web/src/pages/WorkflowPage.tsx").write_text(
                "const WORKFLOW_STEPS = [{ jobs: [{ type: 'report_workbench_capabilities' }] }]\n",
                encoding="utf-8",
            )
            (dsa / "src/services/quant_job_service.py").write_text(
                "def _build_default_command_specs():\n"
                "    def generate_csi300(params):\n"
                "        return _script_command(\"generate_csi300.py\") + [\"--dry-run\"]\n\n"
                "    return {\n"
                "        \"download_baostock\": {\n"
                "            \"label\": \"下载 Baostock 长历史\",\n"
                "            \"allowed_params\": {\n"
                "                \"start\",\n"
                "                \"end\",\n"
                "                \"limit\",\n"
                "            },\n"
                "        },\n"
                "        \"generate_csi300\": {\n"
                "            \"label\": \"预演生成 CSI300 股票池\",\n"
                "            \"command\": generate_csi300,\n"
                "            \"allowed_params\": set(),\n"
                "        },\n"
                "        \"report_workbench_capabilities\": {\n"
                "            \"label\": \"生成工作台能力盘点\",\n"
                "            \"allowed_params\": set(),\n"
                "        },\n"
                "        \"run_dsa_backtest\": {\n"
                "            \"label\": \"运行 DSA 分析准确性回测\",\n"
                "            \"command\": run_dsa_backtest,\n"
                "            \"allowed_params\": {\"code\", \"eval_window_days\", \"force\"},\n"
                "        },\n"
                "    }\n",
                encoding="utf-8",
            )
            (dsa / "apps/dsa-web/src/pages/WorkflowPage.tsx").write_text(
                "const WORKFLOW_STEPS = [{ jobs: [\n"
                "  { type: 'download_baostock' },\n"
                "  { type: 'generate_csi300' },\n"
                "  { type: 'report_workbench_capabilities' },\n"
                "  { type: 'run_dsa_backtest' },\n"
                "] }]\n",
                encoding="utf-8",
            )
            (workflow / "scripts/report_workbench_capabilities.py").write_text("print('ok')\n", encoding="utf-8")
            (workflow / "scripts/download_baostock.py").write_text("print('download')\n", encoding="utf-8")
            (workflow / "scripts/generate_csi300.py").write_text("print('csi300')\n", encoding="utf-8")
            (workflow / "tests/test_report_workbench_capabilities.py").write_text("def test_report(): pass\n", encoding="utf-8")
            (workflow / "tests/__pycache__/test_report_workbench_capabilities.pyc").write_bytes(b"cache")

            paths = self.module.generate_report(
                projects_root=projects,
                reports_root=reports,
                timestamp="20260706_130000",
                generated_at="2026-07-06T13:00:00+08:00",
            )

            data = json.loads(paths.json_path.read_text(encoding="utf-8"))
            api_by_prefix = {item["prefix"]: item for item in data["api_routers"]}
            self.assertEqual(api_by_prefix["/api/v1/quant"]["status"], "available")
            self.assertEqual(api_by_prefix["/api/v1/stocks"]["status"], "missing_test")

            self.assertNotIn("allowed_params", {item["job_type"] for item in data["workflow_jobs"]})
            jobs = {item["job_type"]: item for item in data["workflow_jobs"]}
            self.assertEqual(jobs["download_baostock"]["allowed_params"], ["end", "limit", "start"])
            self.assertTrue(jobs["generate_csi300"]["forced_dry_run"])
            self.assertEqual(jobs["report_workbench_capabilities"]["allowed_params"], [])
            self.assertFalse(jobs["run_dsa_backtest"]["has_script"])
            self.assertEqual(jobs["run_dsa_backtest"]["source"], "dsa_backend_command")
            self.assertEqual(jobs["run_dsa_backtest"]["status"], "available")

            script = {item["script"]: item for item in data["quant_workflow_scripts"]}["report_workbench_capabilities.py"]
            self.assertEqual([Path(path).suffix for path in script["tests"]], [".py"])

    def test_api_router_status_requires_module_and_tests(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            projects = root / "WorkingProjects"
            workflow = projects / "quant_workflow"
            dsa = projects / "daily_stock_analysis"
            reports = workflow / "reports"

            (dsa / "api/v1").mkdir(parents=True)
            (dsa / "api/v1/endpoints").mkdir(parents=True)
            (dsa / "tests").mkdir(parents=True)
            (dsa / "apps/dsa-web/src/pages").mkdir(parents=True)
            (dsa / "src/services").mkdir(parents=True)
            (workflow / "scripts").mkdir(parents=True)
            (workflow / "tests").mkdir(parents=True)

            (dsa / "api/v1/router.py").write_text(
                "router = APIRouter(prefix=\"/api/v1\")\n"
                "router.include_router(stocks.router, prefix=\"/stocks\")\n"
                "router.include_router(alerts.router, prefix=\"/alerts\")\n"
                "router.include_router(history.router, prefix=\"/history\")\n",
                encoding="utf-8",
            )
            (dsa / "api/v1/endpoints/alerts.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "api/v1/endpoints/history.py").write_text("router = object()\n", encoding="utf-8")
            (dsa / "tests/test_stocks_api.py").write_text("def test_stocks(): pass\n", encoding="utf-8")
            (dsa / "tests/test_alerts_api.py").write_text("def test_alerts(): pass\n", encoding="utf-8")

            paths = self.module.generate_report(
                projects_root=projects,
                reports_root=reports,
                timestamp="20260706_140000",
                generated_at="2026-07-06T14:00:00+08:00",
            )

            data = json.loads(paths.json_path.read_text(encoding="utf-8"))
            api_by_prefix = {item["prefix"]: item for item in data["api_routers"]}
            self.assertEqual(api_by_prefix["/api/v1/stocks"]["status"], "missing_implementation")
            self.assertEqual(api_by_prefix["/api/v1/alerts"]["status"], "available")
            self.assertEqual(api_by_prefix["/api/v1/history"]["status"], "missing_test")

    def test_api_router_inventory_includes_no_prefix_health_router(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            projects = root / "WorkingProjects"
            workflow = projects / "quant_workflow"
            dsa = projects / "daily_stock_analysis"
            reports = workflow / "reports"

            (dsa / "api/v1").mkdir(parents=True)
            (dsa / "api/v1/endpoints").mkdir(parents=True)
            (dsa / "tests").mkdir(parents=True)
            (dsa / "apps/dsa-web/src/pages").mkdir(parents=True)
            (dsa / "src/services").mkdir(parents=True)
            (workflow / "scripts").mkdir(parents=True)
            (workflow / "tests").mkdir(parents=True)

            (dsa / "api/v1/router.py").write_text(
                "router = APIRouter(prefix=\"/api/v1\")\n"
                "router.include_router(\n"
                "    health.router,\n"
                "    tags=[\"Health\"],\n"
                ")\n",
                encoding="utf-8",
            )
            (dsa / "api/v1/endpoints/health.py").write_text(
                "@router.get(\"/health\")\n"
                "def health_check():\n"
                "    return {\"status\": \"ok\"}\n",
                encoding="utf-8",
            )
            (dsa / "tests/test_api_health.py").write_text("def test_health(): pass\n", encoding="utf-8")

            paths = self.module.generate_report(
                projects_root=projects,
                reports_root=reports,
                timestamp="20260706_150000",
                generated_at="2026-07-06T15:00:00+08:00",
            )

            data = json.loads(paths.json_path.read_text(encoding="utf-8"))
            api_by_prefix = {item["prefix"]: item for item in data["api_routers"]}
            self.assertEqual(api_by_prefix["/api/v1/health"]["module"], "health")
            self.assertEqual(api_by_prefix["/api/v1/health"]["status"], "available")


if __name__ == "__main__":
    unittest.main()
