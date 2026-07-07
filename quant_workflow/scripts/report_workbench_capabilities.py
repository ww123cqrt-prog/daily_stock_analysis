"""Generate a read-only DSA quant workbench capability inventory."""

import argparse
import ast
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import projects_root as default_projects_root  # noqa: E402
from _paths import reports_root as default_reports_root  # noqa: E402


@dataclass(frozen=True)
class CapabilityReportPaths:
    md_path: Path
    json_path: Path
    latest_md_path: Path
    latest_json_path: Path


SCRIPT_ONLY_BY_DESIGN = {
    "supervise_baostock.py": "Durable Baostock supervision is kept at script layer; DSA job center owns web orchestration.",
}

NOT_WORKFLOW_SCRIPTS = {"_paths.py"}
SPEC_FIELD_NAMES = {"label", "command", "allowed_params", "artifact_patterns", "cwd"}
NON_SCRIPT_BACKEND_JOBS = {
    "run_dsa_backtest": "dsa_backend_command",
}
TRADING_BOUNDARY = "只读研究边界：不接券商交易，不读取交易密码，不自动下单。"


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")


def _generated_at() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _find_tests(test_root: Path, needles: Iterable[str]) -> list[str]:
    if not test_root.exists():
        return []
    normalized_needles = {needle.lower() for needle in needles if needle}
    matches: list[str] = []
    for path in sorted(test_root.rglob("*test*.*")):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        haystack = (path.name + " " + path.stem).lower()
        if any(needle in haystack for needle in normalized_needles):
            matches.append(str(path))
    return matches


def _status(has_impl: bool, has_test: bool) -> str:
    if has_impl and has_test:
        return "available"
    if has_impl:
        return "missing_test"
    if has_test:
        return "missing_implementation"
    return "missing"


def _join_api_prefix(base: str, child: str) -> str:
    base = base.rstrip("/")
    child = child.strip()
    if not child:
        return base or "/"
    return f"{base}/{child.lstrip('/')}" if base else "/" + child.lstrip("/")


def _first_endpoint_route_path(module_file: Path) -> str:
    text = _read_text(module_file)
    route_match = re.search(
        r"@router\.(?:get|post|put|patch|delete|head|options)\(\s*['\"]([^'\"]+)['\"]",
        text,
    )
    return route_match.group(1) if route_match else ""


def _literal_string_set(node: ast.AST | None) -> list[str]:
    if node is None:
        return []
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        values = []
        for element in node.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                values.append(element.value)
        return sorted(set(values))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set":
        if not node.args:
            return []
        return _literal_string_set(node.args[0])
    return []


def _constant_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dict_items(node: ast.Dict) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        key_name = _constant_string(key)
        if key_name:
            values[key_name] = value
    return values


def _node_contains_string(node: ast.AST | None, needle: str) -> bool:
    if node is None:
        return False
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == needle:
            return True
    return False


def _function_nodes(tree: ast.AST | None) -> dict[str, ast.FunctionDef]:
    if tree is None:
        return {}
    return {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}


def _extract_frontend_routes(dsa_root: Path) -> list[dict[str, Any]]:
    app_file = dsa_root / "apps" / "dsa-web" / "src" / "App.tsx"
    pages_root = dsa_root / "apps" / "dsa-web" / "src" / "pages"
    tests_root = pages_root / "__tests__"
    text = _read_text(app_file)

    component_imports: dict[str, str] = {}
    for match in re.finditer(r"const\s+(\w+)\s*=\s*lazy\(\(\)\s*=>\s*import\(['\"]\.\/pages\/([^'\"]+)['\"]\)\)", text):
        component_imports[match.group(1)] = match.group(2)

    routes: list[dict[str, Any]] = []
    seen: set[str] = set()
    route_pattern = re.compile(r"<Route\s+path=['\"]([^'\"]+)['\"]\s+element=\{<(\w+)\s*/>\}", re.MULTILINE)
    for match in route_pattern.finditer(text):
        path_value = match.group(1)
        component = match.group(2)
        if path_value in seen:
            continue
        seen.add(path_value)
        page_stem = component_imports.get(component, component)
        page_file = pages_root / f"{page_stem}.tsx"
        tests = _find_tests(tests_root, [page_stem, component])
        routes.append(
            {
                "path": path_value,
                "component": component,
                "page_file": str(page_file),
                "has_page": page_file.exists(),
                "tests": tests,
                "status": "available" if tests else "missing_test",
            }
        )
    return routes


def _extract_api_routers(dsa_root: Path) -> list[dict[str, Any]]:
    app_file = dsa_root / "api" / "app.py"
    router_file = dsa_root / "api" / "v1" / "router.py"
    endpoints_root = dsa_root / "api" / "v1" / "endpoints"
    tests_root = dsa_root / "tests"
    text = _read_text(app_file)
    router_text = _read_text(router_file)

    routers: list[dict[str, Any]] = []
    seen: set[str] = set()
    direct_pattern = re.compile(r"include_router\(\s*([A-Za-z_][A-Za-z0-9_]*)\.router\s*,\s*prefix=['\"]([^'\"]+)['\"]")
    aggregate_prefix_match = re.search(r"APIRouter\(\s*prefix=['\"]([^'\"]+)['\"]", router_text)
    aggregate_prefix = aggregate_prefix_match.group(1).rstrip("/") if aggregate_prefix_match else ""
    aggregate_pattern = re.compile(
        r"include_router\(\s*([A-Za-z_][A-Za-z0-9_]*)\.router\s*(?:,\s*prefix=['\"]([^'\"]+)['\"])?",
        re.MULTILINE,
    )
    matches = [(m.group(1), m.group(2)) for m in direct_pattern.finditer(text)]
    for match in aggregate_pattern.finditer(router_text):
        module = match.group(1)
        module_file = endpoints_root / f"{module}.py"
        route_prefix = match.group(2)
        if route_prefix:
            prefix = _join_api_prefix(aggregate_prefix, route_prefix)
        else:
            prefix = _join_api_prefix(aggregate_prefix, _first_endpoint_route_path(module_file))
        matches.append((module, prefix))

    for module, prefix in matches:
        if prefix in seen:
            continue
        seen.add(prefix)
        module_file = endpoints_root / f"{module}.py"
        test_needles = {module, module.replace("_jobs", ""), prefix.rstrip("/").split("/")[-1]}
        tests = _find_tests(tests_root, test_needles)
        routers.append(
            {
                "prefix": prefix,
                "module": module,
                "module_file": str(module_file),
                "has_module": module_file.exists(),
                "tests": tests,
                "status": _status(module_file.exists(), bool(tests)),
            }
        )
    return routers


def _extract_backend_jobs(dsa_root: Path) -> dict[str, dict[str, Any]]:
    service_file = dsa_root / "src" / "services" / "quant_job_service.py"
    text = _read_text(service_file)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        tree = None
    functions = _function_nodes(tree)
    jobs: dict[str, dict[str, Any]] = {}
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_build_default_command_specs":
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                        for key, value in zip(child.value.keys, child.value.values):
                            job_type = _constant_string(key)
                            if not job_type:
                                continue
                            spec_items = _dict_items(value) if isinstance(value, ast.Dict) else {}
                            command_node = spec_items.get("command")
                            command_function = command_node.id if isinstance(command_node, ast.Name) else ""
                            command_target = functions.get(command_function)
                            forced_dry_run = _node_contains_string(value, "--dry-run") or _node_contains_string(command_target, "--dry-run")
                            jobs[job_type] = {
                                "job_type": job_type,
                                "service_file": str(service_file),
                                "allowed_params": _literal_string_set(spec_items.get("allowed_params")),
                                "forced_dry_run": forced_dry_run,
                            }
    if not jobs:
        for match in re.finditer(r"['\"]([a-z][a-z0-9_]+)['\"]\s*:\s*\{", text):
            key = match.group(1)
            if key not in SPEC_FIELD_NAMES:
                jobs[key] = {"job_type": key, "service_file": str(service_file), "allowed_params": [], "forced_dry_run": False}
    return jobs


def _extract_frontend_jobs(dsa_root: Path) -> set[str]:
    workflow_page = dsa_root / "apps" / "dsa-web" / "src" / "pages" / "WorkflowPage.tsx"
    text = _read_text(workflow_page)
    jobs = set(re.findall(r"\btype:\s*['\"]([a-z][a-z0-9_]+)['\"]", text))
    jobs.update(re.findall(r"\bjobType:\s*['\"]([a-z][a-z0-9_]+)['\"]", text))
    return jobs


def _collect_quant_workflow_scripts(
    workflow_root: Path,
    backend_jobs: dict[str, dict[str, Any]],
    frontend_jobs: set[str],
) -> list[dict[str, Any]]:
    scripts_root = workflow_root / "scripts"
    tests_root = workflow_root / "tests"
    scripts = []
    if not scripts_root.exists():
        return scripts

    for script in sorted(scripts_root.glob("*.py")):
        if script.name in NOT_WORKFLOW_SCRIPTS:
            continue
        stem = script.stem
        tests = _find_tests(tests_root, [stem])
        backend = stem in backend_jobs
        frontend = stem in frontend_jobs
        if script.name in SCRIPT_ONLY_BY_DESIGN:
            status = "script_only_by_design"
        else:
            status = "available" if backend and frontend and tests else "partial"
            if not tests:
                status = "missing_test"
            if not backend:
                status = "missing_backend_job"
            if backend and not frontend:
                status = "missing_frontend_entry"
        scripts.append(
            {
                "script": script.name,
                "path": str(script),
                "backend_job": backend,
                "frontend_entry": frontend,
                "tests": tests,
                "status": status,
                "note": SCRIPT_ONLY_BY_DESIGN.get(script.name, ""),
            }
        )
    return scripts


def _collect_workflow_jobs(
    workflow_root: Path,
    backend_jobs: dict[str, dict[str, Any]],
    frontend_jobs: set[str],
) -> list[dict[str, Any]]:
    scripts_root = workflow_root / "scripts"
    rows = []
    for job_type in sorted(backend_jobs):
        script = scripts_root / f"{job_type}.py"
        frontend = job_type in frontend_jobs
        has_script = script.exists()
        source = NON_SCRIPT_BACKEND_JOBS.get(job_type, "quant_workflow_script")
        has_executable_entry = has_script or source != "quant_workflow_script"
        status = "available" if frontend and has_executable_entry else "partial"
        if not frontend:
            status = "missing_frontend_entry"
        if not has_executable_entry:
            status = "missing_script"
        rows.append(
            {
                "job_type": job_type,
                "source": source,
                "backend_job": True,
                "frontend_entry": frontend,
                "script": str(script),
                "has_script": has_script,
                "allowed_params": backend_jobs[job_type].get("allowed_params", []),
                "forced_dry_run": bool(backend_jobs[job_type].get("forced_dry_run", False)),
                "status": status,
            }
        )
    return rows


def _collect_china_astock_reuse(projects_root: Path) -> list[dict[str, Any]]:
    root = projects_root / "china-astock-quant"
    indicators = root / "strategies" / "_indicators.py"
    strategy_files = [root / "strategies" / name for name in ("dual_ma.py", "mean_reversion.py", "momentum_rotation.py")]
    backtest_files = [root / "backtest" / name for name in ("engine.py", "portfolio.py", "metrics.py")]
    trading_dir = root / "trading"
    rows = [
        {
            "id": "indicators",
            "label": "技术指标函数",
            "paths": [str(indicators)],
            "status": "available" if indicators.exists() else "missing",
            "recommendation": "优先复用到 quant_workflow 的只读指标报告和 DSA 展示。",
        },
        {
            "id": "sample_strategies",
            "label": "样例策略",
            "paths": [str(path) for path in strategy_files],
            "status": "partial" if any(path.exists() for path in strategy_files) else "missing",
            "recommendation": "作为策略思想和参数实验参考，接入前需统一数据接口和回测口径。",
        },
        {
            "id": "backtest_engine",
            "label": "回测组件",
            "paths": [str(path) for path in backtest_files],
            "status": "partial" if any(path.exists() for path in backtest_files) else "missing",
            "recommendation": "可对照 DSA/Qlib 回测结果，不直接替换主回测入口。",
        },
        {
            "id": "paper_trading",
            "label": "交易/模拟交易模块",
            "paths": [str(trading_dir)],
            "status": "not_planned",
            "recommendation": TRADING_BOUNDARY,
        },
    ]
    return rows


def _table(headers: list[str], rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header, "")
            if isinstance(value, list):
                value = ", ".join(Path(str(item)).name for item in value)
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# DSA 本地量化工作台能力盘点",
        "",
        f"- 生成时间：`{payload['generated_at']}`",
        "- 报告性质：只读代码盘点，不启动下载、转换、清洗、训练任务。",
        f"- DSA：`{payload['paths']['daily_stock_analysis']}`",
        f"- quant_workflow：`{payload['paths']['quant_workflow']}`",
        f"- china-astock-quant：`{payload['paths']['china_astock_quant']}`",
        "",
        "## 前端页面",
        "",
    ]
    lines.extend(_table(["path", "component", "has_page", "tests", "status"], payload["frontend_routes"]))
    lines.extend(["", "## API 模块", ""])
    lines.extend(_table(["prefix", "module", "has_module", "tests", "status"], payload["api_routers"]))
    lines.extend(["", "## 工作流任务", ""])
    lines.extend(_table(["job_type", "source", "frontend_entry", "has_script", "forced_dry_run", "allowed_params", "status"], payload["workflow_jobs"]))
    lines.extend(["", "## quant_workflow 脚本", ""])
    lines.extend(_table(["script", "backend_job", "frontend_entry", "tests", "status"], payload["quant_workflow_scripts"]))
    lines.extend(["", "## china-astock-quant 可复用能力", ""])
    lines.extend(_table(["id", "label", "status", "recommendation"], payload["china_astock_reuse"]))
    lines.extend(
        [
            "",
            "## 边界",
            "",
            f"- {TRADING_BOUNDARY}",
            "- MLflow 只作为实验原始记录页，DSA 负责用户友好的摘要和入口。",
            "- 同花顺仅作为后续只读观察方向，不做交易执行。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_report(
    *,
    projects_root: str | Path | None = None,
    reports_root: str | Path | None = None,
    timestamp: str | None = None,
    generated_at: str | None = None,
) -> CapabilityReportPaths:
    projects = Path(projects_root).expanduser().resolve() if projects_root else default_projects_root().resolve()
    reports = Path(reports_root).expanduser().resolve() if reports_root else default_reports_root().resolve()
    reports.mkdir(parents=True, exist_ok=True)

    stamp = timestamp or _timestamp()
    generated = generated_at or _generated_at()
    dsa_root = projects / "daily_stock_analysis"
    workflow_root = projects / "quant_workflow"
    astock_root = projects / "china-astock-quant"

    backend_jobs = _extract_backend_jobs(dsa_root)
    frontend_jobs = _extract_frontend_jobs(dsa_root)
    payload: dict[str, Any] = {
        "artifact_kind": "workbench_capability_inventory",
        "timestamp": stamp,
        "generated_at": generated,
        "no_download_conversion_cleaning_training_started": True,
        "trading_enabled": False,
        "paths": {
            "projects_root": str(projects),
            "daily_stock_analysis": str(dsa_root),
            "quant_workflow": str(workflow_root),
            "china_astock_quant": str(astock_root),
        },
        "frontend_routes": _extract_frontend_routes(dsa_root),
        "api_routers": _extract_api_routers(dsa_root),
        "workflow_jobs": _collect_workflow_jobs(workflow_root, backend_jobs, frontend_jobs),
        "quant_workflow_scripts": _collect_quant_workflow_scripts(workflow_root, backend_jobs, frontend_jobs),
        "china_astock_reuse": _collect_china_astock_reuse(projects),
        "boundaries": {
            "no_trading": True,
            "no_secret_reading": True,
            "mlflow_is_not_primary_ui": True,
            "tonghuashun_read_only_future_direction": True,
        },
    }

    md_path = reports / f"workbench_capability_inventory_{stamp}.md"
    json_path = reports / f"workbench_capability_inventory_{stamp}.json"
    latest_md_path = reports / "workbench_capability_inventory_latest.md"
    latest_json_path = reports / "workbench_capability_inventory_latest.json"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(md_path, payload)
    shutil.copyfile(json_path, latest_json_path)
    shutil.copyfile(md_path, latest_md_path)
    return CapabilityReportPaths(
        md_path=md_path,
        json_path=json_path,
        latest_md_path=latest_md_path,
        latest_json_path=latest_json_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", default=None)
    parser.add_argument("--reports-root", "--out-dir", dest="reports_root", default=None)
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = generate_report(
        projects_root=args.projects_root,
        reports_root=args.reports_root,
        timestamp=args.timestamp,
    )
    print(paths.md_path)
    print(paths.json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
