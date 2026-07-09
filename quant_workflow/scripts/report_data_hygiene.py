"""Generate data hygiene inventory and dry-run cleanup plan.

This script is intentionally read-only.  It inventories the local A-share data
workspace and writes a dry-run cleanup plan; it never moves, deletes, downloads,
converts, cleans, trains, or edits DSA configuration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import projects_root as default_projects_root
from _paths import reports_root as default_reports_root


@dataclass(frozen=True)
class HygieneReportPaths:
    inventory_md: Path
    inventory_json: Path
    cleanup_plan_md: Path
    cleanup_plan_json: Path
    inventory_latest_md: Path
    inventory_latest_json: Path
    cleanup_plan_latest_md: Path
    cleanup_plan_latest_json: Path


RUNTIME_PATTERNS = ["*.log", "*.pid", "*_launch*.json", "*.out", "*.err"]
TEMP_SCRIPT_NAMES = {"monitor_akshare.py", "smart_retry.py"}
STATE_PATTERNS = [
    "akshare_current.json",
    "akshare_download_summary_latest.json",
    "akshare_failures*.json",
    "baostock_*codes.json",
    "baostock_current.json",
    "baostock_failures.json",
    "eastmoney_current.json",
    "eastmoney_download_summary_latest.json",
    "eastmoney_failures*.json",
    "stock_lists.json",
]
SENSITIVE_TEXT_PATTERNS = [
    re.compile(r"(?i)\b([A-Z0-9_]*(?:PASSWORD|PASSWD|TOKEN|SECRET|COOKIE|API[_-]?KEY|ACCESS[_-]?KEY)[A-Z0-9_]*)=([^\s]+)"),
    re.compile(r"(?i)(--(?:password|passwd|token|secret|cookie|api-key|access-key)\s+)([^\s]+)"),
]


def redact_sensitive_text(value: str) -> str:
    redacted = value
    for pattern in SENSITIVE_TEXT_PATTERNS:
        if pattern.pattern.startswith("(?i)\\b"):
            redacted = pattern.sub(r"\1=[REDACTED]", redacted)
        else:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def run_text(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive for missing system commands
        return 999, f"{type(exc).__name__}: {exc}"
    output = result.stdout.strip()
    if result.stderr.strip():
        output = (output + "\n" if output else "") + "STDERR: " + result.stderr.strip()
    return result.returncode, output


def directory_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    code, output = run_text(["du", "-sh", str(path)])
    if code == 0 and output:
        return output.split()[0]
    return f"unknown ({output})"


def count_lines(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def read_first_last(path: Path) -> tuple[str | None, str | None]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None, None
    if not lines:
        return None, None
    return lines[0], lines[-1]


def collect_process_matches(process_lines: Iterable[str] | None = None) -> list[str]:
    keywords = ("download", "baostock", "jqdata", "akshare", "qlib", "train_baseline", "quant_workflow")
    if process_lines is None:
        code, output = run_text(["ps", "-axo", "pid,ppid,stat,etime,command"])
        if code != 0:
            return [f"ps failed: {output}"]
        lines = output.splitlines()[1:]
    else:
        lines = list(process_lines)

    matches: list[str] = []
    for line in lines:
        lower = line.lower()
        if not any(keyword in lower for keyword in keywords):
            continue
        if any(skip in lower for skip in ("rg -i", "mobileasset", "neatdownloadmanager", "report_data_hygiene.py")):
            continue
        matches.append(redact_sensitive_text(line))
    return matches


def collect_jobs(jobs_db: Path) -> dict[str, Any]:
    status_counts: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    if not jobs_db.exists():
        return {
            "db": str(jobs_db),
            "status_counts": status_counts,
            "attention": [{"status": "missing_db", "path": str(jobs_db)}],
        }
    db_uri = jobs_db.expanduser().resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(db_uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(
            "SELECT status, job_type, count(*) AS count "
            "FROM jobs GROUP BY status, job_type ORDER BY status, job_type"
        ):
            status_counts.append(dict(row))
        for row in conn.execute(
            "SELECT job_id, job_type, status, created_at, started_at, completed_at, "
            "return_code, error FROM jobs "
            "WHERE status IN ('running','pending','failed','cancelled') "
            "ORDER BY created_at DESC LIMIT 50"
        ):
            payload = dict(row)
            if payload.get("error"):
                payload["error"] = redact_sensitive_text(str(payload["error"])[:500])
            attention.append(payload)
    finally:
        conn.close()
    return {"db": str(jobs_db), "status_counts": status_counts, "attention": attention}


def collect_parquet_datasets(parquet_root: Path) -> list[dict[str, Any]]:
    if not parquet_root.exists():
        return []
    datasets = []
    for directory in sorted(path for path in parquet_root.iterdir() if path.is_dir()):
        datasets.append(
            {
                "name": directory.name,
                "path": str(directory),
                "size": directory_size(directory),
                "parquet_file_count": len(list(directory.glob("*.parquet"))),
            }
        )
    return datasets


def collect_qlib_providers(qlib_root: Path) -> list[dict[str, Any]]:
    if not qlib_root.exists():
        return []
    providers = []
    for directory in sorted(path for path in qlib_root.iterdir() if path.is_dir()):
        first, last = read_first_last(directory / "calendars" / "day.txt")
        providers.append(
            {
                "name": directory.name,
                "path": str(directory),
                "size": directory_size(directory),
                "calendar_first": first,
                "calendar_last": last,
                "instruments_all": count_lines(directory / "instruments" / "all.txt"),
            }
        )
    return providers


def classify_root_files(quant_data: Path) -> list[dict[str, Any]]:
    if not quant_data.exists():
        return []
    files = []
    for path in sorted(item for item in quant_data.iterdir() if item.is_file()):
        categories: list[str] = []
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in RUNTIME_PATTERNS):
            categories.append("runtime_log")
        if path.name in TEMP_SCRIPT_NAMES:
            categories.append("temp_script")
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in STATE_PATTERNS):
            categories.append("archive_candidate")
        if not categories:
            categories.append("candidate")
        files.append({"path": str(path), "name": path.name, "size": directory_size(path), "categories": categories})
    return files


def add_inventory_item(
    items: list[dict[str, Any]],
    path: Path,
    category: str,
    role: str,
    reason: str,
    *,
    extra: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> None:
    categories = [category]
    for tag in tags or []:
        if tag not in categories:
            categories.append(tag)
    item = {
        "path": str(path),
        "exists": path.exists(),
        "size": directory_size(path) if path.exists() else "missing",
        "category": category,
        "categories": categories,
        "role": role,
        "reason": reason,
    }
    if extra:
        item.update(extra)
    items.append(item)


def add_plan_action(
    actions: list[dict[str, Any]],
    action: str,
    source: Path | str,
    destination: Path | str | None,
    reason: str,
    risk: str,
    requires_user_confirmation: bool,
) -> None:
    actions.append(
        {
            "action": action,
            "source": str(source),
            "destination": "" if destination is None else str(destination),
            "reason": reason,
            "risk": risk,
            "requires_user_confirmation": requires_user_confirmation,
        }
    )


def markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = [str(row.get(header, "")).replace("\n", " ") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_inventory_markdown(path: Path, inventory: dict[str, Any]) -> None:
    official_rows = [
        {"name": name, **status}
        for name, status in inventory["official_status"].items()
    ]
    category_counts: dict[str, int] = {}
    for item in inventory["items"]:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
    category_rows = [{"category": key, "count": value} for key, value in sorted(category_counts.items())]
    lines = [
        f"# A 股量化数据治理 Inventory（{inventory['timestamp']}）",
        "",
        "本报告为只读盘点。未移动、未删除、未启动下载/转换/清洗/训练任务，未读取或修改 DSA `.env`。",
        "",
        "## 现场状态",
        f"- maintenance lock: {'存在' if inventory['maintenance_lock']['exists'] else '不存在'}",
        f"- quant_data: {inventory['project_sizes']['quant_data']}",
        f"- quant_workflow: {inventory['project_sizes']['quant_workflow']}",
        f"- daily_stock_analysis: {inventory['project_sizes']['daily_stock_analysis']}",
        f"- qlib: {inventory['project_sizes']['qlib']}",
        f"- china-astock-quant: {inventory['project_sizes']['china-astock-quant']}",
        "",
        "## 官方入口复核",
        markdown_table(official_rows, ["name", "exists", "size", "path"]),
        "",
        "## 任务中心状态",
        markdown_table(inventory["jobs"]["status_counts"] or [{"status": "none", "job_type": "none", "count": 0}], ["status", "job_type", "count"]),
        "",
        "### 需关注任务",
        markdown_table(
            inventory["jobs"]["attention"] or [{"job_id": "none", "job_type": "none", "status": "none", "created_at": "", "return_code": "", "error": ""}],
            ["job_id", "job_type", "status", "created_at", "return_code", "error"],
        ),
        "",
        "## 进程匹配",
    ]
    if inventory["process_matches"]:
        lines.extend(f"- `{line}`" for line in inventory["process_matches"])
    else:
        lines.append("- 未发现匹配 download/baostock/jqdata/akshare/qlib/train_baseline/quant_workflow 的业务进程。")
    lines.extend(
        [
            "",
            "## Parquet 数据集",
            markdown_table(inventory["parquet_datasets"], ["name", "size", "parquet_file_count", "path"]),
            "",
            "## Qlib Provider",
            markdown_table(inventory["qlib_providers"], ["name", "size", "calendar_first", "calendar_last", "instruments_all", "path"]),
            "",
            "## 分类计数",
            markdown_table(category_rows, ["category", "count"]),
            "",
            "## 分类明细",
        ]
    )
    for item in inventory["items"]:
        lines.append(f"- `{item['category']}` `{item['path']}` - {item['role']}；{item['reason']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cleanup_plan_markdown(path: Path, plan: dict[str, Any]) -> None:
    lines = [
        f"# A 股量化数据治理 Dry-run 清理计划（{plan['timestamp']}）",
        "",
        "本计划只描述建议动作，不执行真实移动或删除。第一版不包含 `delete` 动作。",
        "",
        "- dry_run_only: true",
        f"- contains_delete_action: {plan['contains_delete_action']}",
        "",
        "## 建议动作",
        markdown_table(
            plan["actions"],
            ["action", "source", "destination", "reason", "risk", "requires_user_confirmation"],
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_data_hygiene_reports(
    *,
    projects_root: Path | None = None,
    reports_root: Path | None = None,
    timestamp: str | None = None,
    process_lines: Iterable[str] | None = None,
) -> HygieneReportPaths:
    root = Path(projects_root or default_projects_root()).expanduser()
    reports = Path(reports_root or default_reports_root()).expanduser()
    reports.mkdir(parents=True, exist_ok=True)

    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_data = root / "quant_data"
    quant_workflow = root / "quant_workflow"
    dsa = root / "daily_stock_analysis"
    qlib = root / "qlib"
    china = root / "china-astock-quant"
    parquet_root = quant_data / "parquet" / "ashare_1d"
    qlib_root = quant_data / "qlib"
    official_cleaned_parquet = parquet_root / "merged_baostock_jqdata_cleaned"
    official_cleaned_qlib = qlib_root / "cn_data_baostock_jqdata_cleaned"
    latest_quality_report = reports / "data_quality_report_latest.md"
    latest_cleaning_strategy = reports / "data_cleaning_strategy_latest.md"
    latest_training_result = reports / "latest_training_result.json"
    official_paths = {
        "official_cleaned_parquet": official_cleaned_parquet,
        "official_cleaned_qlib_provider": official_cleaned_qlib,
        "latest_quality_report": latest_quality_report,
        "latest_cleaning_strategy_report": latest_cleaning_strategy,
        "latest_training_result": latest_training_result,
    }

    parquet_datasets = collect_parquet_datasets(parquet_root)
    qlib_providers = collect_qlib_providers(qlib_root)
    root_files = classify_root_files(quant_data)
    items: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    add_inventory_item(items, official_cleaned_parquet, "official", "正式清洗 parquet 入口", "当前 DSA/Qlib 工作流应优先使用的清洗后 A 股日线 parquet。", tags=["do_not_touch"])
    add_inventory_item(items, official_cleaned_qlib, "official", "正式 Qlib provider", "当前训练和 Qlib 回测应使用的清洗后 provider。", tags=["do_not_touch"])
    for path in (latest_quality_report, latest_cleaning_strategy, latest_training_result):
        add_inventory_item(items, path, "do_not_touch", "latest 指针/结果文件", "当前页面和交接状态会引用，不能在清理阶段移动。")

    for dataset in parquet_datasets:
        path = Path(dataset["path"])
        if path == official_cleaned_parquet:
            continue
        if dataset["name"] in {"qfq", "qfq_full"}:
            category = "legacy"
            reason = "旧测试或旧口径数据，未作为当前正式入口。"
        else:
            category = "candidate"
            reason = "有复现和对照价值，但不是当前正式清洗入口。"
        add_inventory_item(
            items,
            path,
            category,
            f"parquet 数据集 {dataset['name']}",
            reason,
            extra={"parquet_file_count": dataset["parquet_file_count"]},
        )

    for provider in qlib_providers:
        path = Path(provider["path"])
        if path == official_cleaned_qlib:
            continue
        if provider["name"] == "cn_data_baostock_jqdata":
            category = "candidate"
            reason = "原始合并版 provider，可用于与清洗版对照。"
        else:
            category = "legacy"
            reason = "测试 provider 或短历史 provider，不是当前正式训练入口。"
        add_inventory_item(
            items,
            path,
            category,
            f"Qlib provider {provider['name']}",
            reason,
            extra={key: provider[key] for key in ("calendar_first", "calendar_last", "instruments_all")},
        )

    for root_file in root_files:
        path = Path(root_file["path"])
        categories = root_file["categories"]
        role = "quant_data 根目录文件"
        if "runtime_log" in categories:
            role = "运行日志/pid/launch 输出"
        if "temp_script" in categories:
            role = "一次性脚本"
        add_inventory_item(items, path, categories[0], role, "根目录混杂文件；本轮只盘点，后续经确认后再迁移。", tags=categories[1:])

    historical_reports = sorted(
        path
        for path in reports.glob("data_hygiene_*_20*.md")
        if not path.name.endswith(f"{ts}.md")
    )
    if historical_reports:
        items.append(
            {
                "path": str(reports / "data_hygiene_*_20*.md/json"),
                "exists": True,
                "size": f"{len(historical_reports)} markdown files plus matching json files where present",
                "category": "archive_candidate",
                "categories": ["archive_candidate"],
                "role": "历史数据治理报告",
                "reason": "多轮 timestamp 报告可保留；用户确认后可按月份归档，减少 reports 根目录噪声。",
                "sample_files": [str(path) for path in historical_reports[-5:]],
            }
        )

    add_plan_action(actions, "keep", official_cleaned_parquet, None, "正式清洗 parquet 入口，保持原路径。", "移动会导致 DSA/quant_workflow/Qlib 转换链路找不到正式数据。", False)
    add_plan_action(actions, "keep", official_cleaned_qlib, None, "正式 Qlib provider，保持原路径。", "移动会影响训练、回测和 latest_training_result 指向。", False)
    for path in (latest_quality_report, latest_cleaning_strategy, latest_training_result):
        add_plan_action(actions, "keep", path, None, "latest 指针或最新结果文件，保持原路径。", "移动会影响网页状态、交接说明或实验查看。", False)

    for name in ("qfq", "qfq_full"):
        path = parquet_root / name
        if path.exists():
            add_plan_action(actions, "archive", path, quant_data / "archive" / "parquet" / "ashare_1d" / name, "旧测试/旧口径 parquet，不是当前正式入口；用户确认后可归档。", "如果仍有隐藏脚本引用旧路径，归档前需全局引用检查。", True)
    for name in ("cn_data", "cn_data_jqdata", "cn_data_jqdata_full"):
        path = qlib_root / name
        if path.exists():
            add_plan_action(actions, "archive", path, quant_data / "archive" / "qlib" / name, "测试 provider 或短历史 provider，不是当前正式入口；用户确认后可归档。", "可能影响旧实验复现；归档前需核对 MLflow artifact 和 jobs/index.sqlite 引用。", True)
    for name in ("cn_data_baostock_jqdata",):
        path = qlib_root / name
        if path.exists():
            add_plan_action(actions, "document", path, None, "原始合并版 provider 保留作清洗版对照，先写入未来 manifest，不迁移。", "无路径变更风险；主要风险是长期未文档化导致误用。", False)
    for name in ("baostock_qfq", "jqdata", "merged_baostock_jqdata"):
        path = parquet_root / name
        if path.exists():
            add_plan_action(actions, "document", path, None, "源数据/合并前后对照数据保留，用于复现官方清洗版；先文档化，不迁移。", "无路径变更风险；若后续迁移必须更新转换脚本引用。", False)

    for root_file in root_files:
        path = Path(root_file["path"])
        categories = root_file["categories"]
        if "temp_script" in categories:
            add_plan_action(actions, "move", path, quant_workflow / "scripts" / "archive" / path.name, "一次性脚本不应放在 quant_data 根目录；用户确认后迁入脚本归档区。", "若仍有人手动依赖旧路径，迁移前需检查 README/launchd/jobs 引用。", True)
        elif "runtime_log" in categories:
            destination_base = quant_workflow / "logs" / "quant_data"
            if path.suffix == ".pid" or "launch" in path.name:
                destination_base = reports / "jobs" / "legacy-runtime"
            add_plan_action(actions, "move", path, destination_base / path.name, "运行日志、pid、launch 记录应离开数据根目录；用户确认后迁移。", "通常不影响数据读取；但 pid/launch 文件可能用于人工追踪历史任务。", True)
        elif "archive_candidate" in categories:
            add_plan_action(actions, "move", path, reports / "runtime_state" / path.name, "失败清单、空代码/超时代码等状态文件应进入 reports/runtime_state。", "通常不影响正式数据读取；迁移前需确认没有脚本硬编码读取这些根目录状态文件。", True)

    if historical_reports:
        add_plan_action(actions, "archive", reports / "data_hygiene_*_20*.md/json", reports / "archive" / "data_hygiene" / "<same-filename>", "历史 timestamp 盘点/计划报告较多；用户确认后可按专题归档。", "不影响正式数据；可能影响查找历史交接记录，所以只建议归档不删除。", True)

    generated_at = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    inventory = {
        "generated_at": generated_at,
        "timestamp": ts,
        "policy": {
            "dry_run_only": True,
            "no_move_delete_cleanup_training_download_started": True,
            "do_not_modify_env": True,
        },
        "maintenance_lock": {
            "path": str(quant_data / ".maintenance.lock"),
            "exists": (quant_data / ".maintenance.lock").exists(),
            "content": (quant_data / ".maintenance.lock").read_text(encoding="utf-8", errors="replace")[:2000]
            if (quant_data / ".maintenance.lock").exists()
            else "",
        },
        "project_sizes": {
            "quant_data": directory_size(quant_data),
            "quant_workflow": directory_size(quant_workflow),
            "daily_stock_analysis": directory_size(dsa),
            "qlib": directory_size(qlib),
            "china-astock-quant": directory_size(china),
        },
        "jobs": collect_jobs(reports / "jobs" / "index.sqlite"),
        "process_matches": collect_process_matches(process_lines),
        "official_status": {
            name: {"path": str(path), "exists": path.exists(), "size": directory_size(path) if path.exists() else "missing"}
            for name, path in official_paths.items()
        },
        "parquet_datasets": parquet_datasets,
        "qlib_providers": qlib_providers,
        "quant_data_root_files": root_files,
        "items": items,
    }
    plan = {
        "generated_at": generated_at,
        "timestamp": ts,
        "dry_run_only": True,
        "contains_delete_action": any(action["action"] == "delete" for action in actions),
        "requires_confirmation_summary": "All move/archive actions require explicit user confirmation before execution. This plan does not execute them.",
        "actions": actions,
    }

    paths = HygieneReportPaths(
        inventory_md=reports / f"data_hygiene_inventory_{ts}.md",
        inventory_json=reports / f"data_hygiene_inventory_{ts}.json",
        cleanup_plan_md=reports / f"data_hygiene_cleanup_plan_{ts}.md",
        cleanup_plan_json=reports / f"data_hygiene_cleanup_plan_{ts}.json",
        inventory_latest_md=reports / "data_hygiene_inventory_latest.md",
        inventory_latest_json=reports / "data_hygiene_inventory_latest.json",
        cleanup_plan_latest_md=reports / "data_hygiene_cleanup_plan_latest.md",
        cleanup_plan_latest_json=reports / "data_hygiene_cleanup_plan_latest.json",
    )
    write_json(paths.inventory_json, inventory)
    write_json(paths.cleanup_plan_json, plan)
    write_inventory_markdown(paths.inventory_md, inventory)
    write_cleanup_plan_markdown(paths.cleanup_plan_md, plan)
    paths.inventory_latest_json.write_text(paths.inventory_json.read_text(encoding="utf-8"), encoding="utf-8")
    paths.cleanup_plan_latest_json.write_text(paths.cleanup_plan_json.read_text(encoding="utf-8"), encoding="utf-8")
    paths.inventory_latest_md.write_text(paths.inventory_md.read_text(encoding="utf-8"), encoding="utf-8")
    paths.cleanup_plan_latest_md.write_text(paths.cleanup_plan_md.read_text(encoding="utf-8"), encoding="utf-8")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate read-only data hygiene inventory and dry-run cleanup plan.")
    parser.add_argument("--projects-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Report output directory. Defaults to quant_workflow/reports.")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_data_hygiene_reports(
        projects_root=args.projects_root,
        reports_root=args.out_dir,
        timestamp=args.timestamp,
    )
    print("Generated data hygiene inventory and dry-run cleanup plan:")
    for path in (paths.inventory_md, paths.inventory_json, paths.cleanup_plan_md, paths.cleanup_plan_json):
        print(path)


if __name__ == "__main__":
    main()
