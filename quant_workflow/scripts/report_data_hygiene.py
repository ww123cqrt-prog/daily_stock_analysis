"""Generate data hygiene inventory and dry-run cleanup plan.

This script is intentionally read-only.  It inventories the local A-share data
workspace and writes a dry-run cleanup plan; it never moves, deletes, downloads,
converts, cleans, trains, or edits DSA configuration.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
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

from _paths import projects_root as default_projects_root  # noqa: E402
from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from _paths import reports_root as default_reports_root  # noqa: E402
from _robust_membership_integrity import read_membership_snapshot  # noqa: E402
from _robust_reference_integrity import (  # noqa: E402
    reference_snapshot_fingerprint,
    validate_reference_source,
)
from quant_data_contract import (  # noqa: E402
    DataContractError,
    atomic_write_json,
    filesystem_state_fingerprint,
    resolve_dataset,
)


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
    bundle_latest_json: Path


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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROTECTED_ROOT_FILES = {"DATA_MANIFEST.json", "README.md"}
HYGIENE_STATE_SCAN_MAX_ENTRIES = 200_000
HYGIENE_STATE_SCAN_MAX_SECONDS = 10.0


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload


def _read_catalog(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return payload, hashlib.sha256(raw).hexdigest()


def _safe_catalog_child(root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")
    relative = Path(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"{field} must be a safe relative path")
    root = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes quant_data root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field} cannot use symlinks")
    return candidate


def _catalog_date_range(entry: dict[str, Any]) -> tuple[str, str]:
    value = entry.get("date_range")
    if not isinstance(value, dict):
        raise ValueError("canonical fixed-path date range is invalid")
    start = value.get("start_date") or value.get("start")
    end = value.get("end_date") or value.get("end")
    if not isinstance(start, str) or not start or not isinstance(end, str) or not end:
        raise ValueError("canonical fixed-path date range is invalid")
    try:
        start_date = dt.date.fromisoformat(start)
        end_date = dt.date.fromisoformat(end)
    except ValueError as exc:
        raise ValueError("canonical fixed-path date range is invalid") from exc
    if start_date > end_date:
        raise ValueError("canonical fixed-path date range is invalid")
    return start_date.isoformat(), end_date.isoformat()


def _validate_fixed_path_dataset(
    dataset_id: str,
    fixed_path: Path,
    entry: dict[str, Any],
) -> str:
    expected_kind = {
        "pit_csi800": "point_in_time_membership",
        "robust_reference": "point_in_time_reference",
    }.get(dataset_id)
    if expected_kind is None or entry.get("data_kind") != expected_kind:
        raise ValueError("canonical fixed-path data kind is invalid")
    if entry.get("price_basis") != "not_applicable":
        raise ValueError("canonical fixed-path price basis is invalid")
    expected_hash = str(entry.get("snapshot_hash") or "")
    if not SHA256_RE.fullmatch(expected_hash):
        raise ValueError("canonical fixed-path snapshot hash is invalid")
    start_date, end_date = _catalog_date_range(entry)
    if dataset_id == "pit_csi800":
        _metadata, actual_hash = read_membership_snapshot(
            fixed_path,
            start_date=start_date,
            end_date=end_date,
        )
    elif dataset_id == "robust_reference":
        validate_reference_source(
            fixed_path,
            start_date=start_date,
            end_date=end_date,
        )
        actual_hash = reference_snapshot_fingerprint(fixed_path)
    if actual_hash != expected_hash:
        raise ValueError("canonical fixed-path snapshot hash mismatch")
    return actual_hash


def collect_v2_catalog(quant_data: Path) -> dict[str, Any]:
    """Inventory V2 entries and prove each canonical-ready snapshot identity."""

    catalog_path = quant_data / "DATA_MANIFEST.json"
    base = {
        "path": str(catalog_path),
        "status": "missing",
        "schema_version": None,
        "validation_contract_version": 2,
        "path_format": None,
        "manifest_sha256": None,
        "dataset_count": 0,
        "canonical_ready_count": 0,
        "validated_dataset_ids": [],
        "datasets": [],
        "errors": [],
    }
    if not catalog_path.is_file() or catalog_path.is_symlink():
        return base
    try:
        payload, manifest_sha256 = _read_catalog(catalog_path)
    except ValueError as exc:
        return {**base, "status": "invalid", "errors": [str(exc)]}
    base["manifest_sha256"] = manifest_sha256
    base["schema_version"] = payload.get("schema_version")
    base["path_format"] = payload.get("path_format")
    datasets = payload.get("datasets")
    if (
        payload.get("schema_version") != 2
        or payload.get("path_format") != "relative_to_quant_data_root"
        or not isinstance(datasets, dict)
    ):
        return {
            **base,
            "status": "invalid",
            "errors": ["DATA_MANIFEST.json must use schema_version 2 and root-relative paths"],
        }

    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for key, raw_entry in sorted(datasets.items()):
        if not isinstance(raw_entry, dict):
            errors.append(f"{key}: dataset entry must be an object")
            continue
        dataset_id = str(raw_entry.get("dataset_id") or key)
        record: dict[str, Any] = {
            "dataset_id": dataset_id,
            "data_kind": str(raw_entry.get("data_kind") or ""),
            "role": str(raw_entry.get("role") or ""),
            "status": str(raw_entry.get("status") or ""),
            "lifecycle": str(raw_entry.get("lifecycle") or ""),
            "price_basis": str(raw_entry.get("price_basis") or ""),
            "dataset_path": "",
            "snapshot_path": "",
            "snapshot_id": str(raw_entry.get("snapshot_id") or ""),
            "storage_contract": "",
            "canonical_valid": False,
            "error": "",
        }
        registered_path = raw_entry.get("dataset_root") or raw_entry.get("path")
        try:
            if registered_path:
                record["dataset_path"] = str(
                    _safe_catalog_child(quant_data, registered_path, "dataset path")
                )
        except ValueError as exc:
            record["error"] = str(exc)

        if raw_entry.get("dataset_id") != key:
            record["error"] = "catalog key and dataset_id differ"

        if (
            not record["error"]
            and record["role"] == "canonical"
            and record["status"] == "ready"
        ):
            try:
                if (
                    raw_entry.get("snapshot_available") is False
                    and raw_entry.get("path")
                    and not raw_entry.get("dataset_root")
                ):
                    fixed_path = _safe_catalog_child(
                        quant_data, raw_entry.get("path"), "canonical path"
                    )
                    if not fixed_path.exists() or fixed_path.is_symlink():
                        raise ValueError("canonical fixed-path dataset is unavailable")
                    state_before_validation = filesystem_state_fingerprint(
                        fixed_path,
                        max_entries=HYGIENE_STATE_SCAN_MAX_ENTRIES,
                        max_seconds=HYGIENE_STATE_SCAN_MAX_SECONDS,
                    )
                    snapshot_hash = _validate_fixed_path_dataset(
                        dataset_id,
                        fixed_path,
                        raw_entry,
                    )
                    state_after_validation = filesystem_state_fingerprint(
                        fixed_path,
                        max_entries=HYGIENE_STATE_SCAN_MAX_ENTRIES,
                        max_seconds=HYGIENE_STATE_SCAN_MAX_SECONDS,
                    )
                    if state_before_validation != state_after_validation:
                        raise ValueError("canonical snapshot changed during validation")
                    record.update(
                        {
                            "dataset_path": str(fixed_path),
                            "snapshot_path": str(fixed_path),
                            "snapshot_id": snapshot_hash,
                            "storage_contract": "catalog_registered_fixed_path",
                            "state_fingerprint": state_after_validation,
                            "canonical_valid": True,
                            "error": "",
                        }
                    )
                    records.append(record)
                    continue
                dataset_root = _safe_catalog_child(
                    quant_data, raw_entry.get("dataset_root"), "dataset_root"
                )
                latest_path = _safe_catalog_child(
                    quant_data, raw_entry.get("latest_pointer"), "latest_pointer"
                )
                if latest_path != dataset_root / "LATEST.json":
                    raise ValueError("LATEST pointer is outside the dataset root")
                latest_payload, latest_pointer_sha256 = _read_catalog(latest_path)
                snapshot_before_validation = _safe_catalog_child(
                    quant_data,
                    latest_payload.get("snapshot_path"),
                    "LATEST snapshot_path",
                )
                state_before_validation = filesystem_state_fingerprint(
                    snapshot_before_validation,
                    max_entries=HYGIENE_STATE_SCAN_MAX_ENTRIES,
                    max_seconds=HYGIENE_STATE_SCAN_MAX_SECONDS,
                )
                resolved = resolve_dataset(
                    key,
                    quant_data_root=quant_data,
                )
                snapshot = resolved.snapshot_path.resolve()
                manifest = resolved.manifest
                snapshot_id = str(manifest.get("snapshot_id") or "")
                if (
                    snapshot.parent != (dataset_root / "snapshots").resolve()
                    or snapshot.name != snapshot_id
                    or snapshot.is_symlink()
                    or not snapshot.is_dir()
                ):
                    raise ValueError("LATEST must select one immutable snapshot directory")
                if raw_entry.get("snapshot_id") != snapshot_id:
                    raise ValueError("catalog and LATEST snapshot IDs differ")
                if raw_entry.get("data_kind") != manifest.get("data_kind"):
                    raise ValueError("catalog and snapshot data kind differ")
                for field, expected in (
                    ("schema_version", 2),
                    ("dataset_id", dataset_id),
                    ("snapshot_id", snapshot_id),
                    ("role", "canonical"),
                    ("status", "ready"),
                ):
                    if manifest.get(field) != expected:
                        raise ValueError(f"snapshot manifest {field} is invalid")
                if manifest.get("price_basis") != raw_entry.get("price_basis"):
                    raise ValueError("catalog and snapshot price basis differ")
                state_after_validation = filesystem_state_fingerprint(
                    snapshot,
                    max_entries=HYGIENE_STATE_SCAN_MAX_ENTRIES,
                    max_seconds=HYGIENE_STATE_SCAN_MAX_SECONDS,
                )
                if (
                    snapshot_before_validation != snapshot
                    or state_before_validation != state_after_validation
                ):
                    raise ValueError("canonical snapshot changed during validation")
                _latest_after, latest_pointer_sha256_after = _read_catalog(latest_path)
                if latest_pointer_sha256_after != latest_pointer_sha256:
                    raise ValueError("LATEST pointer changed during validation")
                record.update(
                    {
                        "dataset_path": str(dataset_root),
                        "snapshot_path": str(snapshot),
                        "snapshot_id": snapshot_id,
                        "storage_contract": "immutable_snapshot_v2",
                        "latest_pointer_path": str(latest_path),
                        "latest_pointer_sha256": latest_pointer_sha256,
                        "state_fingerprint": state_after_validation,
                        "canonical_valid": True,
                        "error": "",
                    }
                )
            except (DataContractError, OSError, TypeError, ValueError) as exc:
                record["error"] = str(exc)

        if record["error"]:
            errors.append(f"{dataset_id}: {record['error']}")
        records.append(record)

    canonical_ready_count = sum(
        1 for record in records if record["canonical_valid"]
    )
    validated_dataset_ids = sorted(
        record["dataset_id"] for record in records if record["canonical_valid"]
    )
    return {
        **base,
        "status": "available" if not errors else "partial",
        "schema_version": 2,
        "path_format": "relative_to_quant_data_root",
        "dataset_count": len(records),
        "canonical_ready_count": canonical_ready_count,
        "validated_dataset_ids": validated_dataset_ids,
        "datasets": records,
        "errors": errors,
    }


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
    atomic_write_json(path, payload)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def atomic_write_text(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


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
    atomic_write_text(path, "\n".join(lines) + "\n")


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
    atomic_write_text(path, "\n".join(lines) + "\n")


def generate_data_hygiene_reports(
    *,
    projects_root: Path | None = None,
    quant_data_root: Path | None = None,
    reports_root: Path | None = None,
    timestamp: str | None = None,
    process_lines: Iterable[str] | None = None,
) -> HygieneReportPaths:
    explicit_projects_root = projects_root is not None
    root = Path(projects_root or default_projects_root()).expanduser()
    if quant_data_root is not None:
        quant_data = Path(quant_data_root).expanduser()
    elif explicit_projects_root:
        quant_data = root / "quant_data"
    else:
        quant_data = Path(default_quant_data_root()).expanduser().resolve()
    reports = Path(reports_root or default_reports_root()).expanduser()
    reports.mkdir(parents=True, exist_ok=True)

    ts = timestamp or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if not re.fullmatch(r"\d{8}_\d{6}", ts):
        raise ValueError("timestamp must use YYYYMMDD_HHMMSS format")
    quant_workflow = root / "quant_workflow"
    dsa = root / "daily_stock_analysis"
    qlib = root / "qlib"
    china = root / "china-astock-quant"
    parquet_root = quant_data / "parquet" / "ashare_1d"
    qlib_root = quant_data / "qlib"
    latest_quality_report_pointer = reports / "data_quality_report_latest.json"
    latest_cleaning_strategy = reports / "data_cleaning_strategy_latest.md"
    latest_training_result = reports / "latest_training_result.json"
    catalog = collect_v2_catalog(quant_data)
    canonical_datasets = [
        entry for entry in catalog["datasets"] if entry["canonical_valid"]
    ]
    official_paths = {
        **{
            entry["dataset_id"]: Path(entry["snapshot_path"])
            for entry in canonical_datasets
        },
        "latest_quality_report_pointer": latest_quality_report_pointer,
        "latest_cleaning_strategy_report": latest_cleaning_strategy,
        "latest_training_result": latest_training_result,
    }

    parquet_datasets = collect_parquet_datasets(parquet_root)
    qlib_providers = collect_qlib_providers(qlib_root)
    root_files = classify_root_files(quant_data)
    items: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    registered_paths: dict[Path, dict[str, Any]] = {}
    for entry in catalog["datasets"]:
        for field in ("dataset_path", "snapshot_path"):
            value = entry.get(field)
            if value:
                registered_paths[Path(value).resolve()] = entry

    for entry in canonical_datasets:
        snapshot = Path(entry["snapshot_path"])
        immutable = entry["storage_contract"] == "immutable_snapshot_v2"
        add_inventory_item(
            items,
            snapshot,
            "official",
            (
                f"原生 V2 正式快照 {entry['dataset_id']}"
                if immutable
                else f"原生 V2 正式参考数据 {entry['dataset_id']}"
            ),
            (
                "由 DATA_MANIFEST.json 和 LATEST.json 锁定的 canonical ready 不可变快照。"
                if immutable
                else "由 DATA_MANIFEST.json 登记的 canonical ready 固定路径参考数据。"
            ),
            extra={
                "dataset_id": entry["dataset_id"],
                "data_kind": entry["data_kind"],
                "snapshot_id": entry["snapshot_id"],
                "price_basis": entry["price_basis"],
                "storage_contract": entry["storage_contract"],
            },
            tags=["do_not_touch"],
        )
    for path in (
        latest_quality_report_pointer,
        latest_cleaning_strategy,
        latest_training_result,
    ):
        add_inventory_item(items, path, "do_not_touch", "latest 指针/结果文件", "当前页面和交接状态会引用，不能在清理阶段移动。")

    for dataset in parquet_datasets:
        path = Path(dataset["path"])
        registered = registered_paths.get(path.resolve())
        if registered and registered["canonical_valid"]:
            continue
        if registered and registered["role"] == "legacy":
            category = "legacy"
            reason = "原生 V2 清单登记的兼容输出，不是正式训练入口。"
        elif registered and registered["role"] == "audit":
            category = "candidate"
            reason = "旧采集审计来源，只用于核对，不是正式训练入口。"
        elif dataset["name"] in {"qfq", "qfq_full"}:
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
        registered = registered_paths.get(path.resolve())
        if registered and registered["canonical_valid"]:
            continue
        if registered and registered["role"] == "legacy":
            category = "legacy"
            reason = "原生 V2 清单登记的兼容 provider，不是正式训练入口。"
        elif registered and registered["role"] == "audit":
            category = "candidate"
            reason = "旧审计 provider，只用于核对，不是正式训练入口。"
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
        if path.name in PROTECTED_ROOT_FILES:
            add_inventory_item(
                items,
                path,
                "do_not_touch",
                "原生 V2 数据目录说明/清单",
                "解析正式快照和维护数据契约所需，不能归档或移动。",
            )
            continue
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

    add_plan_action(
        actions,
        "keep",
        quant_data / "DATA_MANIFEST.json",
        None,
        "原生 V2 根清单，保持原路径。",
        "移动会使所有按清单解析的消费者停止工作。",
        False,
    )
    for entry in canonical_datasets:
        immutable = entry["storage_contract"] == "immutable_snapshot_v2"
        add_plan_action(
            actions,
            "keep",
            entry["snapshot_path"],
            None,
            (
                f"原生 V2 正式快照 {entry['dataset_id']}，保持不可变。"
                if immutable
                else f"原生 V2 正式参考数据 {entry['dataset_id']}，保持原路径。"
            ),
            (
                "移动或修改会使 LATEST 指针和清单哈希失效。"
                if immutable
                else "移动会使历史时点股票池或鲁棒训练参考数据失效。"
            ),
            False,
        )
    for path in (
        latest_quality_report_pointer,
        latest_cleaning_strategy,
        latest_training_result,
    ):
        add_plan_action(actions, "keep", path, None, "latest 指针或最新结果文件，保持原路径。", "移动会影响网页状态、交接说明或实验查看。", False)

    for name in ("qfq", "qfq_full"):
        path = parquet_root / name
        if path.exists():
            add_plan_action(actions, "archive", path, quant_data / "archive" / "parquet" / "ashare_1d" / name, "旧测试/旧口径 parquet，不是当前正式入口；用户确认后可归档。", "如果仍有隐藏脚本引用旧路径，归档前需全局引用检查。", True)
    for name in ("cn_data", "cn_data_jqdata", "cn_data_jqdata_full"):
        path = qlib_root / name
        if path.exists():
            add_plan_action(actions, "archive", path, quant_data / "archive" / "qlib" / name, "测试 provider 或短历史 provider，不是当前正式入口；用户确认后可归档。", "可能影响旧实验复现；归档前需核对 MLflow artifact 和 jobs/index.sqlite 引用。", True)
    for name in ("cn_data_baostock_jqdata", "cn_data_baostock_jqdata_cleaned"):
        path = qlib_root / name
        if path.exists():
            add_plan_action(actions, "document", path, None, "V1 兼容 provider 仅供旧消费者使用，不是正式训练入口。", "无路径变更风险；主要风险是误把兼容输出当作 canonical。", False)
    for name in ("baostock_qfq", "jqdata", "merged_baostock_jqdata", "merged_baostock_jqdata_cleaned"):
        path = parquet_root / name
        if path.exists():
            add_plan_action(actions, "document", path, None, "审计来源或 V1 兼容输出保留用于核对，不是正式训练入口。", "无路径变更风险；若后续迁移必须先核对旧消费者。", False)

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
        "catalog": catalog,
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

    bundle_digest = hashlib.sha256(
        _json_bytes(inventory) + b"\0" + _json_bytes(plan)
    ).hexdigest()
    artifact_suffix = f"{ts}_{bundle_digest[:16]}"
    paths = HygieneReportPaths(
        inventory_md=reports / f"data_hygiene_inventory_{artifact_suffix}.md",
        inventory_json=reports / f"data_hygiene_inventory_{artifact_suffix}.json",
        cleanup_plan_md=reports / f"data_hygiene_cleanup_plan_{artifact_suffix}.md",
        cleanup_plan_json=reports / f"data_hygiene_cleanup_plan_{artifact_suffix}.json",
        inventory_latest_md=reports / "data_hygiene_inventory_latest.md",
        inventory_latest_json=reports / "data_hygiene_inventory_latest.json",
        cleanup_plan_latest_md=reports / "data_hygiene_cleanup_plan_latest.md",
        cleanup_plan_latest_json=reports / "data_hygiene_cleanup_plan_latest.json",
        bundle_latest_json=reports / "data_hygiene_reports_latest.json",
    )
    write_json(paths.inventory_json, inventory)
    write_json(paths.cleanup_plan_json, plan)
    write_inventory_markdown(paths.inventory_md, inventory)
    write_cleanup_plan_markdown(paths.cleanup_plan_md, plan)
    write_json(paths.inventory_latest_json, inventory)
    write_json(paths.cleanup_plan_latest_json, plan)
    atomic_write_text(
        paths.inventory_latest_md,
        paths.inventory_md.read_text(encoding="utf-8"),
    )
    atomic_write_text(
        paths.cleanup_plan_latest_md,
        paths.cleanup_plan_md.read_text(encoding="utf-8"),
    )
    write_json(
        paths.bundle_latest_json,
        {
            "schema_version": 1,
            "publication_contract": (
                "atomic_json_pointer_to_immutable_hygiene_bundle_v1"
            ),
            "bundle_id": f"data-hygiene-{bundle_digest[:16]}",
            "generated_at": generated_at,
            "inventory_artifact": paths.inventory_json.name,
            "inventory_sha256": hashlib.sha256(
                paths.inventory_json.read_bytes()
            ).hexdigest(),
            "cleanup_plan_artifact": paths.cleanup_plan_json.name,
            "cleanup_plan_sha256": hashlib.sha256(
                paths.cleanup_plan_json.read_bytes()
            ).hexdigest(),
        },
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate read-only data hygiene inventory and dry-run cleanup plan.")
    parser.add_argument("--projects-root", type=Path, default=None)
    parser.add_argument("--quant-data-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None, help="Report output directory. Defaults to quant_workflow/reports.")
    parser.add_argument("--timestamp", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_data_hygiene_reports(
        projects_root=args.projects_root,
        quant_data_root=args.quant_data_root,
        reports_root=args.out_dir,
        timestamp=args.timestamp,
    )
    print("Generated data hygiene inventory and dry-run cleanup plan:")
    for path in (
        paths.inventory_md,
        paths.inventory_json,
        paths.cleanup_plan_md,
        paths.cleanup_plan_json,
        paths.bundle_latest_json,
    ):
        print(path)


if __name__ == "__main__":
    main()
