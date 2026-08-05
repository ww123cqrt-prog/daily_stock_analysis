#!/usr/bin/env python3
"""Compare isolated real LightGBM runs launched from the web and script paths."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import reports_root  # noqa: E402


EQUIVALENCE_EXPERIMENT_NAME = "dsa_training_equivalence"
MAX_RESULT_BYTES = 5_000_000
EQUIVALENCE_EXPECTED_PERIODS = {
    "training": ["2024-01-02", "2024-06-28"],
    "validation": ["2024-07-01", "2024-09-30"],
    "test": ["2024-10-08", "2024-12-31"],
}
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}")
PROVIDER_IDENTITY_FIELDS = (
    "provider_contract",
    "dataset_id",
    "snapshot_id",
    "manifest_sha256",
    "content_sha256",
)
REPLAY_HASH_FIELDS = (
    "source_config_sha256",
    "runtime_config_sha256",
    "config_hash",
)


def _check(check_id: str, label: str, passed: bool, detail: str) -> dict[str, Any]:
    return {
        "id": check_id,
        "label": label,
        "passed": bool(passed),
        "detail": detail,
    }


def _same_metrics(left: Any, right: Any) -> tuple[bool, str]:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False, "metrics must be JSON objects"
    if not left or not right:
        return False, "metrics must be non-empty JSON objects"
    left_keys = set(left)
    right_keys = set(right)
    if left_keys != right_keys:
        missing = sorted(left_keys ^ right_keys)
        return False, f"metric keys differ: {', '.join(missing[:10])}"
    mismatches: list[str] = []
    for key in sorted(left_keys):
        left_raw = left[key]
        right_raw = right[key]
        if (
            isinstance(left_raw, bool)
            or isinstance(right_raw, bool)
            or not isinstance(left_raw, (int, float))
            or not isinstance(right_raw, (int, float))
        ):
            mismatches.append(str(key))
            continue
        left_value = float(left_raw)
        right_value = float(right_raw)
        if not (
            math.isfinite(left_value)
            and math.isfinite(right_value)
            and math.isclose(left_value, right_value, rel_tol=1e-12, abs_tol=1e-12)
        ):
            mismatches.append(str(key))
    if mismatches:
        return False, f"metric values differ: {', '.join(mismatches[:10])}"
    return True, f"{len(left_keys)} metrics match within 1e-12 tolerance"


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _valid_provider_identity(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if any(not isinstance(value.get(key), str) or not value.get(key) for key in PROVIDER_IDENTITY_FIELDS):
        return False
    return _is_sha256(value.get("manifest_sha256")) and _is_sha256(
        value.get("content_sha256")
    )


def _valid_replay_contract(value: Any) -> bool:
    return isinstance(value, Mapping) and all(
        _is_sha256(value.get(key)) for key in REPLAY_HASH_FIELDS
    )


def compare_training_equivalence_results(
    web: Mapping[str, Any],
    script: Mapping[str, Any],
) -> dict[str, Any]:
    web_run_id = str(web.get("run_id") or "")
    script_run_id = str(script.get("run_id") or "")
    scope_ok = all(
        (
            item.get("status") == "completed"
            and item.get("result_scope") == "training_equivalence"
            and item.get("test_only") is True
        )
        for item in (web, script)
    ) and web.get("equivalence_channel") == "web" and script.get(
        "equivalence_channel"
    ) == "script"
    ownership_ok = (
        bool(web_run_id)
        and bool(script_run_id)
        and web_run_id != script_run_id
        and web.get("experiment_name") == EQUIVALENCE_EXPERIMENT_NAME
        and script.get("experiment_name") == EQUIVALENCE_EXPERIMENT_NAME
        and bool(web.get("experiment_id"))
        and web.get("experiment_id") == script.get("experiment_id")
    )

    web_identity = web.get("provider_identity")
    script_identity = script.get("provider_identity")
    identity_ok = (
        _valid_provider_identity(web_identity)
        and _valid_provider_identity(script_identity)
        and dict(web_identity) == dict(script_identity)
    )

    web_contract = web.get("replay_contract")
    script_contract = script.get("replay_contract")
    contract_keys = (
        "source_config_sha256",
        "runtime_config_sha256",
        "config_hash",
        "periods",
    )
    config_ok = (
        _valid_replay_contract(web_contract)
        and _valid_replay_contract(script_contract)
        and all(web_contract.get(key) == script_contract.get(key) for key in contract_keys)
    )
    bounded_period_ok = (
        isinstance(web_contract, Mapping)
        and isinstance(script_contract, Mapping)
        and web_contract.get("periods") == EQUIVALENCE_EXPECTED_PERIODS
        and script_contract.get("periods") == EQUIVALENCE_EXPECTED_PERIODS
    )

    web_hashes = web.get("artifact_sha256")
    script_hashes = script.get("artifact_sha256")
    prediction_ok = (
        isinstance(web_hashes, Mapping)
        and isinstance(script_hashes, Mapping)
        and _is_sha256(web_hashes.get("pred.pkl"))
        and _is_sha256(script_hashes.get("pred.pkl"))
        and web_hashes.get("pred.pkl") == script_hashes.get("pred.pkl")
    )
    label_ok = (
        isinstance(web_hashes, Mapping)
        and isinstance(script_hashes, Mapping)
        and _is_sha256(web_hashes.get("label.pkl"))
        and _is_sha256(script_hashes.get("label.pkl"))
        and web_hashes.get("label.pkl") == script_hashes.get("label.pkl")
    )
    metrics_ok, metrics_detail = _same_metrics(web.get("metrics"), script.get("metrics"))

    checks = [
        _check(
            "isolated_test_contract",
            "隔离测试契约",
            scope_ok,
            "both runs are completed, test-only, and have the expected channels",
        ),
        _check(
            "mlflow_ownership",
            "MLflow 归属",
            ownership_ok,
            "dedicated experiment, same experiment ID, and distinct Run IDs",
        ),
        _check(
            "provider_identity",
            "数据快照",
            identity_ok,
            "provider identities match exactly",
        ),
        _check(
            "config_hashes",
            "配置与时间范围",
            config_ok,
            "source/runtime/replay hashes and periods match",
        ),
        _check(
            "bounded_period",
            "受控训练周期",
            bounded_period_ok,
            "training, validation, and test periods match the registered bounded contract",
        ),
        _check(
            "prediction_hash",
            "预测哈希",
            prediction_ok,
            "pred.pkl SHA-256 values match",
        ),
        _check(
            "label_hash",
            "标签哈希",
            label_ok,
            "label.pkl SHA-256 values match",
        ),
        _check(
            "metrics_and_backtest",
            "指标与组合回测",
            metrics_ok,
            metrics_detail,
        ),
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "pass" if all(item["passed"] for item in checks) else "fail",
        "test_only": True,
        "warning": "测试数据，不可用于研究或交易。",
        "web_run_id": web_run_id,
        "script_run_id": script_run_id,
        "experiment_name": EQUIVALENCE_EXPERIMENT_NAME,
        "provider_snapshot_id": (
            str(web_identity.get("snapshot_id") or "")
            if isinstance(web_identity, Mapping)
            else ""
        ),
        "checks": checks,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _markdown(report: Mapping[str, Any]) -> str:
    rows = [
        "# LightGBM 网页/脚本真实训练等价报告",
        "",
        "> **测试数据，不可用于研究或交易。** 本报告只验证两条运行路径是否一致。",
        "",
        f"- 结果：`{report.get('status')}`",
        f"- 网页 Run：`{report.get('web_run_id')}`",
        f"- 脚本 Run：`{report.get('script_run_id')}`",
        f"- 数据快照：`{report.get('provider_snapshot_id')}`",
        "",
        "| 检查 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for item in report.get("checks") or []:
        rows.append(
            f"| {item.get('label')} | {'通过' if item.get('passed') else '失败'} | {item.get('detail')} |"
        )
    return "\n".join(rows) + "\n"


def write_comparison_reports(
    report: Mapping[str, Any],
    *,
    reports_root: Path,
    timestamp: str,
) -> dict[str, Path]:
    base = Path(reports_root) / "test_runs" / "training_equivalence"
    comparison_dir = base / "comparisons"
    stem = f"comparison_{timestamp}"
    json_path = comparison_dir / f"{stem}.json"
    markdown_path = comparison_dir / f"{stem}.md"
    latest_path = base / "comparison_latest.json"
    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_write(json_path, json_text)
    _atomic_write(markdown_path, _markdown(report))
    _atomic_write(latest_path, json_text)
    return {"json": json_path, "markdown": markdown_path, "latest": latest_path}


def _load_result(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    if size > MAX_RESULT_BYTES:
        raise ValueError(f"result exceeds {MAX_RESULT_BYTES} bytes")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("result must be a JSON object")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-result", type=Path, required=True)
    parser.add_argument("--script-result", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, default=reports_root())
    args = parser.parse_args()
    try:
        report = compare_training_equivalence_results(
            _load_result(args.web_result),
            _load_result(args.script_result),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = write_comparison_reports(
        report,
        reports_root=args.reports_root,
        timestamp=timestamp,
    )
    print(json.dumps({key: str(value) for key, value in paths.items()}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
