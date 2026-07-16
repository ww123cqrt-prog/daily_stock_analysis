#!/usr/bin/env python3
"""Build DSA's compact MLflow translation catalog from official 3.14.0 data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


MLFLOW_VERSION = "3.14.0"
MLFLOW_UPSTREAM_COMMIT = "86cd7f5a1dc25a1387ad07c87811bbc30f62951c"
OFFICIAL_SOURCE_URL = (
    f"https://raw.githubusercontent.com/mlflow/mlflow/{MLFLOW_UPSTREAM_COMMIT}/"
    "mlflow/server/js/src/lang/zh-CN.json"
)
MESSAGE_PATTERN = re.compile(
    r'id:"((?:\\.|[^"\\])*)",defaultMessage:"((?:\\.|[^"\\])*)"'
)
MANUAL_OVERRIDES = {
    "Artifacts": "产物",
    "Experiments": "实验",
    "Getting Started": "开始使用",
    "Home": "首页",
    "Information about UI telemetry": "关于界面使用情况统计",
    "New to MLflow?": "初次使用 MLflow？",
    "Runs": "运行记录",
    "Settings": "设置",
    "Tag": "标签",
    "Welcome to MLflow": "欢迎使用 MLflow",
}


def validate_mlflow_version(actual_version: str) -> None:
    if actual_version != MLFLOW_VERSION:
        raise RuntimeError(
            f"expected MLflow {MLFLOW_VERSION}, found {actual_version}; "
            "update the recorded version and upstream commit before rebuilding"
        )


def _decode_js_string(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except json.JSONDecodeError:
        return value


def build_catalog(zh_cn_path: Path, js_dir: Path) -> dict[str, str]:
    official_payload = json.loads(zh_cn_path.read_text(encoding="utf-8"))
    translated_by_id = {
        message_id: item.get("defaultMessage", "")
        for message_id, item in official_payload.items()
        if isinstance(item, dict) and isinstance(item.get("defaultMessage"), str)
    }

    catalog: dict[str, str] = {}
    for bundle_path in sorted(js_dir.glob("*.js")):
        source = bundle_path.read_text(encoding="utf-8", errors="ignore")
        for raw_id, raw_english in MESSAGE_PATTERN.findall(source):
            message_id = _decode_js_string(raw_id)
            english = _decode_js_string(raw_english)
            chinese = translated_by_id.get(message_id, "").strip()
            if english.strip() and chinese and english != chinese:
                catalog.setdefault(english, chinese)

    catalog.update(MANUAL_OVERRIDES)
    return dict(sorted(catalog.items()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a compact English-to-Chinese catalog for the installed MLflow UI."
    )
    parser.add_argument(
        "--zh-cn",
        type=Path,
        required=True,
        help="Official MLflow 3.14.0 zh-CN.json downloaded from the recorded source URL.",
    )
    parser.add_argument(
        "--js-dir",
        type=Path,
        help="Installed mlflow/server/js/build/static/js directory (auto-detected by default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent.parent
            / "src"
            / "assets"
            / "mlflow_i18n"
            / f"mlflow-{MLFLOW_VERSION}-zh-CN.json"
        ),
    )
    args = parser.parse_args()

    import mlflow

    validate_mlflow_version(mlflow.__version__)
    if args.js_dir is None:
        args.js_dir = Path(mlflow.__file__).resolve().parent / "server/js/build/static/js"

    if not args.zh_cn.is_file():
        parser.error(f"official translation file not found: {args.zh_cn}")
    if not args.js_dir.is_dir():
        parser.error(f"installed MLflow JS directory not found: {args.js_dir}")

    catalog = build_catalog(args.zh_cn, args.js_dir)
    payload = {
        "metadata": {
            "mlflowVersion": MLFLOW_VERSION,
            "upstreamCommit": MLFLOW_UPSTREAM_COMMIT,
            "source": OFFICIAL_SOURCE_URL,
            "translationSourceSha256": hashlib.sha256(args.zh_cn.read_bytes()).hexdigest(),
            "generator": "scripts/build_mlflow_i18n_catalog.py",
            "translationCount": len(catalog),
        },
        "translations": catalog,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(catalog)} translations to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
