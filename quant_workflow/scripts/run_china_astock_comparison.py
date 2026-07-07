from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from shutil import copyfile
from typing import Any, NamedTuple


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import china_astock_root, projects_root, reports_root, workflow_root  # noqa: E402


PROJECTS_ROOT = projects_root()
QUANT_WORKFLOW_ROOT = workflow_root()
REPORTS_ROOT = reports_root()
CHINA_ASTOCK_ROOT = china_astock_root()
CHINA_ASTOCK_PYTHON = Path(os.environ.get("CHINA_ASTOCK_PYTHON", sys.executable)).expanduser()
DEFAULT_BUILD_TIMEOUT_SECONDS = 300.0


class ComparisonPaths(NamedTuple):
    project_root: Path
    full_search: Path
    evaluation: Path
    out_csv: Path
    out_md: Path
    latest_csv: Path
    latest_md: Path
    latest_json: Path


def _resolve_under_project(project_root: Path, value: str | Path | None, default_relative: str) -> Path:
    path = Path(value) if value else project_root / default_relative
    if not path.is_absolute():
        path = project_root / path
    return path.expanduser().resolve()


def _validate_project_root(project_root: Path) -> Path:
    root = project_root.expanduser().resolve()
    expected_script = root / "scripts" / "build_comparison.py"
    if not expected_script.is_file():
        raise ValueError(f"china-astock-quant build script not found: {expected_script}")
    return root


def build_paths(
    project_root: str | Path | None,
    full_search: str | Path | None,
    evaluation: str | Path | None,
    output_dir: str | Path | None,
    *,
    timestamp: str | None = None,
) -> ComparisonPaths:
    root = _validate_project_root(Path(project_root) if project_root else CHINA_ASTOCK_ROOT)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(output_dir).expanduser().resolve() if output_dir else REPORTS_ROOT.resolve()
    return ComparisonPaths(
        project_root=root,
        full_search=_resolve_under_project(root, full_search, "reports/full_search.json"),
        evaluation=_resolve_under_project(root, evaluation, "reports/evaluate_winners.csv"),
        out_csv=reports_dir / f"china_astock_strategy_comparison_{stamp}.csv",
        out_md=reports_dir / f"china_astock_strategy_comparison_{stamp}.md",
        latest_csv=reports_dir / "china_astock_strategy_comparison_latest.csv",
        latest_md=reports_dir / "china_astock_strategy_comparison_latest.md",
        latest_json=reports_dir / "china_astock_strategy_comparison_latest.json",
    )


def _python_bin(project_root: Path) -> str:
    if project_root == CHINA_ASTOCK_ROOT.expanduser().resolve() and CHINA_ASTOCK_PYTHON.exists():
        return str(CHINA_ASTOCK_PYTHON)
    return str(Path(sys.executable))


def _validate_input_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is required but not found: {path}")


def _timeout_seconds(value: float | int | str | None = None) -> float:
    raw = os.environ.get("CHINA_ASTOCK_COMPARISON_TIMEOUT_SECONDS") if value is None else value
    if raw in (None, ""):
        return DEFAULT_BUILD_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timeout must be a positive number of seconds.") from exc
    if timeout <= 0:
        raise ValueError("Timeout must be a positive number of seconds.")
    return timeout


def _print_completed_output(stdout: str | bytes | None, stderr: str | bytes | None) -> None:
    if stdout:
        print(stdout.decode("utf-8", errors="replace") if isinstance(stdout, bytes) else stdout, file=sys.stdout, end="")
    if stderr:
        print(stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr, file=sys.stderr, end="")


def run_report(
    project_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    full_search: str | Path | None = None,
    evaluation: str | Path | None = None,
    timestamp: str | None = None,
    timeout_seconds: float | int | str | None = None,
) -> dict[str, Any]:
    paths = build_paths(project_root, full_search, evaluation, output_dir, timestamp=timestamp)
    _validate_input_file(paths.full_search, "full_search.json")
    _validate_input_file(paths.evaluation, "evaluate_winners.csv")
    paths.out_csv.parent.mkdir(parents=True, exist_ok=True)

    command = [
        _python_bin(paths.project_root),
        str(paths.project_root / "scripts" / "build_comparison.py"),
        "--full-search",
        str(paths.full_search),
        "--evaluation",
        str(paths.evaluation),
        "--out-csv",
        str(paths.out_csv),
        "--out-md",
        str(paths.out_md),
    ]
    timeout = _timeout_seconds(timeout_seconds)
    try:
        completed = subprocess.run(
            command,
            cwd=paths.project_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        _print_completed_output(exc.stdout, exc.stderr)
        raise TimeoutError(f"build_comparison.py timed out after {timeout:g} seconds") from exc
    if completed.returncode != 0:
        _print_completed_output(completed.stdout, completed.stderr)
        raise RuntimeError(f"build_comparison.py failed with return code {completed.returncode}")

    copyfile(paths.out_csv, paths.latest_csv)
    copyfile(paths.out_md, paths.latest_md)
    status = {
        "status": "completed",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(paths.project_root),
        "full_search": str(paths.full_search),
        "evaluation": str(paths.evaluation),
        "output_csv": str(paths.out_csv),
        "output_md": str(paths.out_md),
        "latest_csv": str(paths.latest_csv),
        "latest_md": str(paths.latest_md),
        "command": command,
    }
    paths.latest_json.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True))
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build china-astock-quant strategy comparison artifacts into quant_workflow reports.")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--full-search", default=None)
    parser.add_argument("--evaluation", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Seconds to wait for build_comparison.py; default 300.")
    args = parser.parse_args(argv)
    run_report(
        args.project_root,
        args.output_dir,
        full_search=args.full_search,
        evaluation=args.evaluation,
        timeout_seconds=args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
