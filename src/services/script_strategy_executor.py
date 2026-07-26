# -*- coding: utf-8 -*-
"""Launch a checked strategy script in a constrained local child process."""

from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Sequence

from src.core.script_strategy import ScriptExecutionError, validate_strategy_script
from src.core.strategy_backtest import PriceBar, TargetAllocation


def _profile_literal(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def _sandbox_profile(temp_dir: Path, repo_root: Path) -> str:
    return f'''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
    (subpath {_profile_literal(temp_dir)})
    (literal "/dev/null"))
'''


def run_script_in_sandbox(
    source: str,
    *,
    bars: Sequence[PriceBar],
    symbols: Sequence[str],
    start_date: date | None,
    end_date: date | None,
    timeout_seconds: int = 8,
) -> tuple[list[TargetAllocation], list[str], dict[str, Any]]:
    check = validate_strategy_script(source)
    repo_root = Path(__file__).resolve().parents[2]
    payload = {
        "script": source,
        "symbols": list(symbols),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "bars": [
            {
                "symbol": bar.symbol,
                "date": bar.date.isoformat(),
                "open": bar.open,
                "close": bar.close,
            }
            for bar in bars
        ],
    }
    with tempfile.TemporaryDirectory(prefix="dsa-strategy-") as temp_name:
        temp_dir = Path(temp_name).resolve()
        input_path = temp_dir / "input.json"
        output_path = temp_dir / "output.json"
        profile_path = temp_dir / "strategy.sb"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        python_executable = str(Path(sys.executable).resolve())
        command = [
            python_executable,
            "-m",
            "src.core.script_strategy_runner",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        sandbox_exec = shutil.which("sandbox-exec")
        if platform.system() == "Darwin":
            if not sandbox_exec:
                raise ScriptExecutionError("macOS sandbox-exec is unavailable")
            profile_path.write_text(_sandbox_profile(temp_dir, repo_root), encoding="utf-8")
            command = [sandbox_exec, "-f", str(profile_path), *command]

        env = {
            "HOME": str(temp_dir),
            "LANG": "C.UTF-8",
            "PATH": f"{Path(python_executable).parent}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(repo_root),
            "TMPDIR": str(temp_dir),
        }
        try:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ScriptExecutionError(f"script exceeded the {timeout_seconds}-second limit") from exc
        if not output_path.is_file():
            message = (completed.stderr or completed.stdout or "sandbox produced no result")[-1000:]
            raise ScriptExecutionError(
                f"sandbox failed with exit code {completed.returncode}: {message}"
            )
        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ScriptExecutionError("sandbox returned an unreadable result") from exc
        if completed.returncode != 0 or result.get("status") != "completed":
            raise ScriptExecutionError(str(result.get("message") or "script execution failed"))

    allocations = [
        TargetAllocation(
            source_id=str(item["source_id"]),
            signal_date=date.fromisoformat(str(item["signal_date"])),
            target_weights={str(key): float(value) for key, value in item["target_weights"].items()},
            reason=str(item.get("reason") or ""),
        )
        for item in result.get("allocations", [])
    ]
    return allocations, [str(item) for item in result.get("logs", [])], result.get("check") or check
