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

from src.core.script_market import MarketBundle, MarketView
from src.core.script_strategy import (
    ScriptExecutionError,
    validate_strategy_script,
    view_from_bars,
)
from src.core.strategy_backtest import PriceBar, TargetAllocation


def _profile_literal(path: Path) -> str:
    return json.dumps(str(path.resolve()))


def _sandbox_profile(
    temp_dir: Path,
    repo_root: Path,
    bundle_dir: Path,
    python_executable: Path,
) -> str:
    """sandbox-exec profile: no network, no writes, and no reads outside
the repo/venv/interpreter trees.

    A blanket ``(deny file-read*)`` breaks dyld/exec on macOS, so the profile
    denies the user home subtree (secrets, keychains, ``.ssh``/``.aws``,
    browser data, iCloud) and re-allows the specific trees the runner needs.
    Everything under the home directory outside those trees is unreadable,
    which closes the AST-whitelist-escape exfiltration path through
    ``context.log``.
    """
    venv_root = Path(sys.executable).parent.parent
    base_prefix = Path(sys.base_prefix).resolve()
    home = Path.home()
    # Read access inside the repo is limited to ``src/`` plus a literal read
    # of the repo root directory itself (the ``-m`` package finder needs to
    # list it to locate the ``src`` package). Runtime files like ``.env`` at
    # the repo root stay unreadable even if the AST whitelist is bypassed.
    read_entries = [
        f"    (subpath {_profile_literal(temp_dir)})",
        f"    (subpath {_profile_literal(bundle_dir)})",
        f"    (subpath {_profile_literal(repo_root / 'src')})",
        f"    (subpath {_profile_literal(venv_root)})",
        f"    (subpath {_profile_literal(base_prefix)})",
        f"    (subpath {_profile_literal(python_executable.parent)})",
        f"    (literal {_profile_literal(repo_root)})",
    ]
    allow_read = "\n".join(read_entries)
    return f'''(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write*
    (subpath {_profile_literal(temp_dir)})
    (literal "/dev/null"))
(deny file-read* (subpath {_profile_literal(home)}))
(allow file-read*
{allow_read})
'''


def _run_strategy_process(
    source: str,
    *,
    bundle_dir: Path | None = None,
    view: MarketView | None = None,
    bars: Sequence[PriceBar] | None = None,
    symbols: Sequence[str] | None = None,
    start_date: date | None,
    end_date: date | None,
    timeout_seconds: int = 30,
    execution_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a strategy script against a market bundle in the sandbox.

    ``bundle_dir`` (preferred) points at a ``MarketBundle`` directory written
    by the backend. Legacy callers may pass ``bars``/``symbols`` instead; the
    backend assembles a small bundle from them.
    """
    check = validate_strategy_script(source)
    repo_root = Path(__file__).resolve().parents[2]

    with tempfile.TemporaryDirectory(prefix="dsa-strategy-") as temp_name:
        temp_dir = Path(temp_name).resolve()
        if bundle_dir is None:
            if bars is None or symbols is None:
                raise ScriptExecutionError("either bundle_dir or bars+symbols are required")
            legacy_view = view_from_bars(bars, symbols)
            bundle_dir = temp_dir / "market"
            MarketBundle(
                dates=legacy_view.dates,
                symbols=legacy_view.symbols,
                open=legacy_view.open,
                close=legacy_view.close,
                members=legacy_view.members,
                meta={},
            ).save(bundle_dir)

        payload = {
            "script": source,
            "bundle_dir": str(bundle_dir),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
        }
        if execution_config is not None:
            payload["execution_config"] = execution_config
        input_path = temp_dir / "input.json"
        output_path = temp_dir / "output.json"
        profile_path = temp_dir / "strategy.sb"
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        # Resolve the interpreter to its real path: sandboxed execvp rejects
        # symlink chains under a denied subtree, and uv-managed venvs symlink
        # the interpreter into ``~/.local/share/uv``.
        python_executable = str(Path(sys.executable).resolve())
        base_prefix = str(Path(sys.base_prefix).resolve())
        # Explicitly expose the venv site-packages inside the sandbox: the
        # sandboxed interpreter can lose the venv prefix (symlink/pyvenv.cfg
        # resolution differs under sandbox-exec), which otherwise drops
        # numpy/pyarrow from sys.path.
        venv_root = Path(sys.executable).parent.parent
        venv_site = (
            venv_root
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        pythonpath = os.pathsep.join(
            str(entry)
            for entry in (repo_root, venv_site)
            if Path(entry).exists()
        )
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
        if platform.system() != "Darwin":
            # The AST whitelist alone is not a security boundary; never run
            # user scripts without OS-level isolation. On non-macOS hosts the
            # platform sandbox does not exist, so execution is refused rather
            # than silently degrading to a bare subprocess.
            raise ScriptExecutionError(
                "script strategy execution requires the macOS sandbox "
                "(sandbox-exec); this platform is not supported"
            )
        if not sandbox_exec:
            raise ScriptExecutionError("macOS sandbox-exec is unavailable")
        profile_path.write_text(
            _sandbox_profile(temp_dir, repo_root, bundle_dir, Path(python_executable)),
            encoding="utf-8",
        )
        command = [sandbox_exec, "-f", str(profile_path), *command]

        env = {
            "HOME": str(temp_dir),
            "LANG": "C.UTF-8",
            "PATH": f"{Path(python_executable).parent}:/usr/bin:/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            # uv-managed interpreters build with a placeholder prefix
            # ("/install"); PYTHONHOME points the sandboxed child at the real
            # stdlib location since the venv pyvenv.cfg resolution differs
            # under sandbox-exec.
            "PYTHONHOME": base_prefix,
            "PYTHONPATH": pythonpath,
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
            raise ScriptExecutionError(
                f"script exceeded the {timeout_seconds}-second limit"
            ) from exc
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

    return result


def _allocations(result):
    allocations = [
        TargetAllocation(
            source_id=str(item["source_id"]),
            signal_date=date.fromisoformat(str(item["signal_date"])),
            target_weights={str(key): float(value) for key, value in item["target_weights"].items()},
            reason=str(item.get("reason") or ""),
        )
        for item in result.get("allocations", [])
    ]
    return allocations


def run_script_in_sandbox(
    source: str, *, bundle_dir: Path | None = None, view: MarketView | None = None,
    bars: Sequence[PriceBar] | None = None, symbols: Sequence[str] | None = None,
    start_date: date | None, end_date: date | None, timeout_seconds: int = 30,
) -> tuple[list[TargetAllocation], list[str], dict[str, Any]]:
    """Legacy signal-only scripts retain their existing sandbox contract."""
    result = _run_strategy_process(source, bundle_dir=bundle_dir, view=view, bars=bars, symbols=symbols,
        start_date=start_date, end_date=end_date, timeout_seconds=timeout_seconds)
    return _allocations(result), result.get("logs", []), result["check"]


def run_script_backtest_in_sandbox(
    source: str, *, bundle_dir: Path, start_date: date | None, end_date: date | None,
    config: Any, timeout_seconds: int = 30,
) -> tuple[dict[str, Any], list[TargetAllocation], list[str], dict[str, Any]]:
    """Run fill-aware signals and the existing matcher in the same OS sandbox."""
    from dataclasses import asdict
    result = _run_strategy_process(source, bundle_dir=bundle_dir, start_date=start_date,
        end_date=end_date, timeout_seconds=timeout_seconds, execution_config=asdict(config))
    return result["backtest_report"], _allocations(result), result.get("logs", []), result["check"]
