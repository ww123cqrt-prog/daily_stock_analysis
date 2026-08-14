# -*- coding: utf-8 -*-
"""Child-process entrypoint for bounded DSA strategy script execution.

Input JSON layout::

    {
      "script": "...",
      "bundle_dir": "/abs/path/to/universe/bundle",
      "start_date": "YYYY-MM-DD" | null,
      "end_date": "YYYY-MM-DD" | null
    }

The bundle directory is written by the DSA backend before the sandbox starts
and contains ``market.npz`` + ``universe.json`` (see ``src.core.script_market``).
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import resource
import sys
from typing import Any

from src.core.script_market import MarketView
from src.core.script_strategy import execute_strategy_script


def _apply_limits() -> None:
    # CPU budget guards the strategy script logic; data loading happens before
    # this process starts, so a generous but still bounded budget is enough to
    # stop runaway loops without limiting legitimate research scripts.
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    # Address-space cap: even if the AST whitelist and sandbox profile are
    # bypassed, runaway allocations (huge lists, strings) hit this ceiling
    # instead of exhausting the host. 4 GiB is far above any legitimate
    # script but far below a host takeover.
    try:
        resource.setrlimit(
            resource.RLIMIT_AS, (4 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024)
        )
    except (ValueError, OSError):
        # Platforms where RLIMIT_AS is unavailable (older macOS kernels).
        pass


class _LimitedStdout:
    """sys.stdout replacement that caps total output at 256 KiB.

    A hostile script could otherwise flood stdout (captured into parent
    memory) with unbounded text.
    """

    limit = 256 * 1024

    def __init__(self) -> None:
        self._buffer = ""
        self._truncated = False

    def write(self, text: str) -> int:
        remaining = self.limit - len(self._buffer)
        if remaining <= 0:
            self._truncated = True
            return len(text)
        chunk = text[:remaining]
        self._buffer += chunk
        if len(chunk) < len(text):
            self._truncated = True
        return len(text)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _serialize_result(payload: dict[str, Any]) -> dict[str, Any]:
    bundle_dir = Path(str(payload.get("bundle_dir") or ""))
    if not bundle_dir.is_dir():
        raise ValueError("bundle_dir is missing or not a directory")
    view = MarketView.load(bundle_dir)
    start_date = date.fromisoformat(payload["start_date"]) if payload.get("start_date") else None
    end_date = date.fromisoformat(payload["end_date"]) if payload.get("end_date") else None
    result = execute_strategy_script(
        str(payload.get("script") or ""),
        view=view,
        start_date=start_date,
        end_date=end_date,
    )
    return {
        "status": "completed",
        "allocations": [
            {
                "source_id": item.source_id,
                "signal_date": item.signal_date.isoformat(),
                "target_weights": dict(item.target_weights),
                "reason": item.reason,
            }
            for item in result.allocations
        ],
        "logs": list(result.logs),
        "check": result.check,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _apply_limits()
    # Cap stdout volume: subprocess.run captures stdout into parent memory,
    # so a runaway print loop must not be able to inflate the host.
    sys.stdout = _LimitedStdout()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = _serialize_result(payload)
    except Exception as exc:
        result = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
