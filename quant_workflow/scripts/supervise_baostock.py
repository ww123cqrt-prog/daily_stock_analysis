"""Supervise Baostock long-history downloads with restartable small batches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import projects_root, quant_data_root, workflow_root  # noqa: E402


ROOT = projects_root()
QLIB = ROOT / "qlib"
PYTHON = Path(os.environ.get("QUANT_WORKFLOW_PYTHON") or sys.executable).expanduser()
SCRIPT = workflow_root() / "scripts" / "download_baostock.py"
DATA = quant_data_root()
OUT = DATA / "parquet" / "ashare_1d" / "baostock_qfq"
LOG = DATA / "baostock_supervisor.log"
PID_FILE = DATA / "baostock_supervisor.pid"
CURRENT_FILE = DATA / "baostock_current.json"
EMPTY_CODES_FILE = DATA / "baostock_empty_codes.json"
TIMEOUT_CODES_FILE = DATA / "baostock_timeout_codes.json"
STOCK_LIST = DATA / "stock_lists.json"


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp()}] {message}\n")
        handle.flush()


def load_json_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    return set(data if isinstance(data, list) else [])


def save_json_list(path: Path, values: set[str]) -> None:
    path.write_text(json.dumps(sorted(values), ensure_ascii=False, indent=2), encoding="utf-8")


def count_files() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    return sum(1 for _ in OUT.glob("*.parquet"))


def total_codes() -> int:
    data = json.loads(STOCK_LIST.read_text(encoding="utf-8"))
    return len(data["all"])


def remaining_estimate() -> tuple[int, int, int, int]:
    total = total_codes()
    done = count_files()
    empty = len(load_json_list(EMPTY_CODES_FILE))
    timeout = len(load_json_list(TIMEOUT_CODES_FILE))
    remaining = max(total - done - empty - timeout, 0)
    return total, done, empty + timeout, remaining


def mark_current_timeout() -> None:
    if not CURRENT_FILE.exists():
        return
    try:
        current = json.loads(CURRENT_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    symbol = current.get("symbol")
    if not symbol:
        return
    codes = load_json_list(TIMEOUT_CODES_FILE)
    codes.add(symbol)
    save_json_list(TIMEOUT_CODES_FILE, codes)
    log(f"marked timeout symbol={symbol} current={current}")


def terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_batch(args: argparse.Namespace) -> int:
    cmd = [
        str(PYTHON),
        "-u",
        str(SCRIPT),
        "--start",
        args.start,
        "--end",
        args.end,
        "--sleep",
        str(args.sleep),
        "--retry-failed",
        "--limit",
        str(args.batch_size),
        "--timeout",
        str(args.symbol_timeout),
        "--max-consecutive-failures",
        "50",
    ]
    log(f"starting batch: {' '.join(cmd)}")
    with LOG.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(QLIB),
            stdout=handle,
            stderr=handle,
            text=True,
            start_new_session=True,
        )

    last_count = count_files()
    last_progress_at = time.time()
    while proc.poll() is None:
        time.sleep(args.check_interval)
        current_count = count_files()
        if current_count > last_count:
            log(f"progress: {current_count} files (+{current_count - last_count})")
            last_count = current_count
            last_progress_at = time.time()
            continue

        idle_for = time.time() - last_progress_at
        if idle_for >= args.idle_timeout:
            log(f"idle timeout: no new files for {idle_for:.0f}s; terminating batch")
            mark_current_timeout()
            terminate(proc)
            return 124

    rc = proc.returncode
    log(f"batch exited rc={rc}")
    return int(rc or 0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="20200101")
    parser.add_argument("--end", default="20250611")
    parser.add_argument("--sleep", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--symbol-timeout", type=int, default=30)
    parser.add_argument("--idle-timeout", type=int, default=720)
    parser.add_argument("--check-interval", type=int, default=30)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(143)))
    log("supervisor started")

    batches = 0
    while True:
        total, done, skipped, remaining = remaining_estimate()
        log(f"status: done={done} skipped={skipped} remaining={remaining} total={total}")
        if remaining <= 0:
            log("complete")
            return 0
        if args.max_batches and batches >= args.max_batches:
            log(f"max batches reached: {batches}")
            return 0

        run_batch(args)
        batches += 1
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
