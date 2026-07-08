#!/usr/bin/env python3
"""
Thin wrapper for Qlib Alpha158 + LightGBM baseline training + backtest.

Usage:
    python scripts/train_baseline_lgb.py                     # default config
    python scripts/train_baseline_lgb.py --config other.yaml # custom config
    python scripts/train_baseline_lgb.py --train-only        # skip backtest
"""
import argparse
import json
import os
import re
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - fcntl is unavailable on Windows.
    fcntl = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import china_astock_root, mlruns_root, qlib_provider_uri, quant_data_root, reports_root, workflow_root  # noqa: E402
from _symbols import to_qlib_name as _to_qlib_name  # noqa: E402


WORKFLOW_ROOT = workflow_root()
CONFIGS_DIR = WORKFLOW_ROOT / "configs"
REPORT_DIR = reports_root()
RUNTIME_CONFIG_DIR = REPORT_DIR / "runtime_configs"
LATEST_TRAINING_RESULT = REPORT_DIR / "latest_training_result.json"
EXPERIMENT_LOG = REPORT_DIR / "experiment_log.md"
EXPERIMENT_RUNS = REPORT_DIR / "experiment_runs.jsonl"
MLRUNS_DIR = mlruns_root()
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"
QUANT_DATA = quant_data_root()
CANONICAL_DATA_DIR = qlib_provider_uri()
CLEANED_DATA_DIR = qlib_provider_uri("cleaned")
PROVIDER_URIS = {
    "canonical": CANONICAL_DATA_DIR,
    "cleaned": CLEANED_DATA_DIR,
}
ASTOCK_PROJECT = china_astock_root()

# ── Environment fixes (must run before qlib import) ──────────────────
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)


@contextmanager
def experiment_file_lock(report_dir: Path = REPORT_DIR) -> Iterator[None]:
    """Guard experiment notebook writes shared with the DSA web API."""
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / ".experiment.lock"
    with lock_path.open("a", encoding="utf-8") as lock_handle:
        if fcntl is not None:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def resolve_provider_uri(provider: str | None) -> Path:
    name = (provider or "canonical").strip().lower()
    if name not in PROVIDER_URIS:
        choices = ", ".join(sorted(PROVIDER_URIS))
        raise ValueError(f"Provider must be one of: {choices}.")
    return PROVIDER_URIS[name]


def to_qlib_name(sym: str) -> str:
    return _to_qlib_name(sym, default_exchange="SZ")


def load_provider_instrument_ranges(provider_uri: Path) -> dict[str, tuple[str, str]]:
    all_path = provider_uri / "instruments" / "all.txt"
    if not all_path.exists():
        raise FileNotFoundError(f"Missing provider instruments file: {all_path}")
    ranges: dict[str, tuple[str, str]] = {}
    for line in all_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        ranges[parts[0].upper()] = (parts[1], parts[2])
    return ranges


def build_csi300_lines(symbols: Iterable[str], ranges: dict[str, tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for sym in symbols:
        qlib_name = to_qlib_name(sym)
        if qlib_name in seen or qlib_name not in ranges:
            continue
        seen.add(qlib_name)
        start, end = ranges[qlib_name]
        lines.append(f"{qlib_name}\t{start}\t{end}")
    return lines


def load_csi300_symbols() -> list[str]:
    if not ASTOCK_PROJECT.exists():
        raise FileNotFoundError(f"Missing project for CSI300 universe: {ASTOCK_PROJECT}")
    sys.path.insert(0, str(ASTOCK_PROJECT))
    from data_fetcher.universe import csi300

    return list(csi300())


def ensure_csi300_instruments(provider_uri: Path, csi300_symbols: Iterable[str] | None = None):
    """Generate instruments/csi300.txt if missing."""
    instruments_path = provider_uri / "instruments" / "csi300.txt"
    if instruments_path.exists():
        return

    print(f"[setup] Generating CSI300 instruments file for {provider_uri} ...")
    symbols = list(csi300_symbols) if csi300_symbols is not None else load_csi300_symbols()
    lines = build_csi300_lines(symbols, load_provider_instrument_ranges(provider_uri))
    if not lines:
        raise ValueError(f"No CSI300 symbols matched provider instruments: {provider_uri}")

    instruments_path.parent.mkdir(parents=True, exist_ok=True)
    instruments_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[setup] Written {len(lines)} symbols to {instruments_path}")


def write_runtime_config_with_provider(
    config_path: Path,
    provider_uri: Path,
    provider_name: str,
    out_dir: Path = RUNTIME_CONFIG_DIR,
) -> Path:
    import yaml

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    qlib_init = dict(payload.get("qlib_init") or {})
    qlib_init["provider_uri"] = str(provider_uri)
    payload["qlib_init"] = qlib_init

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runtime_path = out_dir / f"{config_path.stem}_{provider_name}_{timestamp}.yaml"
    runtime_path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    latest_path = out_dir / f"{config_path.stem}_{provider_name}_latest.yaml"
    latest_path.write_text(runtime_path.read_text(encoding="utf-8"), encoding="utf-8")
    return runtime_path


def run_qrun(config_path: str, experiment_name: str = "baseline_lgb_alpha158"):
    """Run qrun programmatically (avoids subprocess)."""
    from qlib.cli.run import workflow

    workflow(config_path, experiment_name=experiment_name, uri_folder=str(MLRUNS_DIR))


def load_latest_metrics(experiment_name: str = "baseline_lgb_alpha158") -> dict:
    """Load metrics from the latest MLflow run."""
    import mlflow
    import yaml

    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        experiments = client.search_experiments()
        exp = next((item for item in experiments if item.name == experiment_name), None)
    if exp is None:
        return {}
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        exp_dir = MLRUNS_DIR / exp.experiment_id
        if not exp_dir.exists():
            return {}
        run_meta_files = sorted(
            (p for p in exp_dir.iterdir() if p.is_dir() and (p / "meta.yaml").exists()),
            key=lambda path: (path / "meta.yaml").stat().st_mtime,
            reverse=True,
        )
        if not run_meta_files:
            return {}
        run_meta = yaml.safe_load((run_meta_files[0] / "meta.yaml").read_text())
        run_id = run_meta["run_id"]
        metrics: dict[str, float] = {}
        metrics_dir = run_meta_files[0] / "metrics"
        if metrics_dir.exists():
            for metric_file in metrics_dir.iterdir():
                if metric_file.is_file():
                    try:
                        last_line = metric_file.read_text().strip().splitlines()[-1]
                        _, value, *_ = last_line.split()
                        metrics[metric_file.name] = float(value)
                    except Exception:
                        continue
        return {
            "status": "completed" if str(run_meta.get("status", "")).upper() == "3" else str(run_meta.get("status", "")).lower(),
            "experiment_name": experiment_name,
            "run_id": run_id,
            "start_time": run_meta.get("start_time"),
            "end_time": run_meta.get("end_time"),
            "metrics": metrics,
            "params": {},
            "artifacts": [a.name for a in (run_meta_files[0] / "artifacts").iterdir()] if (run_meta_files[0] / "artifacts").exists() else [],
        }

    run = runs[0]
    status = "completed" if run.info.status == "FINISHED" else run.info.status.lower()
    return {
        "status": status,
        "experiment_name": experiment_name,
        "run_id": run.info.run_id,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "artifacts": [a.path for a in client.list_artifacts(run.info.run_id)],
    }


def write_latest_training_result(info: dict) -> None:
    """Write a stable JSON summary for the DSA research workspace."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **info,
    }
    LATEST_TRAINING_RESULT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"\n[artifact] Latest training summary: {LATEST_TRAINING_RESULT}")


def _metric_subset(metrics: dict) -> dict:
    keys = [
        "1day.excess_return_with_cost.annualized_return",
        "1day.excess_return_with_cost.information_ratio",
        "1day.excess_return_with_cost.max_drawdown",
        "1day.excess_return_without_cost.annualized_return",
        "1day.excess_return_without_cost.information_ratio",
        "1day.excess_return_without_cost.max_drawdown",
    ]
    return {key: metrics[key] for key in keys if key in metrics}


def _format_metric(value: object) -> str:
    return f"{value:.5f}" if isinstance(value, float) else str(value)


def _dataset_range(params: dict) -> str:
    train_dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(params.get("dataset.kwargs.segments.train", "")))
    test_dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(params.get("dataset.kwargs.segments.test", "")))
    if train_dates and test_dates:
        return f"{train_dates[0]} ~ {test_dates[-1]}"
    return ""


def _strategy_name(params: dict) -> str:
    record = str(params.get("record", ""))
    if "TopkDropoutStrategy" in record:
        return "TopkDropoutStrategy"
    if "PortAnaRecord" in record:
        return "Qlib PortAnaRecord"
    return ""


def _experiment_conclusion(provider: str) -> tuple[str, str]:
    if provider == "cleaned":
        return (
            "清洗后数据 baseline 已跑通，异常成交量和极端事件已按清洗策略处理；结果可用于和原始合并数据对比，但仍不是可直接实盘的策略。",
            "在网页端继续对比 canonical/cleaned，并补充择时、风控、低换手和行业约束实验。",
        )
    return (
        "原始合并数据 baseline 已跑通；当前策略仍是长多排序轮动，没有大盘择时、清仓规则或风险预算。",
        "补充风险控制和市场择时实验，例如指数趋势过滤、波动率降仓、最大回撤保护、行业中性或低换手约束。",
    )


def build_experiment_record(info: dict) -> dict:
    params = info.get("params") if isinstance(info.get("params"), dict) else {}
    provider = str(info.get("provider") or "canonical")
    conclusion, next_step = _experiment_conclusion(provider)
    return {
        "run_id": info.get("run_id"),
        "experiment_name": info.get("experiment_name"),
        "status": info.get("status"),
        "provider": provider,
        "qlib_provider": info.get("provider_uri"),
        "config_path": info.get("source_config"),
        "runtime_config": info.get("runtime_config"),
        "model": params.get("model.class", "LGBModel"),
        "strategy": _strategy_name(params),
        "dataset_range": _dataset_range(params),
        "metrics": _metric_subset(info.get("metrics") or {}),
        "conclusion": conclusion,
        "next_step": next_step,
        "generated_at": info.get("generated_at") or datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def append_experiment_record(info: dict, report_dir: Path = REPORT_DIR) -> bool:
    run_id = str(info.get("run_id") or "").strip()
    if not run_id:
        return False

    with experiment_file_lock(report_dir):
        runs_path = report_dir / "experiment_runs.jsonl"
        log_path = report_dir / "experiment_log.md"

        existing_ids: set[str] = set()
        if runs_path.exists():
            for line in runs_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict) and item.get("run_id"):
                    existing_ids.add(str(item["run_id"]))
        if run_id in existing_ids:
            return False

        record = build_experiment_record(info)
        with runs_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        metrics = record.get("metrics") or {}
        metric_text = "；".join(f"{key}: {_format_metric(value)}" for key, value in metrics.items()) or "暂无关键指标"
        generated_at = str(record.get("generated_at") or "")[:10] or datetime.now().strftime("%Y-%m-%d")
        existing_log = log_path.read_text(encoding="utf-8").rstrip() if log_path.exists() else "# 实验记录本"
        block = "\n".join(
            [
                f"<!-- experiment-run:{run_id} -->",
                f"## {generated_at} {record.get('experiment_name')}",
                "",
                f"- 数据版本：{record.get('provider')} `{record.get('qlib_provider')}`",
                f"- 模型：{record.get('model')}；策略：{record.get('strategy') or '未记录'}。",
                f"- 区间：{record.get('dataset_range') or '未记录'}。",
                f"- 结果：{metric_text}。",
                f"- 结论：{record.get('conclusion')}",
                f"- 下一步：{record.get('next_step')}",
                f"<!-- /experiment-run:{run_id} -->",
            ]
        )
        log_path.write_text(existing_log + "\n\n" + block + "\n", encoding="utf-8")
    return True


def main():
    parser = argparse.ArgumentParser(description="Qlib Alpha158+LGB baseline")
    parser.add_argument(
        "--config",
        default=str(CONFIGS_DIR / "baseline_lgb_alpha158.yaml"),
        help="Path to Qlib workflow YAML",
    )
    parser.add_argument(
        "--experiment-name",
        default="baseline_lgb_alpha158",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--provider",
        default="canonical",
        choices=sorted(PROVIDER_URIS),
        help="Local Qlib provider to use: canonical or cleaned",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] Config not found: {config_path}")
        sys.exit(1)
    try:
        provider_uri = resolve_provider_uri(args.provider)
        if not provider_uri.exists():
            raise FileNotFoundError(f"Provider does not exist: {provider_uri}")
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)

    # ── Step 1: Ensure instruments ──
    try:
        ensure_csi300_instruments(provider_uri)
    except Exception as exc:
        print(f"[error] Failed to prepare CSI300 instruments: {exc}")
        sys.exit(1)
    runtime_config_path = write_runtime_config_with_provider(config_path, provider_uri, args.provider)

    # ── Step 2: Run training + backtest ──
    print(f"\n{'='*60}")
    print(f"[run] Source config: {config_path}")
    print(f"[run] Runtime config: {runtime_config_path}")
    print(f"[run] Provider: {args.provider} -> {provider_uri}")
    print(f"[run] Experiment: {args.experiment_name}")
    print(f"{'='*60}\n")

    run_qrun(str(runtime_config_path), experiment_name=args.experiment_name)

    # ── Step 3: Print summary ──
    print(f"\n{'='*60}")
    print("[done] Training + backtest complete.")
    print(f"{'='*60}")

    info = load_latest_metrics(args.experiment_name)
    if info:
        info.update(
            {
                "provider": args.provider,
                "provider_uri": str(provider_uri),
                "source_config": str(config_path),
                "runtime_config": str(runtime_config_path),
            }
        )
        write_latest_training_result(info)
        append_experiment_record({"generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), **info})
        print(f"\nRun ID: {info['run_id']}")
        print("\nMetrics:")
        for k, v in sorted(info["metrics"].items()):
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\nArtifacts: {info['artifacts']}")
        print(f"\nMLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
