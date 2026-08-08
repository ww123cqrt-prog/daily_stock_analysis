"""Research bounded parameter improvements for the two-ETF momentum rule.

This is deterministic rule research, not machine-learning training. Candidate
selection uses only the declared development split. Later periods are reported
without being allowed to change the selected candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root, reports_root  # noqa: E402
from etf_momentum_rotation import (  # noqa: E402
    build_signal_table,
    load_price_data,
    sanitize_json_value,
    simulate_open_to_open,
    summarize_periods,
    with_execution_lag,
)
from quant_data_contract import resolve_dataset_component  # noqa: E402


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "etf_momentum_improvement.yaml"
)
DEFAULT_REPORT_DIR = reports_root() / "etf_momentum_rotation" / "improvement"


def normalize_commission_wan(value: float) -> dict[str, float]:
    """Convert a Chinese ``万X`` single-side commission into explicit units."""

    amount = float(value)
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("commission_wan_each_side must be finite and nonnegative")
    rate = amount / 10_000.0
    return {
        "commission_wan_each_side": amount,
        "commission_bps_each_side": amount,
        "commission_pct_each_side": rate * 100.0,
        "commission_rate_each_side": rate,
        "round_trip_commission_bps": amount * 2.0,
    }


def normalize_time_splits(
    raw_splits: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Validate chronological, non-overlapping research periods."""

    if not raw_splits:
        raise ValueError("time_splits must not be empty")
    result: list[dict[str, object]] = []
    previous_end: pd.Timestamp | None = None
    for name, raw in raw_splits.items():
        start = pd.Timestamp(raw.get("start")).normalize()
        end = pd.Timestamp(raw.get("end")).normalize()
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"time split {name} has an invalid date")
        if start > end:
            raise ValueError(f"time split {name} starts after it ends")
        if previous_end is not None and start <= previous_end:
            raise ValueError(f"time split {name} overlap is not allowed")
        result.append(
            {
                "name": str(name),
                "start": str(start.date()),
                "end": str(end.date()),
            }
        )
        previous_end = end
    return result


def _indicator_ready_table(
    data: Mapping[str, pd.DataFrame],
    *,
    lookback_days: int,
    moving_average_days: int,
) -> pd.DataFrame:
    table = build_signal_table(
        data,
        lookback_days=lookback_days,
        moving_average_days=moving_average_days,
    )
    required = [
        column for column in table if "return_" in column or "_ma_" in column
    ]
    ready = table.loc[table[required].notna().all(axis=1)].reset_index(drop=True)
    if ready.empty:
        raise ValueError(
            f"rule {lookback_days}/{moving_average_days} has no rows after warm-up"
        )
    return ready


def _yearly_metrics(periods: pd.DataFrame) -> list[dict[str, object]]:
    years = sorted(
        pd.to_datetime(periods["date"], errors="coerce").dropna().dt.year.unique()
    )
    rows: list[dict[str, object]] = []
    for year in years:
        summary = summarize_periods(
            periods,
            start_date=f"{int(year)}-01-01",
            end_date=f"{int(year)}-12-31",
        )
        if int(summary["periods"]) == 0:
            continue
        rows.append(
            {
                "year": int(year),
                **summary,
                "positive": bool(float(summary["total_return"]) > 0),
            }
        )
    return rows


def evaluate_variant(
    data: Mapping[str, pd.DataFrame],
    *,
    lookback_days: int,
    moving_average_days: int,
    side_cost_bps: float,
    splits: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    table = _indicator_ready_table(
        data,
        lookback_days=lookback_days,
        moving_average_days=moving_average_days,
    )
    result = simulate_open_to_open(
        with_execution_lag(table, lag_days=1),
        side_cost_bps=side_cost_bps,
    )
    metrics = dict(result["metrics"])
    periods = result["periods"]
    split_metrics = {
        str(split["name"]): summarize_periods(
            periods,
            start_date=str(split["start"]),
            end_date=str(split["end"]),
        )
        for split in splits
    }
    yearly = _yearly_metrics(periods)
    return {
        "lookback_days": int(lookback_days),
        "moving_average_days": int(moving_average_days),
        "side_cost_bps_each_side": float(side_cost_bps),
        "full_period": metrics,
        "splits": split_metrics,
        "switch_count": int(metrics["switch_count"]),
        "yearly": yearly,
        "positive_years": int(sum(bool(row["positive"]) for row in yearly)),
        "year_count": len(yearly),
    }


def select_development_candidate(
    variants: Sequence[Mapping[str, object]],
    *,
    max_drawdown_abs: float,
    selection_split: str = "development",
    original_rule: tuple[int, int] = (20, 20),
) -> Mapping[str, object]:
    """Choose using development metrics only; full/validation data are ignored."""

    limit = float(max_drawdown_abs)
    eligible: list[Mapping[str, object]] = []
    for variant in variants:
        development = variant["splits"][selection_split]
        annualized = float(development["annualized_return"])
        drawdown = abs(float(development["max_drawdown"]))
        if math.isfinite(annualized) and math.isfinite(drawdown) and drawdown <= limit:
            eligible.append(variant)
    if not eligible:
        raise ValueError("no parameter variant passed the development drawdown limit")

    def ranking_key(item: Mapping[str, object]) -> tuple[float, float, int, int]:
        development = item["splits"][selection_split]
        parameter_distance = abs(int(item["lookback_days"]) - original_rule[0]) + abs(
            int(item["moving_average_days"]) - original_rule[1]
        )
        return (
            -float(development["annualized_return"]),
            abs(float(development["max_drawdown"])),
            int(item.get("switch_count", 0)),
            parameter_distance,
        )

    return min(eligible, key=ranking_key)


def _metric_comparison(
    original: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, float]:
    original_annualized = float(original["annualized_return"])
    candidate_annualized = float(candidate["annualized_return"])
    original_drawdown = float(original["max_drawdown"])
    candidate_drawdown = float(candidate["max_drawdown"])
    return {
        "original_annualized_return": original_annualized,
        "candidate_annualized_return": candidate_annualized,
        "annualized_return_delta": candidate_annualized - original_annualized,
        "original_max_drawdown": original_drawdown,
        "candidate_max_drawdown": candidate_drawdown,
        "max_drawdown_improvement": abs(original_drawdown)
        - abs(candidate_drawdown),
    }


def build_candidate_comparison(
    *,
    original: Mapping[str, object],
    candidate: Mapping[str, object],
    validation_split: str,
    grid_previously_exposed: bool,
) -> dict[str, object]:
    same_rule = (
        int(original["lookback_days"]) == int(candidate["lookback_days"])
        and int(original["moving_average_days"])
        == int(candidate["moving_average_days"])
    )
    if same_rule:
        status = "original_rule_retained"
    elif grid_previously_exposed:
        status = "retrospective_improvement_candidate"
    else:
        status = "prospective_validation_candidate"
    return {
        "status": status,
        "qualified_strategy": False,
        "original_rule": {
            "lookback_days": int(original["lookback_days"]),
            "moving_average_days": int(original["moving_average_days"]),
        },
        "candidate_rule": {
            "lookback_days": int(candidate["lookback_days"]),
            "moving_average_days": int(candidate["moving_average_days"]),
        },
        "development": _metric_comparison(
            original["splits"]["development"],
            candidate["splits"]["development"],
        ),
        "validation": _metric_comparison(
            original["splits"][validation_split],
            candidate["splits"][validation_split],
        ),
        "full_period": _metric_comparison(
            original["full_period"], candidate["full_period"]
        ),
        "reason": (
            "The candidate was selected only from the development split, but the "
            "full parameter grid had already been viewed in earlier research. It is "
            "therefore a retrospective candidate, not a clean holdout-qualified strategy."
        ),
    }


def _find_variant(
    variants: Sequence[Mapping[str, object]], *, lookback: int, moving_average: int
) -> Mapping[str, object]:
    for variant in variants:
        if (
            int(variant["lookback_days"]) == lookback
            and int(variant["moving_average_days"]) == moving_average
        ):
            return variant
    raise ValueError(f"parameter grid is missing rule {lookback}/{moving_average}")


def _cost_stress(
    data: Mapping[str, pd.DataFrame],
    *,
    rules: Mapping[str, tuple[int, int]],
    commission_bps: float,
    slippage_scenarios: Sequence[float],
    legacy_total_cost_bps: float,
    splits: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    result: dict[str, list[dict[str, object]]] = {}
    scenarios = list(dict.fromkeys([0.0, *[float(value) for value in slippage_scenarios]]))
    for name, (lookback, moving_average) in rules.items():
        rows: list[dict[str, object]] = []
        for slippage in scenarios:
            metrics = evaluate_variant(
                data,
                lookback_days=lookback,
                moving_average_days=moving_average,
                side_cost_bps=commission_bps + slippage,
                splits=splits,
            )["full_period"]
            rows.append(
                {
                    "scenario": "commission_only"
                    if slippage == 0
                    else f"commission_plus_{slippage:g}bp_slippage_each_side",
                    "commission_bps_each_side": commission_bps,
                    "slippage_bps_each_side": slippage,
                    "total_side_cost_bps": commission_bps + slippage,
                    "metrics": metrics,
                }
            )
        legacy = evaluate_variant(
            data,
            lookback_days=lookback,
            moving_average_days=moving_average,
            side_cost_bps=legacy_total_cost_bps / 2.0,
            splits=splits,
        )["full_period"]
        rows.append(
            {
                "scenario": "legacy_report_total_cost",
                "commission_bps_each_side": None,
                "slippage_bps_each_side": None,
                "total_side_cost_bps": legacy_total_cost_bps / 2.0,
                "round_trip_total_cost_bps": legacy_total_cost_bps,
                "metrics": legacy,
            }
        )
        result[name] = rows
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
    return {"git_commit": commit, "git_dirty": dirty}


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_text(
        path,
        json.dumps(
            sanitize_json_value(dict(payload)), ensure_ascii=False, indent=2
        )
        + "\n",
    )


def _pct(value: object) -> str:
    return f"{float(value):.2%}"


def format_fee_percentage(value: float) -> str:
    return f"{float(value):.5f}%"


def _markdown(report: Mapping[str, object]) -> str:
    fee = report["fee"]
    original = report["original_rule_result"]
    candidate = report["candidate_rule_result"]
    comparison = report["candidate_comparison"]
    lines = [
        "# 创业板ETF-纳指ETF动量轮动规则改进研究",
        "",
        "## 先说结果",
        "",
        f"- 原始规则：{original['lookback_days']}日动量 + {original['moving_average_days']}日均线。",
        (
            f"- 早期数据选出的候选：{candidate['lookback_days']}日动量 + "
            f"{candidate['moving_average_days']}日均线。"
        ),
        f"- 当前状态：`{comparison['status']}`，不是已合格策略。",
        "- 这是固定规则的参数研究，不是训练机器学习模型。",
        "",
        "## 你的交易费率",
        "",
        (
            f"- 单边佣金：万{fee['commission_wan_each_side']:.3f}，即 "
            f"{format_fee_percentage(fee['commission_pct_each_side'])}，也就是 "
            f"{fee['commission_bps_each_side']:.3f}bp。"
        ),
        f"- 买入一次再卖出一次的纯佣金：{fee['round_trip_commission_bps']:.3f}bp。",
        (
            "- 基准结果不含滑点，也没有加入最低5元佣金；滑点在压力测试里"
            "单列。ETF不收股票印花税。"
        ),
        "",
        "## 原始规则与候选规则",
        "",
        "| 规则 | 全期年化 | 全期总收益 | 最大回撤 | 正收益年份 | 换仓状态变化 |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {original['lookback_days']}/{original['moving_average_days']} 原始规则 | "
            f"{_pct(original['full_period']['annualized_return'])} | "
            f"{_pct(original['full_period']['total_return'])} | "
            f"{_pct(original['full_period']['max_drawdown'])} | "
            f"{original['positive_years']}/{original['year_count']} | "
            f"{original['switch_count']} |"
        ),
        (
            f"| {candidate['lookback_days']}/{candidate['moving_average_days']} 候选规则 | "
            f"{_pct(candidate['full_period']['annualized_return'])} | "
            f"{_pct(candidate['full_period']['total_return'])} | "
            f"{_pct(candidate['full_period']['max_drawdown'])} | "
            f"{candidate['positive_years']}/{candidate['year_count']} | "
            f"{candidate['switch_count']} |"
        ),
        "",
        "## 按时间分开看",
        "",
        "| 时段 | 原始规则年化 | 候选规则年化 | 原始回撤 | 候选回撤 |",
        "|---|---:|---:|---:|---:|",
    ]
    split_names = {
        "development": "2019-2022 选择期",
        "validation": "2023-2024 验证期",
        "later_observation": "2025-2026 后续观察",
    }
    for split in report["time_splits"]:
        name = str(split["name"])
        original_metrics = original["splits"][name]
        candidate_metrics = candidate["splits"][name]
        lines.append(
            f"| {split_names.get(name, name)} | {_pct(original_metrics['annualized_return'])} | "
            f"{_pct(candidate_metrics['annualized_return'])} | "
            f"{_pct(original_metrics['max_drawdown'])} | "
            f"{_pct(candidate_metrics['max_drawdown'])} |"
        )
    lines.extend(
        [
            "",
            (
                "候选规则只按2019-2022的结果选出。2023年以后的数据没有参与"
                "挑选。验证期里候选年化收益略低于原始规则，但回撤更小；后续"
                "观察期和全期则更好。"
            ),
            "",
            "## 成本压力测试",
            "",
            "| 规则 | 成本场景 | 年化收益 | 最大回撤 |",
            "|---|---|---:|---:|",
        ]
    )
    for rule_name, rows in report["cost_stress"].items():
        for row in rows:
            metrics = row["metrics"]
            lines.append(
                f"| {rule_name} | {row['scenario']} | "
                f"{_pct(metrics['annualized_return'])} | "
                f"{_pct(metrics['max_drawdown'])} |"
            )
    lines.extend(
        [
            "",
            "## 为什么还不能叫正式改进版",
            "",
            (
                "- 早先的报告已经展示过完整参数九宫格，因此这里的2023-2026"
                "结果并不是从未看过的纯盲测。"
            ),
            (
                "- 下一步应固定20/25规则，不再改参数，做新的前向纸面跟踪；"
                "不能继续看历史结果后反复换参数。"
            ),
            "- 本研究没有连接券商，也没有自动下单。",
            "",
        ]
    )
    return "\n".join(lines)


def run(
    *,
    config_path: Path = DEFAULT_CONFIG,
    report_dir: Path = DEFAULT_REPORT_DIR,
    quant_data_dir: Path | None = None,
    generated_at: str | None = None,
) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    fee = normalize_commission_wan(config["fee"]["commission_wan_each_side"])
    splits = normalize_time_splits(config["time_splits"])
    root = (
        Path(quant_data_dir).expanduser().resolve()
        if quant_data_dir is not None
        else quant_data_root().resolve()
    )
    data_config = config["data"]
    resolved = resolve_dataset_component(
        str(data_config["dataset_id"]),
        quant_data_root=root,
        price_basis=str(data_config["required_price_basis"]),
        require_status=str(data_config["required_status"]),
    )
    data = load_price_data(resolved.component_path)

    parameter_config = config["parameter_research"]
    variants = [
        evaluate_variant(
            data,
            lookback_days=int(lookback),
            moving_average_days=int(moving_average),
            side_cost_bps=float(fee["commission_bps_each_side"]),
            splits=splits,
        )
        for lookback in parameter_config["lookback_days"]
        for moving_average in parameter_config["moving_average_days"]
    ]
    original_config = config["strategy"]["original_rule"]
    original_rule = (
        int(original_config["lookback_days"]),
        int(original_config["moving_average_days"]),
    )
    original = _find_variant(
        variants, lookback=original_rule[0], moving_average=original_rule[1]
    )
    candidate = select_development_candidate(
        variants,
        max_drawdown_abs=float(
            parameter_config["max_development_drawdown_abs"]
        ),
        selection_split=str(parameter_config["selection_split"]),
        original_rule=original_rule,
    )
    policy = config["evidence_policy"]
    comparison = build_candidate_comparison(
        original=original,
        candidate=candidate,
        validation_split="validation",
        grid_previously_exposed=bool(policy["grid_previously_exposed"]),
    )
    candidate_rule = (
        int(candidate["lookback_days"]),
        int(candidate["moving_average_days"]),
    )
    stress = _cost_stress(
        data,
        rules={"original_20_20": original_rule, "candidate": candidate_rule},
        commission_bps=float(fee["commission_bps_each_side"]),
        slippage_scenarios=config["fee"]["stress_slippage_bps_each_side"],
        legacy_total_cost_bps=float(
            config["fee"]["legacy_report_total_cost_bps_round_trip"]
        ),
        splits=splits,
    )

    now = generated_at or datetime.now().astimezone().isoformat(timespec="seconds")
    parsed_now = pd.Timestamp(now)
    run_id = f"etf_momentum_improvement_{parsed_now.strftime('%Y%m%d_%H%M%S')}"
    project_root = Path(__file__).resolve().parents[2]
    report = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": now,
        "status": comparison["status"],
        "research_kind": "deterministic_rule_parameter_research",
        "strategy_name": str(config["strategy"]["name"]),
        "fee": {
            **fee,
            "minimum_commission_applied": bool(
                config["fee"]["minimum_commission_applied"]
            ),
            "minimum_commission_note": str(
                config["fee"]["minimum_commission_note"]
            ),
            "etf_stamp_duty_applied": bool(
                config["fee"]["etf_stamp_duty_applied"]
            ),
        },
        "data": {
            "dataset_id": resolved.dataset.dataset_id,
            "snapshot_id": resolved.dataset.manifest["snapshot_id"],
            "price_basis": resolved.price_basis,
            "provider": resolved.dataset.manifest["source"]["provider"],
            "date_range": resolved.dataset.manifest["date_range"],
            "manifest_sha256": _sha256(
                resolved.dataset.snapshot_path / "DATASET_MANIFEST.json"
            ),
        },
        "time_splits": splits,
        "selection": {
            "selection_split": str(parameter_config["selection_split"]),
            "selection_metric": str(parameter_config["selection_metric"]),
            "max_development_drawdown_abs": float(
                parameter_config["max_development_drawdown_abs"]
            ),
            "candidate_was_not_selected_from_validation_or_later_data": True,
            "grid_previously_exposed": bool(policy["grid_previously_exposed"]),
        },
        "original_rule_result": original,
        "candidate_rule_result": candidate,
        "candidate_comparison": comparison,
        "all_parameter_variants": variants,
        "cost_stress": stress,
        "dsa_replay_parameters": {
            "strategy_id": "momentum_rotation",
            "symbols": ["159915", "513100"],
            "initial_capital": float(config["dsa_replay"]["initial_capital_cny"]),
            "side_cost_pct": float(fee["commission_pct_each_side"]),
            "lot_size": int(config["dsa_replay"]["lot_size"]),
            "original": {
                "lookback_days": original_rule[0],
                "moving_average_days": original_rule[1],
            },
            "candidate": {
                "lookback_days": candidate_rule[0],
                "moving_average_days": candidate_rule[1],
            },
        },
        "reproducibility": {
            "config_path": str(config_path.relative_to(project_root)),
            "config_sha256": _sha256(config_path),
            **_git_metadata(project_root),
        },
        "boundaries": {
            "qualified_strategy": False,
            "original_rule_overwritten": False,
            "broker_connected": False,
            "automatic_orders": False,
            "base_result_excludes_slippage": True,
        },
    }

    report_dir = Path(report_dir).expanduser().resolve()
    run_dir = report_dir / "runs" / run_id
    markdown = _markdown(report)
    _atomic_json(run_dir / "result.json", report)
    _atomic_text(run_dir / "report.md", markdown)
    _atomic_json(report_dir / "etf_momentum_improvement_latest.json", report)
    _atomic_text(report_dir / "etf_momentum_improvement_latest.md", markdown)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research bounded improvements to the two-ETF momentum rule."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--quant-data-root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = run(
        config_path=args.config,
        report_dir=args.report_dir,
        quant_data_dir=args.quant_data_root,
    )
    print(f"ETF momentum improvement report: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
