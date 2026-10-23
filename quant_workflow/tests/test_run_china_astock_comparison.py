import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_china_astock_comparison.py"
    spec = importlib.util.spec_from_file_location("run_china_astock_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_build_comparison_fixture(scripts: Path) -> None:
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "build_comparison.py").write_text(
        "\n".join(
            [
                "import argparse",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--full-search', required=True)",
                "parser.add_argument('--evaluation', required=True)",
                "parser.add_argument('--out-csv', required=True)",
                "parser.add_argument('--out-md', required=True)",
                "args = parser.parse_args()",
                "Path(args.out_csv).write_text('strategy,score\\ndual_ma,1.0\\n', encoding='utf-8')",
                "Path(args.out_md).write_text('# comparison\\n', encoding='utf-8')",
                "",
            ]
        ),
        encoding="utf-8",
    )


class RunChinaAstockComparisonTest(unittest.TestCase):
    def test_builds_safe_default_paths(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "china-astock-quant"
            write_build_comparison_fixture(project_root / "scripts")

            paths = module.build_paths(project_root, None, None, None, timestamp="20260705_010203")

        self.assertEqual(paths.project_root.name, "china-astock-quant")
        self.assertEqual(paths.full_search, paths.project_root / "reports" / "full_search.json")
        self.assertEqual(paths.evaluation, paths.project_root / "reports" / "evaluate_winners.csv")
        self.assertEqual(paths.out_csv.parts[-3:], (
            "quant_workflow", "reports", "china_astock_strategy_comparison_20260705_010203.csv",
        ))
        self.assertEqual(paths.out_md.parts[-3:], (
            "quant_workflow", "reports", "china_astock_strategy_comparison_20260705_010203.md",
        ))

    def test_rejects_project_root_without_expected_script(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            with self.assertRaises(ValueError):
                module.build_paths(root, None, None, None, timestamp="20260705_010203")

    def test_python_bin_defaults_to_current_interpreter_even_when_project_venv_exists(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "china-astock-quant"
            scripts = root / "scripts"
            python_bin = root / ".venv" / "bin" / "python"
            scripts.mkdir(parents=True)
            python_bin.parent.mkdir(parents=True)
            (scripts / "build_comparison.py").write_text("print('ok')\n", encoding="utf-8")
            python_bin.write_text("#!/bin/sh\n", encoding="utf-8")

            self.assertEqual(module._python_bin(root), sys.executable)

    def test_runs_build_comparison_and_updates_latest_aliases(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "china-astock-quant"
            reports = project_root / "reports"
            scripts = project_root / "scripts"
            reports.mkdir(parents=True)
            write_build_comparison_fixture(scripts)
            (reports / "full_search.json").write_text(
                json.dumps(
                    {
                        "args": {"universe": "demo", "start": "2020-01-01", "end": "2020-02-01", "n_symbols_loaded": 2},
                        "per_strategy": {
                            "dual_ma": {
                                "rows": [
                                    {
                                        "params": "fast=5,slow=30,atr_stop_mult=3.0",
                                        "capital": 100000,
                                        "cagr": 0.1,
                                        "sharpe": 1.2,
                                        "sortino": 1.5,
                                        "max_dd": -0.05,
                                        "calmar": 2.0,
                                        "monthly_win_rate": 0.6,
                                        "stability_score": 0.8,
                                        "n_trades": 3,
                                        "final_equity": 110000,
                                    }
                                ]
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "strategy": "dual_ma_alt",
                        "params": "atr_stop_mult=2.0,fast=5,slow=60",
                        "period": "y2020",
                        "capital": 100000,
                        "cagr": 0.1,
                        "sharpe": 1.0,
                        "max_dd": -0.03,
                        "calmar": 3.0,
                        "monthly_win_rate": 0.7,
                    }
                ]
            ).to_csv(reports / "evaluate_winners.csv", index=False)
            out_dir = base / "quant_workflow" / "reports"

            status = module.run_report(project_root, out_dir, timestamp="20260705_010203")

            csv_path = out_dir / "china_astock_strategy_comparison_20260705_010203.csv"
            md_path = out_dir / "china_astock_strategy_comparison_20260705_010203.md"
            latest_json = out_dir / "china_astock_strategy_comparison_latest.json"

            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["project_root"], str(project_root.resolve()))
            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertEqual(out_dir.joinpath("china_astock_strategy_comparison_latest.csv").read_text(encoding="utf-8"), csv_path.read_text(encoding="utf-8"))
            self.assertEqual(out_dir.joinpath("china_astock_strategy_comparison_latest.md").read_text(encoding="utf-8"), md_path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(latest_json.read_text(encoding="utf-8"))["output_md"], str(md_path.resolve()))

    def test_times_out_when_build_comparison_hangs(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            project_root = base / "china-astock-quant"
            reports = project_root / "reports"
            scripts = project_root / "scripts"
            scripts.mkdir(parents=True)
            reports.mkdir(parents=True)
            (scripts / "build_comparison.py").write_text(
                "import time\n"
                "time.sleep(5)\n",
                encoding="utf-8",
            )
            (reports / "full_search.json").write_text("{}", encoding="utf-8")
            (reports / "evaluate_winners.csv").write_text("strategy,params\n", encoding="utf-8")
            out_dir = base / "quant_workflow" / "reports"

            with self.assertRaises(TimeoutError) as ctx:
                module.run_report(project_root, out_dir, timestamp="20260705_010203", timeout_seconds=0.1)

            self.assertIn("timed out", str(ctx.exception))
            self.assertFalse((out_dir / "china_astock_strategy_comparison_latest.json").exists())

    def test_timeout_seconds_rejects_invalid_values(self):
        module = load_module()

        for value in (-1, 0, "abc"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    module._timeout_seconds(value)
                self.assertIn("positive number", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
