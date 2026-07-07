import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_csi300.py"
    spec = importlib.util.spec_from_file_location("generate_csi300", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class GenerateCsi300Test(unittest.TestCase):
    def test_loads_csi300_from_local_stock_list(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            stock_list = Path(tmp) / "stock_lists.json"
            stock_list.write_text(
                json.dumps(
                    {
                        "all": [],
                        "csi300": ["000001.XSHE", "600000.XSHG", "430001"],
                    }
                ),
                encoding="utf-8",
            )

            symbols = module.load_csi300_symbols(stock_list)

        self.assertEqual(symbols, ["000001", "600000", "430001"])

    def test_converts_symbols_to_qlib_names(self):
        module = load_module()

        names = [module.to_qlib_name(symbol) for symbol in ["000001", "600000", "430001"]]

        self.assertEqual(names, ["SZ000001", "SH600000", "BJ430001"])

    def test_dry_run_does_not_write_output(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_list = root / "stock_lists.json"
            calendar = root / "day.txt"
            output = root / "csi300.txt"
            stock_list.write_text(json.dumps({"csi300": ["000001.XSHE", "600000.XSHG"]}), encoding="utf-8")
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            status = module.write_csi300(output, calendar, stock_list, root / "missing-astock", dry_run=True)

        self.assertEqual(status["status"], "dry-run")
        self.assertEqual(status["symbols"], 2)
        self.assertFalse(output.exists())

    def test_dry_run_requires_cached_stock_list_without_network_fallback(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_list = root / "stock_lists.json"
            calendar = root / "day.txt"
            output = root / "csi300.txt"
            stock_list.write_text(json.dumps({"csi300": []}), encoding="utf-8")
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            original = module._load_csi300_from_project
            module._load_csi300_from_project = lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("network fallback must not be used during dry-run")
            )
            try:
                with self.assertRaises(ValueError) as ctx:
                    module.write_csi300(output, calendar, stock_list, root / "astock", dry_run=True)
            finally:
                module._load_csi300_from_project = original

        self.assertIn("cached CSI300", str(ctx.exception))
        self.assertFalse(output.exists())

    def test_write_csi300_creates_instrument_file(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stock_list = root / "stock_lists.json"
            calendar = root / "day.txt"
            output = root / "instruments" / "csi300.txt"
            stock_list.write_text(json.dumps({"csi300": ["000001.XSHE", "600000.XSHG"]}), encoding="utf-8")
            calendar.write_text("2026-01-02\n2026-01-05\n", encoding="utf-8")

            status = module.write_csi300(output, calendar, stock_list, root / "missing-astock")
            text = output.read_text(encoding="utf-8")

        self.assertEqual(status["status"], "completed")
        self.assertIn("SZ000001\t2026-01-02\t2026-01-05", text)
        self.assertIn("SH600000\t2026-01-02\t2026-01-05", text)


if __name__ == "__main__":
    unittest.main()
