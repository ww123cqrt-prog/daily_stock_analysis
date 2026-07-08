import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "_symbols.py"
    spec = importlib.util.spec_from_file_location("script_symbols", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SymbolsTest(unittest.TestCase):
    def test_to_qlib_name_keeps_existing_exchange_prefixes(self):
        module = load_module()

        self.assertEqual(module.to_qlib_name("SZ000001"), "SZ000001")
        self.assertEqual(module.to_qlib_name("sh600000"), "SH600000")

    def test_to_qlib_name_maps_a_share_prefixes(self):
        module = load_module()

        self.assertEqual(module.to_qlib_name("000001"), "SZ000001")
        self.assertEqual(module.to_qlib_name("600000"), "SH600000")
        self.assertEqual(module.to_qlib_name("430001"), "BJ430001")

    def test_to_qlib_name_preserves_unknown_without_default_exchange(self):
        module = load_module()

        self.assertEqual(module.to_qlib_name("123456"), "123456")

    def test_to_qlib_name_can_default_unknown_to_sz_for_csi300_helpers(self):
        module = load_module()

        self.assertEqual(module.to_qlib_name("123456", default_exchange="SZ"), "SZ123456")

    def test_normalize_symbol_strips_common_suffixes_and_prefixes(self):
        module = load_module()

        self.assertEqual(module.normalize_symbol("000001.XSHE", strip_any_suffix=True), "000001")
        self.assertEqual(module.normalize_symbol("SH600000", strip_prefix=True), "600000")


if __name__ == "__main__":
    unittest.main()
