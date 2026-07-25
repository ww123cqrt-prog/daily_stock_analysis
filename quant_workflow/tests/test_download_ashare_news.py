from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_ashare_news.py"
    spec = importlib.util.spec_from_file_location("download_ashare_news", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class DownloadAshareNewsTest(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_requires_bounded_default_universe(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            self.module.validate_bounded_request(symbols_file=None, limit=0)

    def test_writes_traceable_raw_jsonp_capture(self):
        raw_text = 'jQuery_catdesk({"result":{"cmsArticleWebOld":[{"code":"1","title":"news"}]}});'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.module.run_download(
                symbols=["600519"],
                raw_root=root / "raw" / "ashare_news",
                summary_dir=root / "reports" / "download_state",
                limit=1,
                sleep_seconds=0,
                fetch_text=lambda symbol, page_size: raw_text,
            )
            captures = list((root / "raw" / "ashare_news").rglob("*.json"))
            payload = json.loads(captures[0].read_text(encoding="utf-8"))

        self.assertEqual(result["ok"], 1)
        self.assertEqual(len(captures), 1)
        self.assertEqual(payload["source"], "eastmoney_news_search")
        self.assertEqual(payload["symbol"], "600519")
        self.assertEqual(payload["raw_text"], raw_text)
        self.assertEqual(len(payload["raw_text_sha256"]), 64)
        self.assertEqual(payload["payload"]["result"]["cmsArticleWebOld"][0]["title"], "news")
