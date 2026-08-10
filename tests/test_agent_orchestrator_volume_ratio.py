# -*- coding: utf-8 -*-
"""Regression: LLM 量比占位 0 不得覆盖确定性 N/A / 真实量比。"""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

from src.agent.orchestrator import AgentOrchestrator
from src.agent.executor import AgentExecutor
from src.agent.llm_adapter import LLMResponse
from src.agent.protocols import AgentContext
from src.agent.tools.registry import ToolRegistry
from src.core.pipeline import StockAnalysisPipeline


class TestVolumeRatioConsistency(unittest.TestCase):
    def _orch(self) -> AgentOrchestrator:
        return AgentOrchestrator(tool_registry=MagicMock(), llm_adapter=MagicMock())

    def test_llm_zero_volume_ratio_becomes_na_when_realtime_missing(self):
        orch = self._orch()
        ctx = AgentContext(query="test", stock_code="00700.HK", stock_name="腾讯控股")
        ctx.set_data(
            "realtime_quote",
            {"price": 490.4, "turnover_rate": 0.43, "source": "longbridge"},
        )
        ctx.set_data(
            "trend_result",
            {"volume_status": "放量上涨", "volume_trend": "成交量放大", "volume_ratio_5d": 0.0},
        )

        payload = {
            "decision_type": "hold",
            "analysis_summary": "观望",
            "dashboard": {
                "data_perspective": {
                    "volume_analysis": {
                        "volume_ratio": 0,
                        "volume_status": "放量",
                        "turnover_rate": 0,
                        "volume_meaning": "模型编造",
                    }
                }
            },
        }

        normalized = orch._finalize_dashboard_payload(payload, ctx)
        vol = normalized["dashboard"]["data_perspective"]["volume_analysis"]
        self.assertEqual(vol["volume_ratio"], "N/A")
        self.assertEqual(vol["turnover_rate"], 0.43)

    def test_trend_volume_ratio_5d_fills_when_realtime_missing(self):
        orch = self._orch()
        ctx = AgentContext(query="test", stock_code="00700.HK", stock_name="腾讯控股")
        ctx.set_data("realtime_quote", {"price": 490.4, "source": "longbridge"})
        ctx.set_data(
            "trend_result",
            {
                "volume_status": "放量上涨",
                "volume_trend": "量价齐升",
                "volume_ratio_5d": 1.24,
            },
        )

        payload = {
            "decision_type": "hold",
            "dashboard": {
                "data_perspective": {
                    "volume_analysis": {"volume_ratio": 0, "volume_status": "", "turnover_rate": 0}
                }
            },
        }
        normalized = orch._finalize_dashboard_payload(payload, ctx)
        vol = normalized["dashboard"]["data_perspective"]["volume_analysis"]
        self.assertEqual(vol["volume_ratio"], 1.24)

    def test_realtime_volume_ratio_wins_over_llm(self):
        orch = self._orch()
        ctx = AgentContext(query="test", stock_code="00700.HK", stock_name="腾讯控股")
        ctx.set_data(
            "realtime_quote",
            {"price": 490.4, "volume_ratio": 1.31, "turnover_rate": 0.5},
        )

        payload = {
            "decision_type": "hold",
            "dashboard": {
                "data_perspective": {
                    "volume_analysis": {"volume_ratio": 0, "turnover_rate": 0}
                }
            },
        }
        normalized = orch._finalize_dashboard_payload(payload, ctx)
        vol = normalized["dashboard"]["data_perspective"]["volume_analysis"]
        self.assertEqual(vol["volume_ratio"], 1.31)
        self.assertEqual(vol["turnover_rate"], 0.5)

    def test_fill_volume_ratio_from_trend_helper_semantics(self):
        """Pipeline writes the daily fallback into the shared realtime snapshot."""
        quote = SimpleNamespace(code="00700.HK", volume_ratio=None)
        trend = SimpleNamespace(volume_ratio_5d=1.24)

        resolved = StockAnalysisPipeline._fill_volume_ratio_from_trend(
            quote,
            trend,
            quote.code,
        )

        self.assertIs(resolved, quote)
        self.assertEqual(quote.volume_ratio, 1.24)

    def test_single_executor_reconciles_volume_ratio_from_deterministic_context(self):
        cases = [
            (
                "realtime wins",
                {"volume_ratio": 1.31},
                {"volume_ratio_5d": 1.24},
                0,
                1.31,
            ),
            (
                "trend fallback",
                {"volume_ratio": None},
                {"volume_ratio_5d": 1.24},
                "N/A",
                1.24,
            ),
            (
                "missing is unavailable",
                {"volume_ratio": 0},
                {"volume_ratio_5d": None},
                None,
                "N/A",
            ),
        ]

        for label, realtime, trend, llm_ratio, expected in cases:
            with self.subTest(label=label):
                adapter = MagicMock()
                adapter.call_with_tools.return_value = LLMResponse(
                    content=json.dumps(
                        {
                            "decision_type": "hold",
                            "dashboard": {
                                "data_perspective": {
                                    "volume_analysis": {"volume_ratio": llm_ratio}
                                }
                            },
                        }
                    ),
                    provider="test",
                    model="test-model",
                )
                executor = AgentExecutor(ToolRegistry(), adapter, max_steps=1)

                result = executor.run(
                    "Analyze 00700.HK",
                    context={
                        "stock_code": "00700.HK",
                        "realtime_quote": realtime,
                        "trend_result": trend,
                    },
                )

                self.assertTrue(result.success)
                self.assertEqual(
                    result.dashboard["dashboard"]["data_perspective"]["volume_analysis"]["volume_ratio"],
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
