#!/usr/bin/env python3
"""Custom Qlib handler: Alpha158 + Fundamental factors, with optional PCA."""
import numpy as np
import pandas as pd
from qlib.contrib.data.handler import Alpha158, DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.utils import init_instance_by_config


# ── Fundamental feature expressions ──────────────────────────────
# These reference .day.bin files in the features directory
FUNDAMENTAL_EXPRS = [
    ("$peTTM", "peTTM"),
    ("$pbMRQ", "pbMRQ"),
    ("$psTTM", "psTTM"),
    ("$pcfNcfTTM", "pcfNcfTTM"),
    ("$turn", "turn"),
    ("$roeAvg", "roeAvg"),
    ("$npMargin", "npMargin"),
    ("$gpMargin", "gpMargin"),
    ("$epsTTM", "epsTTM"),
    ("$YOYNI", "YOYNI"),
    ("$YOYEPSBasic", "YOYEPSBasic"),
    ("$YOYPNI", "YOYPNI"),
]

# Value/quality composite factors (computed from raw fundamentals)
COMPOSITE_EXPRS = [
    ("1/$peTTM", "EP"),           # Earnings yield
    ("1/$pbMRQ", "BP"),           # Book-to-price
    ("$roeAvg/$pbMRQ", "ROE_PB"), # Quality-value composite
    ("$npMargin*$roeAvg", "NM_ROE"), # Profitability composite
]


class Alpha158Fund(Alpha158):
    """Alpha158 + fundamental factors handler."""

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        config = Alpha158.get_feature_config(self, conf)

        # Add fundamental expressions
        config = list(config)  # convert from tuple if needed
        for expr, name in FUNDAMENTAL_EXPRS + COMPOSITE_EXPRS:
            config.append(([expr], [name]))

        return config


def create_alpha158fund_handler_config(
    start_time, end_time, fit_start_time, fit_end_time, instruments="csi300"
):
    """Create handler config dict for YAML use."""
    return {
        "class": "Alpha158Fund",
            "module_path": "quant_workflow.experimental.fundamental_handler",
        "kwargs": {
            "start_time": start_time,
            "end_time": end_time,
            "fit_start_time": fit_start_time,
            "fit_end_time": fit_end_time,
            "instruments": instruments,
        },
    }
