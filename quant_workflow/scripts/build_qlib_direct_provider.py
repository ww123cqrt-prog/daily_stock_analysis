#!/usr/bin/env python3
"""Validate an existing QMT input; Qlib now needs no generated provider copy."""
from pathlib import Path
import sys

DSA_ROOT = Path(__file__).resolve().parents[2]
if str(DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_ROOT))

from quant_workflow.scripts.check_qlib_data import main


if __name__ == "__main__":
    raise SystemExit(main())
