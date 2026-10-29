"""Check the selected frozen experiment through Qlib without rebuilding data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

DSA_ROOT = Path(__file__).resolve().parents[2]
if str(DSA_ROOT) not in sys.path:
    sys.path.insert(0, str(DSA_ROOT))

from quant_workflow.scripts._paths import qmt_dataset_dir

REQUIRED_FIELDS = ("$open", "$high", "$low", "$close", "$volume")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("qmt_native",), default="qmt_native")
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args(argv)
    directory = (args.input_dir or qmt_dataset_dir()).resolve()
    from qlib.data import D
    from quant_workflow.quant_data_qlib_provider import _load_input, init_qlib_from_quant_data

    init_qlib_from_quant_data(input_dir=directory, kernels=1)
    context = _load_input(str(directory))
    metadata = context.frame.attrs
    selected = metadata["universe"][:2]
    sample = D.features(selected, list(REQUIRED_FIELDS),
        start_time=context.calendar[-min(5, len(context.calendar))],
        end_time=context.calendar[-1], disk_cache=0)
    if sample.empty:
        raise ValueError("Qlib returned no features for the frozen selection")
    print(json.dumps({
        "provider": "qmt_native", "status": "readable",
        "experiment_id": metadata["experiment_id"], "bars_sha256": metadata["bars_sha256"],
        "price_basis": metadata["price_basis"], "rows": len(context.frame),
        "calendar": {"start": metadata["calendar"][0], "end": metadata["calendar"][-1], "count": len(context.calendar)},
        "universe": metadata["universe"], "universe_kind": metadata["universe_kind"],
        "missing_symbols": metadata["missing_symbols"],
        "feature_rows": len(sample), "missing_feature_rows": int(sample.isna().any(axis=1).sum()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
