"""Read-only daily readiness check; QMT owns downloads and maintenance."""
import argparse
from datetime import date
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from quant_qmt_data import QMTClient, load_dataset


def main(argv=None):
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=os.getenv("QMT_DATASET_DIR"))
    parser.add_argument("--required-through", type=date.fromisoformat)
    args = parser.parse_args(argv)
    if args.input_dir:
        frame = load_dataset(args.input_dir)
        actual_end = frame.attrs["actual_end"]
        enough = args.required_through is None or (
            actual_end is not None and date.fromisoformat(actual_end) >= args.required_through)
        result = dict(status="ready" if enough else "insufficient_dates", read_only=True,
                      experiment_id=frame.attrs["experiment_id"], actual_end=actual_end)
    else:
        result = dict(QMTClient.from_env().status(), status="connected")
        enough = args.required_through is None
        if not enough:
            result.update(status="experiment_input_required", required_through=args.required_through.isoformat())
    print(json.dumps(result, ensure_ascii=False))
    return 0 if enough else 1


if __name__ == "__main__":
    raise SystemExit(main())
