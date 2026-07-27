#!/usr/bin/env python3
"""Build an immutable Qlib provider from the governed A-share qfq snapshot."""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import quant_data_root as default_quant_data_root  # noqa: E402
from merge_to_qlib import convert_to_bin  # noqa: E402
from quant_data_contract import (  # noqa: E402
    DataContractError,
    finalize_qlib_snapshot,
    publish_qlib_snapshot,
    resolve_dataset,
)


def build_qlib_v2(*, quant_data_root: Path, publish: bool = False) -> Path:
    root = Path(quant_data_root).expanduser().resolve()
    qfq = resolve_dataset(
        "ashare_daily_qfq",
        quant_data_root=root,
        require_basis="qfq",
    )
    run_id = (
        datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    snapshot_id = f"qlib-ashare-v2-{run_id}"
    staging = root / "staging" / f"qlib-v2-{run_id}" / snapshot_id
    upstream = {
        "dataset_id": qfq.dataset_id,
        "snapshot_id": str(qfq.manifest["snapshot_id"]),
        "content_sha256": str(qfq.manifest["content_sha256"]),
        "price_basis": str(qfq.manifest["price_basis"]),
    }
    result = convert_to_bin(
        qfq.snapshot_path / "bars",
        staging,
        overwrite=True,
        source_contract=upstream,
        force_unit_factor=True,
    )
    if result != 0:
        raise DataContractError("Qlib v2 conversion failed")
    finalize_qlib_snapshot(
        staging,
        dataset_id="qlib_ashare_v2",
        snapshot_id=snapshot_id,
        source={
            "provider": "quant_workflow",
            "interface": "v2_snapshot_to_qlib",
        },
        source_retrieved_at=str(qfq.manifest["source_retrieved_at"]),
        producer={"script": "build_qlib_v2.py", "version": "2"},
        upstream_snapshot=upstream,
    )
    if not publish:
        return staging
    dataset_root = root / "qlib" / "cn_data_ashare_v2"
    published = publish_qlib_snapshot(
        staging,
        dataset_root=dataset_root,
        quant_data_root=root,
        update_root_catalog=True,
    )
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quant-data-root", type=Path, default=default_quant_data_root()
    )
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path = build_qlib_v2(
            quant_data_root=args.quant_data_root,
            publish=args.publish,
        )
    except (DataContractError, OSError, ValueError, KeyError) as exc:
        print(f"Qlib v2 build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
