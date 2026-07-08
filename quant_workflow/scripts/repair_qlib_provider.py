#!/usr/bin/env python3
"""Repair local Qlib provider metadata without rebuilding the full provider."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _paths import qlib_provider_uri  # noqa: E402
from _symbols import to_qlib_name  # noqa: E402


DEFAULT_PROVIDER = qlib_provider_uri()
BENCHMARK_SYMBOLS = {"SH000300"}


def read_instruments(path: Path) -> list[tuple[str, str, str]]:
    if not path.exists():
        return []
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append((parts[0].strip().upper(), parts[1], parts[2]))
    return rows


def write_instruments(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[str, tuple[str, str, str]] = {}
    for sym, start, end in rows:
        if sym in BENCHMARK_SYMBOLS:
            continue
        unique[sym] = (sym, start, end)
    lines = [f"{sym}\t{start}\t{end}" for sym, start, end in sorted(unique.values())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def repair_feature_dirs(provider: Path, dry_run: bool) -> list[tuple[str, str]]:
    features = provider / "features"
    renames: list[tuple[str, str]] = []
    if not features.exists():
        return renames

    for path in sorted(p for p in features.iterdir() if p.is_dir()):
        current = path.name.upper()
        repaired = to_qlib_name(current)
        if repaired == current:
            continue
        dst = features / repaired.lower()
        renames.append((current, repaired))
        if dry_run:
            continue
        if dst.exists():
            for child in path.iterdir():
                target = dst / child.name
                if not target.exists():
                    shutil.move(str(child), str(target))
            path.rmdir()
        else:
            path.rename(dst)
    return renames


def repair_instrument_file(path: Path, dry_run: bool) -> list[tuple[str, str]]:
    rows = read_instruments(path)
    repaired_rows: list[tuple[str, str, str]] = []
    changes: list[tuple[str, str]] = []
    for sym, start, end in rows:
        repaired = to_qlib_name(sym)
        if repaired != sym:
            changes.append((sym, repaired))
        if repaired in BENCHMARK_SYMBOLS:
            changes.append((sym, "<removed-benchmark>"))
            continue
        repaired_rows.append((repaired, start, end))

    if not dry_run:
        write_instruments(path, repaired_rows)
    return changes


def filter_instrument_file(
    path: Path,
    allowed_symbols: set[str],
    dry_run: bool,
) -> list[tuple[str, str]]:
    rows = read_instruments(path)
    kept: list[tuple[str, str, str]] = []
    changes: list[tuple[str, str]] = []
    for sym, start, end in rows:
        if sym in allowed_symbols:
            kept.append((sym, start, end))
        else:
            changes.append((sym, "<removed-not-in-all>"))

    if not dry_run:
        write_instruments(path, kept)
    return changes


def remove_orphan_feature_dirs(provider: Path, dry_run: bool) -> list[str]:
    features = provider / "features"
    all_symbols = {sym.lower() for sym, _, _ in read_instruments(provider / "instruments" / "all.txt")}
    allowed_extra = {sym.lower() for sym in BENCHMARK_SYMBOLS}
    removed: list[str] = []
    if not features.exists():
        return removed

    for path in sorted(p for p in features.iterdir() if p.is_dir()):
        if path.name in all_symbols or path.name in allowed_extra:
            continue
        removed.append(path.name)
        if not dry_run:
            shutil.rmtree(path)
    return removed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=Path, default=DEFAULT_PROVIDER)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    provider = args.provider
    if not provider.exists():
        if args.dry_run:
            print(f"DRY RUN: provider not found: {provider}")
            print("Feature directory renames: 0")
            print("all.txt: 0 metadata changes")
            print("csi300.txt: 0 metadata changes")
            print("csi300.txt: 0 non-all removals")
            print("Orphan feature directories removed: 0")
            return 0
        print(f"ERROR: provider not found: {provider}")
        return 1

    dir_changes = repair_feature_dirs(provider, args.dry_run)
    print(f"Feature directory renames: {len(dir_changes)}")
    for old, new in dir_changes[:20]:
        print(f"  {old} -> {new}")

    inst_dir = provider / "instruments"
    for name in ["all.txt", "csi300.txt"]:
        path = inst_dir / name
        changes = repair_instrument_file(path, args.dry_run)
        print(f"{name}: {len(changes)} metadata changes")
        for old, new in changes[:20]:
            print(f"  {old} -> {new}")

    all_symbols = {sym for sym, _, _ in read_instruments(inst_dir / "all.txt")}
    csi_changes = filter_instrument_file(inst_dir / "csi300.txt", all_symbols, args.dry_run)
    print(f"csi300.txt: {len(csi_changes)} non-all removals")
    for old, new in csi_changes[:20]:
        print(f"  {old} -> {new}")

    orphan_dirs = remove_orphan_feature_dirs(provider, args.dry_run)
    print(f"Orphan feature directories removed: {len(orphan_dirs)}")
    for name in orphan_dirs[:20]:
        print(f"  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
