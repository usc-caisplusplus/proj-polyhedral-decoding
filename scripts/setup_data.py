#!/usr/bin/env python3
"""
Extract bundled dataset zips into ./data/ so paths match the rest of the repo.

Expected archives at repository root (same layout as Google Drive bundles):
  - Singapore.zip   -> contains top-level folder ``Singapore/*.pt``
  - NewYork.zip     -> contains top-level folder ``NewYork/*.pt``

After extraction:
  ./data/Singapore/point_point_dis_data.pt
  ./data/NewYork/...

Usage (from repo root):

    python scripts/setup_data.py
    python scripts/setup_data.py --force   # re-extract over existing files
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def extract_zip(zip_path: Path, dest: Path, force: bool) -> None:
    if not zip_path.is_file():
        print(f"Skip (not found): {zip_path}", file=sys.stderr)
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        # First member is usually the folder name; avoid nuking unrelated files.
        names = zf.namelist()
        if not names:
            print(f"Skip (empty zip): {zip_path}", file=sys.stderr)
            return
        dest.mkdir(parents=True, exist_ok=True)
        if not force:
            # If top-level folder already has .pt files, skip
            top = names[0].split("/")[0].rstrip("/") or None
            if top:
                marker = dest / top
                if marker.is_dir() and any(marker.glob("*.pt")):
                    print(f"Skip (already present): {dest / top}  (use --force to re-extract)")
                    return
        print(f"Extracting {zip_path.name} -> {dest}")
        zf.extractall(dest)


def main() -> None:
    p = argparse.ArgumentParser(description="Extract Singapore.zip / NewYork.zip into ./data/")
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO,
        help="Repository root (default: parent of scripts/)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Destination for extraction (default: <repo-root>/data)",
    )
    p.add_argument("--force", action="store_true", help="Extract even if ./data/<City>/ already has .pt files")
    args = p.parse_args()

    root = args.repo_root.resolve()
    dest = (args.dest or (root / "data")).resolve()

    for name in ("Singapore.zip", "NewYork.zip"):
        extract_zip(root / name, dest, args.force)

    print("Done. Expect ablation paths like:", dest / "Singapore" / "point_point_dis_data.pt")


if __name__ == "__main__":
    main()
