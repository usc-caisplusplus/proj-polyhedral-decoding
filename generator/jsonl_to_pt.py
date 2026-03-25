#!/usr/bin/env python3
"""
Convert generator JSONL output to the .pt format expected by run.py.

Usage (from repo root):
    python -m generator.jsonl_to_pt \
        --jsonl path/to/train.jsonl [val.jsonl test.jsonl ...] \
        --out   path/to/output.pt \
        --task  topological   # or directional | distance

Output: a single .pt file containing (X1, X2, L1, L2, Y) tensors, where:
    X1, X2 : float32 (N, max_len, 2)  padded coordinate arrays
    L1, L2 : int64   (N,)             actual vertex counts (incl. NaN sentinels)
    Y       : float32 or int64 (N,)   labels
              - topological: int64 class index
              - directional: int64 class index (0-15)
              - distance:    float32 scalar
"""

from __future__ import annotations

import argparse
import json
from typing import List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Label vocabularies (must match generator/labels.py)
# ---------------------------------------------------------------------------

TOPOLOGY_VOCAB = {
    "polygon-polygon":   ["disjoint", "touches", "intersects", "contains", "within", "equals"],
    "polyline-polygon":  ["disjoint", "touches", "within", "intersects"],
    "point-polygon":     ["disjoint", "inside"],
    "polygon-point":     ["disjoint", "contains"],
    "polyline-point":    ["disjoint", "intersects"],
    "polyline-polyline": ["disjoint", "intersects"],
}


def topology_label_to_idx(label: str, pair_type: str) -> int:
    vocab = TOPOLOGY_VOCAB.get(pair_type)
    if vocab is None:
        raise ValueError(f"Unknown pair_type: {pair_type}")
    if label not in vocab:
        raise ValueError(f"Label '{label}' not in vocab for {pair_type}: {vocab}")
    return vocab.index(label)


# ---------------------------------------------------------------------------
# Geometry deserialization → flat coordinate array (with NaN hole sentinels)
# ---------------------------------------------------------------------------

def geom_to_coords(geom: dict) -> np.ndarray:
    """
    Convert a JSON geometry dict (from io_utils.to_json_geom) to a flat
    (K, 2) float32 array.

    For polygons with holes, interior rings are appended after a [NaN, NaN]
    sentinel row — matching the format expected by moment_encoder.preprocess_polygon.

    For multipolygon, parts are concatenated with NaN sentinels between them
    (treated as the largest-part exterior only for now, since the moment
    encoder picks the largest piece anyway).
    """
    gtype = geom["type"]

    if gtype == "point":
        return np.array([[geom["coords"][0], geom["coords"][1]]], dtype=np.float32)

    if gtype == "polyline":
        coords = geom["coords"]
        if len(coords) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.array(coords, dtype=np.float32)

    if gtype == "polygon":
        raw = geom["coords"]
        coords = np.array(raw, dtype=np.float32).reshape(-1, 2) if raw else np.zeros((0, 2), dtype=np.float32)
        rings = [coords]
        for hole in geom.get("holes", []):
            rings.append(np.array([[np.nan, np.nan]], dtype=np.float32))
            rings.append(np.array(hole, dtype=np.float32))
        return np.concatenate(rings, axis=0)

    if gtype == "multipolygon":
        # Concatenate all parts with NaN separators; moment_encoder takes largest
        parts = []
        for i, part in enumerate(geom["parts"]):
            if i > 0:
                parts.append(np.array([[np.nan, np.nan]], dtype=np.float32))
            parts.append(geom_to_coords(part))
        return np.concatenate(parts, axis=0)

    raise ValueError(f"Unknown geometry type: {gtype}")


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def load_jsonl(paths: List[str]) -> List[dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def convert(records: List[dict], task: str) -> Tuple[torch.Tensor, ...]:
    # Extract coordinate arrays and lengths
    coords1 = [geom_to_coords(r["geom1"]) for r in records]
    coords2 = [geom_to_coords(r["geom2"]) for r in records]

    L1 = np.array([len(c) for c in coords1], dtype=np.int64)
    L2 = np.array([len(c) for c in coords2], dtype=np.int64)

    max_len = int(max(L1.max(), L2.max()))

    N = len(records)
    X1 = np.zeros((N, max_len, 2), dtype=np.float32)
    X2 = np.zeros((N, max_len, 2), dtype=np.float32)

    for i, c in enumerate(coords1):
        k = len(c)
        X1[i, :k] = c
    for i, c in enumerate(coords2):
        k = len(c)
        X2[i, :k] = c

    # Build labels
    if task == "topological":
        # Infer pair_type from first record
        pair_type = records[0]["pair_type"]
        Y = np.array(
            [topology_label_to_idx(r["label"], r["pair_type"]) for r in records],
            dtype=np.int64,
        )
    elif task == "directional":
        Y = np.array([r["label_id"] for r in records], dtype=np.int64)
    elif task == "distance":
        Y = np.array([r["label"] for r in records], dtype=np.float32)
    else:
        raise ValueError(f"Unknown task: {task}")

    return (
        torch.from_numpy(X1),
        torch.from_numpy(X2),
        torch.from_numpy(L1),
        torch.from_numpy(L2),
        torch.from_numpy(Y),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert generator JSONL to .pt")
    parser.add_argument(
        "--jsonl", nargs="+", required=True,
        help="One or more JSONL files to merge (e.g. train.jsonl val.jsonl test.jsonl)",
    )
    parser.add_argument("--out", required=True, help="Output .pt file path")
    parser.add_argument(
        "--task", required=True,
        choices=["topological", "directional", "distance"],
    )
    args = parser.parse_args()

    print(f"Loading {len(args.jsonl)} file(s)...")
    records = load_jsonl(args.jsonl)
    print(f"  {len(records)} records loaded")

    tensors = convert(records, args.task)
    X1, X2, L1, L2, Y = tensors
    print(f"  X1: {tuple(X1.shape)}, X2: {tuple(X2.shape)}")
    print(f"  L1 max={int(L1.max())}, L2 max={int(L2.max())}")
    print(f"  Y dtype={Y.dtype}, shape={tuple(Y.shape)}")

    torch.save(tensors, args.out)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
