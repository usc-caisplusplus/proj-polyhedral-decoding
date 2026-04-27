#!/usr/bin/env python3
"""
Convert generator_3d JSONL files to .pt training tensors for run.py.

Output format (saved with torch.save):
    (X1, X2, L1, L2, FC1, FC2, Y)

where:
    X1, X2  : geometry tensors
    L1, L2  : length tensors (vertex/point counts for points/polylines; ignored for polyhedra)
    FC1,FC2 : face-count tensors — number of triangular faces per polyhedron
              (all zeros when the geometry is not a polyhedron)
    Y       : label tensor

Geometry tensor shapes per type:
    point      X: (B, 1, 3)      L: 1   FC: 0
    polyline   X: (B, maxN, 3)   L: N   FC: 0
    polyhedron X: (B, maxF, 3, 3)  L: 0  FC: F
                  where each face is (v0, v1, v2) as (3,3) rows

Polyhedron triangulation:
    A Polyhedron3D is an extruded prism: a 2D footprint polygon extruded from
    z_bot to z_top.  We produce the closed triangular mesh of the prism:
      - bottom cap  (fan from vertex 0 of the bottom ring)
      - top cap     (fan, reversed winding for outward normals)
      - side quads  (2 triangles per edge of the footprint ring)

Label encoding:
    distance task    -> Y float32 (continuous)
    directional task -> Y int64   (0..47)
    topological task -> Y int64   (class index)

Usage:
    python utils/convert_jsonl_to_pt_3d.py \\
        --jsonl data/3d/polyhedron_polyhedron_topo/train.jsonl \\
               data/3d/polyhedron_polyhedron_topo/val.jsonl \\
               data/3d/polyhedron_polyhedron_topo/test.jsonl \\
        --out   data/3d/polyhedron_polyhedron_topological_data.pt \\
        --task  topological \\
        --pair_type polyhedron-polyhedron
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Topology label maps (must match labels_3d.py)
# ---------------------------------------------------------------------------

TOPO_LABELS: Dict[str, List[str]] = {
    "polyhedron-polyhedron": ["disjoint", "touches", "intersects", "contains", "within", "equals"],
    "point-polyhedron":      ["disjoint", "inside"],
    "polyhedron-point":      ["disjoint", "contains"],
    "polyline-polyhedron":   ["disjoint", "touches", "within", "intersects"],
    "polyline-polyline":     ["disjoint", "intersects"],
    "point-point":           ["disjoint", "equals"],
}


def _topo_label_to_int(label: str, pair_type: str) -> int:
    labels = TOPO_LABELS.get(pair_type)
    if labels is None:
        raise ValueError(f"Unknown pair_type: {pair_type}")
    if label not in labels:
        raise ValueError(f"Label '{label}' not in {labels} for {pair_type}")
    return labels.index(label)


# ---------------------------------------------------------------------------
# Polyhedron triangulation
# ---------------------------------------------------------------------------

def triangulate_prism(footprint: np.ndarray, z_bot: float, z_top: float) -> np.ndarray:
    """
    Triangulate an extruded prism into a closed triangular mesh.

    Args:
        footprint: (K, 2) array of the exterior ring vertices (not closed)
        z_bot:     bottom z coordinate
        z_top:     top z coordinate

    Returns:
        faces: (F, 3, 3) float32 array where each face is three (x,y,z) vertices.
               F = (K-2) + (K-2) + 2*(K)  =  2*K + 2*(K-2)  =  4K - 4  for K>=3
               (bottom fan + top fan + 2*K side triangles)
    """
    K = len(footprint)
    assert K >= 3

    bot = np.column_stack([footprint, np.full(K, z_bot)])   # (K, 3)
    top = np.column_stack([footprint, np.full(K, z_top)])   # (K, 3)

    faces = []

    # Bottom cap — fan from bot[0], winding inward (normal pointing -z)
    for i in range(1, K - 1):
        faces.append([bot[0], bot[i + 1], bot[i]])  # reversed winding → outward = -z

    # Top cap — fan from top[0], winding outward (+z)
    for i in range(1, K - 1):
        faces.append([top[0], top[i], top[i + 1]])

    # Side quads — two triangles per footprint edge
    for i in range(K):
        j = (i + 1) % K
        # quad: bot[i], bot[j], top[j], top[i]
        faces.append([bot[i], bot[j], top[j]])
        faces.append([bot[i], top[j], top[i]])

    return np.array(faces, dtype=np.float32)  # (F, 3, 3)


# ---------------------------------------------------------------------------
# JSON geom -> (array, length, face_count)
# ---------------------------------------------------------------------------

def geom_to_parts(geom: dict) -> Tuple[np.ndarray, int, int]:
    """
    Convert a JSON geom dict to (array, length, face_count).

    point:
        array  = (1, 3) float32
        length = 1,  face_count = 0

    polyline:
        array  = (N, 3) float32
        length = N,  face_count = 0

    polyhedron:
        array  = (F, 3, 3) float32   triangular faces
        length = 0,  face_count = F
    """
    gtype = geom["type"]

    if gtype == "point":
        arr = np.array(geom["coords"], dtype=np.float32).reshape(1, 3)
        return arr, 1, 0

    if gtype == "polyline":
        arr = np.array(geom["coords"], dtype=np.float32)   # (N, 3)
        return arr, len(arr), 0

    if gtype == "polyhedron":
        fp = np.array(geom["footprint"], dtype=np.float32)  # (K, 2)
        z_bot = float(geom["z_bot"])
        z_top = float(geom["z_top"])
        faces = triangulate_prism(fp, z_bot, z_top)         # (F, 3, 3)
        return faces, 0, len(faces)

    raise ValueError(f"Unknown geom type: {gtype}")


def geom_dataset_type(geom: dict) -> str:
    t = geom["type"]
    if t == "point":      return "points"
    if t == "polyline":   return "polylines"
    if t == "polyhedron": return "polyhedra"
    raise ValueError(f"Unknown geom type: {t}")


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def pad_geom_batch(
    arrays: List[np.ndarray],
    lengths: List[int],
    face_counts: List[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad a heterogeneous list of geometry arrays into a single tensor.

    For points/polylines: arrays are (N, 3)  → padded to (B, maxN, 3)
    For polyhedra:        arrays are (F, 3, 3) → padded to (B, maxF, 3, 3)

    Returns X, L, FC tensors.
    """
    if all(fc == 0 for fc in face_counts):
        # points or polylines — shape (N, 3)
        max_len = max(a.shape[0] for a in arrays)
        B = len(arrays)
        padded = np.zeros((B, max_len, 3), dtype=np.float32)
        for i, a in enumerate(arrays):
            padded[i, :a.shape[0]] = a
        X = torch.tensor(padded, dtype=torch.float32)
        L = torch.tensor(lengths, dtype=torch.int64)
        FC = torch.zeros(B, dtype=torch.int64)
        return X, L, FC
    else:
        # polyhedra — shape (F, 3, 3)
        max_faces = max(a.shape[0] for a in arrays)
        B = len(arrays)
        padded = np.zeros((B, max_faces, 3, 3), dtype=np.float32)
        for i, a in enumerate(arrays):
            padded[i, :a.shape[0]] = a
        X = torch.tensor(padded, dtype=torch.float32)
        L = torch.zeros(B, dtype=torch.int64)
        FC = torch.tensor(face_counts, dtype=torch.int64)
        return X, L, FC


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_records(jsonl_paths: List[str]) -> List[dict]:
    records = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    jsonl_paths: List[str],
    out_path: str,
    task: str,
    pair_type: str,
) -> None:
    records = load_records(jsonl_paths)
    print(f"Loaded {len(records)} records.")

    arrays1, lengths1, fc1 = [], [], []
    arrays2, lengths2, fc2 = [], [], []
    labels = []

    for r in records:
        a1, l1, f1 = geom_to_parts(r["geom1"])
        a2, l2, f2 = geom_to_parts(r["geom2"])
        arrays1.append(a1); lengths1.append(l1); fc1.append(f1)
        arrays2.append(a2); lengths2.append(l2); fc2.append(f2)

        if task == "distance":
            labels.append(float(r["label"]))
        elif task == "directional":
            labels.append(int(r["label_id"]))
        elif task == "topological":
            labels.append(_topo_label_to_int(r["label"], pair_type))
        else:
            raise ValueError(f"Unknown task: {task}")

    X1, L1, FC1 = pad_geom_batch(arrays1, lengths1, fc1)
    X2, L2, FC2 = pad_geom_batch(arrays2, lengths2, fc2)

    if task == "distance":
        Y = torch.tensor(labels, dtype=torch.float32)
    else:
        Y = torch.tensor(labels, dtype=torch.int64)

    print(f"X1: {X1.shape}  X2: {X2.shape}")
    print(f"L1: {L1.shape}  L2: {L2.shape}")
    print(f"FC1: {FC1.shape} (max={FC1.max()})  FC2: {FC2.shape} (max={FC2.max()})")
    print(f"Y: {Y.shape}  dtype={Y.dtype}")
    if task != "distance":
        vals, cnts = torch.unique(Y, return_counts=True)
        print(f"Label distribution: { {int(v): int(c) for v, c in zip(vals, cnts)} }")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save((X1, X2, L1, L2, FC1, FC2, Y), out_path)
    print(f"Saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert 3D JSONL to .pt training file")
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--task", required=True, choices=["distance", "directional", "topological"])
    parser.add_argument("--pair_type", required=True)
    args = parser.parse_args()
    convert(args.jsonl, args.out, args.task, args.pair_type)


if __name__ == "__main__":
    main()
