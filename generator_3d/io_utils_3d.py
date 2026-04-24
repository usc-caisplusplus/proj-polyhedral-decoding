"""
I/O utilities for 3D geoms: serialization to JSON dicts and JSONL writes.

JSON shapes (aligned with 2D io_utils.to_json_geom where it makes sense):
  point:       {"type": "point", "coords": [x, y, z]}
  polyline:    {"type": "polyline", "coords": [[x, y, z], ...]}
  polyhedron:  {"type": "polyhedron",
                "footprint": [[x, y], ...],      # exterior ring (not closed)
                "holes":     [[[x, y], ...], ...],
                "z_bot": z0, "z_top": z1}
"""

from __future__ import annotations

import json
from typing import Dict, List, Any

import numpy as np

from .geometry_generators_3d import Geom3D, Polyhedron3D


def to_json_geom_3d(geom: Geom3D) -> Dict[str, Any]:
    if isinstance(geom, np.ndarray):
        if geom.ndim == 1:
            return {"type": "point", "coords": [float(v) for v in geom]}
        return {
            "type": "polyline",
            "coords": [[float(p[0]), float(p[1]), float(p[2])] for p in geom],
        }
    if isinstance(geom, Polyhedron3D):
        ext = [[float(x), float(y)] for x, y in geom.footprint.exterior.coords]
        # Drop the closing duplicate vertex to match 2D convention.
        if len(ext) > 1 and ext[0] == ext[-1]:
            ext = ext[:-1]
        holes = [
            [[float(x), float(y)] for x, y in ring.coords]
            for ring in geom.footprint.interiors
        ]
        return {
            "type": "polyhedron",
            "footprint": ext,
            "holes": holes,
            "z_bot": float(geom.z_bot),
            "z_top": float(geom.z_top),
        }
    raise ValueError(f"Unsupported 3D geometry for serialization: {type(geom)}")


def geom_meta_3d(geom: Geom3D) -> Dict[str, Any]:
    if isinstance(geom, np.ndarray):
        if geom.ndim == 1:
            return {"vertices": 1, "has_hole": False}
        return {"vertices": int(geom.shape[0]), "has_hole": False}
    if isinstance(geom, Polyhedron3D):
        # footprint.exterior.coords includes the closing duplicate vertex.
        return {
            "vertices": max(len(geom.footprint.exterior.coords) - 1, 0),
            "has_hole": len(list(geom.footprint.interiors)) > 0,
        }
    return {"vertices": 0, "has_hole": False}


def write_jsonl_3d(path: str, records: List[Dict[str, Any]], encoding: str = "utf-8") -> None:
    with open(path, "w", encoding=encoding) as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
