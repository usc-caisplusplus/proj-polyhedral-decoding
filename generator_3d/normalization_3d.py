"""
3D normalization modes (lifted from 2D generator).

  - raw:       no change
  - bbox:      fit both geoms in a unit cube via shared affine (preserves relative pose)
  - centroid:  translate so g1's centroid is at the origin; g2 moved by same vector

`m00` (volume-normalized) is omitted from the MVP because polyhedra need a real
3D volume computation; easy to add once the moment machinery is in place.
"""

from __future__ import annotations

from typing import Tuple

from .geometry_generators_3d import (
    Geom3D,
    bounds_3d,
    translate_3d,
    scale_3d,
)


def _normalize_raw(g1: Geom3D, g2: Geom3D) -> Tuple[Geom3D, Geom3D]:
    return g1, g2


def _normalize_bbox(g1: Geom3D, g2: Geom3D) -> Tuple[Geom3D, Geom3D]:
    b1 = bounds_3d(g1)
    b2 = bounds_3d(g2)
    minx = min(b1[0], b2[0]); miny = min(b1[1], b2[1]); minz = min(b1[2], b2[2])
    maxx = max(b1[3], b2[3]); maxy = max(b1[4], b2[4]); maxz = max(b1[5], b2[5])
    w = max(maxx - minx, 1e-10)
    h = max(maxy - miny, 1e-10)
    d = max(maxz - minz, 1e-10)
    s = 1.0 / max(w, h, d)
    g1t = translate_3d(g1, -minx, -miny, -minz)
    g1s = scale_3d(g1t, s, s, s, (0.0, 0.0, 0.0))
    g2t = translate_3d(g2, -minx, -miny, -minz)
    g2s = scale_3d(g2t, s, s, s, (0.0, 0.0, 0.0))
    return g1s, g2s


def _normalize_centroid(g1: Geom3D, g2: Geom3D) -> Tuple[Geom3D, Geom3D]:
    # Import locally to avoid a generator_3d -> labels_3d -> generator_3d cycle at import time.
    from .labels_3d import representative_xyz
    c = representative_xyz(g1)
    return (
        translate_3d(g1, -float(c[0]), -float(c[1]), -float(c[2])),
        translate_3d(g2, -float(c[0]), -float(c[1]), -float(c[2])),
    )


def normalize_pair_3d(g1: Geom3D, g2: Geom3D, mode: str) -> Tuple[Geom3D, Geom3D]:
    if mode == "raw":
        return _normalize_raw(g1, g2)
    if mode == "bbox":
        return _normalize_bbox(g1, g2)
    if mode == "centroid":
        return _normalize_centroid(g1, g2)
    raise ValueError(f"Unknown 3D normalization mode: {mode}")
