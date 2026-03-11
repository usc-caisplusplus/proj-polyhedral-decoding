"""
Normalization modes for geometries.
Used as a post-processing step on generated pairs; compatible with moment encoding.
"""

from __future__ import annotations

import math
from typing import Tuple, List, Union

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, scale
from shapely.geometry.base import BaseGeometry


def _get_bounds(geom: BaseGeometry) -> Tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return minx, miny, maxx, maxy


def _scale_geom(geom: BaseGeometry, sx: float, sy: float, origin: Tuple[float, float]) -> BaseGeometry:
    return scale(geom, xfact=sx, yfact=sy, origin=origin)


def _translate_geom(geom: BaseGeometry, dx: float, dy: float) -> BaseGeometry:
    return translate(geom, xoff=dx, yoff=dy)


def normalize_raw(geom: BaseGeometry) -> BaseGeometry:
    """No change; return as-is (coordinates already in [-1, 1] from generator)."""
    return geom


def normalize_bbox(geom: BaseGeometry) -> BaseGeometry:
    """Scale and translate so bounding box is [0,0]-[1,1] (or unit scale)."""
    minx, miny, maxx, maxy = _get_bounds(geom)
    w = maxx - minx
    h = maxy - miny
    if w < 1e-10:
        w = 1e-10
    if h < 1e-10:
        h = 1e-10
    # center of bbox
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    # scale so max side = 1, then center at (0.5, 0.5)
    s = 1.0 / max(w, h)
    g = _scale_geom(geom, s, s, (cx, cy))
    g = _translate_geom(g, 0.5 - s * cx, 0.5 - s * cy)
    return g


def _area_or_length(geom: BaseGeometry) -> float:
    if isinstance(geom, Point):
        return 1.0
    if isinstance(geom, LineString):
        return geom.length
    if hasattr(geom, "area"):
        return geom.area
    return 1.0


def normalize_m00(geom: BaseGeometry) -> BaseGeometry:
    """
    Scale so that M00 (area for polygon, length for line, 1 for point) is 1.
    Centroid is moved to origin.
    """
    if isinstance(geom, Point):
        return geom  # already point
    m00 = _area_or_length(geom)
    if m00 < 1e-12:
        return geom
    cx = geom.centroid.x
    cy = geom.centroid.y
    s = 1.0 / math.sqrt(m00)
    g = _translate_geom(geom, -cx, -cy)
    g = _scale_geom(g, s, s, (0, 0))
    return g


def normalize_centroid(geom: BaseGeometry) -> BaseGeometry:
    """Translate so centroid is at (0, 0); no scaling."""
    cx = geom.centroid.x
    cy = geom.centroid.y
    return _translate_geom(geom, -cx, -cy)


def normalize_pair(
    geom1: BaseGeometry,
    geom2: BaseGeometry,
    mode: str,
) -> Tuple[BaseGeometry, BaseGeometry]:
    """
    Normalize a pair so they share the same coordinate frame.
    - raw: no change.
    - bbox: fit both in a common bbox (union of bounds) then scale to [0,1]^2.
    - m00: each geometry normalized by its own M00 and centroid (independent).
    - centroid: translate so geom1's centroid is at origin; geom2 moved by same vector.
    """
    if mode == "raw":
        return geom1, geom2
    if mode == "m00":
        return normalize_m00(geom1), normalize_m00(geom2)
    if mode == "centroid":
        cx, cy = geom1.centroid.x, geom1.centroid.y
        g1 = _translate_geom(geom1, -cx, -cy)
        g2 = _translate_geom(geom2, -cx, -cy)
        return g1, g2
    if mode == "bbox":
        # union bounds of both
        minx1, miny1, maxx1, maxy1 = _get_bounds(geom1)
        minx2, miny2, maxx2, maxy2 = _get_bounds(geom2)
        minx = min(minx1, minx2)
        miny = min(miny1, miny2)
        maxx = max(maxx1, maxx2)
        maxy = max(maxy1, maxy2)
        w = maxx - minx
        h = maxy - miny
        if w < 1e-10:
            w = 1e-10
        if h < 1e-10:
            h = 1e-10
        s = 1.0 / max(w, h)
        # scale and translate both by same transform so relative pose preserved
        g1 = _translate_geom(geom1, -minx, -miny)
        g1 = _scale_geom(g1, s, s, (0, 0))
        g2 = _translate_geom(geom2, -minx, -miny)
        g2 = _scale_geom(g2, s, s, (0, 0))
        return g1, g2
    raise ValueError(f"Unknown normalization mode: {mode}")


