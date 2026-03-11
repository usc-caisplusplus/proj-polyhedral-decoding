"""
Label generation for directional, distance, and topological tasks.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional, Any

from shapely.geometry.base import BaseGeometry


DIR16_LABELS = [
    "E", "ENE", "NE", "NNE",
    "N", "NNW", "NW", "WNW",
    "W", "WSW", "SW", "SSW",
    "S", "SSE", "SE", "ESE",
]
BIN_W = 2.0 * math.pi / 16.0


def wrap_0_2pi(theta: float) -> float:
    t = theta % (2.0 * math.pi)
    return t if t >= 0 else t + 2.0 * math.pi


def angle_to_dir16(theta: float) -> Tuple[int, str]:
    t = wrap_0_2pi(theta)
    bid = int((t + BIN_W / 2.0) // BIN_W) % 16
    return bid, DIR16_LABELS[bid]


def representative_xy(geom: BaseGeometry) -> Tuple[float, float]:
    p = geom.representative_point()
    return (p.x, p.y)


def directional_label(g1: BaseGeometry, g2: BaseGeometry) -> Tuple[int, str]:
    x1, y1 = representative_xy(g1)
    x2, y2 = representative_xy(g2)
    theta = math.atan2(y2 - y1, x2 - x1)
    return angle_to_dir16(theta)


def distance_label(g1: BaseGeometry, g2: BaseGeometry) -> float:
    return float(g1.distance(g2))


# Topology labels supported per pair_type (subset of DE-9IM / named relations)
TOPOLOGY_LABELS_POLYGON_POLYGON = [
    "disjoint", "touches", "intersects", "contains", "within", "equals",
]
TOPOLOGY_LABELS_POLYLINE_POLYGON = ["disjoint", "touches", "within", "intersects"]
TOPOLOGY_LABELS_POINT_POLYGON = ["disjoint", "inside"]  # inside = contained by polygon
TOPOLOGY_LABELS_POLYGON_POINT = ["disjoint", "contains"]
TOPOLOGY_LABELS_POLYLINE_POINT = ["disjoint", "intersects"]
TOPOLOGY_LABELS_POLYLINE_POLYLINE = ["disjoint", "intersects"]


def topology_label(
    pair_type: str,
    g1: BaseGeometry,
    g2: BaseGeometry,
) -> Optional[str]:
    """Compute topology label for the given pair type."""
    if pair_type == "point-polygon":
        if g2.contains(g1):
            return "inside"
        return "disjoint"

    if pair_type == "polygon-point":
        if g1.contains(g2):
            return "contains"
        return "disjoint"

    if pair_type == "polyline-point":
        if g1.intersects(g2):
            return "intersects"
        return "disjoint"

    if pair_type == "polyline-polyline":
        if g1.intersects(g2):
            return "intersects"
        return "disjoint"

    if pair_type == "polyline-polygon":
        if g2.contains(g1):
            return "within"
        if g1.touches(g2):
            return "touches"
        if g1.intersects(g2):
            return "intersects"
        return "disjoint"

    if pair_type == "polygon-polygon":
        if g1.equals(g2):
            return "equals"
        if g1.contains(g2):
            return "contains"
        if g1.within(g2):
            return "within"
        if g1.touches(g2):
            return "touches"
        if g1.intersects(g2):
            return "intersects"
        return "disjoint"

    return None


def get_topology_labels_for_pair_type(pair_type: str) -> list[str]:
    """Return list of topology label names supported for this pair type."""
    m = {
        "polygon-polygon": TOPOLOGY_LABELS_POLYGON_POLYGON,
        "polyline-polygon": TOPOLOGY_LABELS_POLYLINE_POLYGON,
        "point-polygon": TOPOLOGY_LABELS_POINT_POLYGON,
        "polygon-point": TOPOLOGY_LABELS_POLYGON_POINT,
        "polyline-point": TOPOLOGY_LABELS_POLYLINE_POINT,
        "polyline-polyline": TOPOLOGY_LABELS_POLYLINE_POLYLINE,
    }
    return m.get(pair_type, [])
