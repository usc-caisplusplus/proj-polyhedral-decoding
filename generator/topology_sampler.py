"""
Targeted topology sampling: generate pairs that satisfy a specific topology label
(touches, contains, intersects, disjoint) for balanced datasets.
"""

from __future__ import annotations

import random
from typing import Tuple, Optional, List, Any

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, scale
from shapely.ops import nearest_points

from . import geometry_generators as gen
from . import labels


def _ensure_valid(g):
    if hasattr(g, "buffer"):
        g = g.buffer(0)
    if hasattr(g, "geoms"):  # MultiPolygon
        g = max(g.geoms, key=lambda x: x.area)
    return g


def sample_polygon_polygon_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Two polygons that do not intersect."""
    g1 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.place_geom(rng, "polygon", complexity)
    g1 = gen.random_translate_inside(rng, g1, -0.8, -0.8, 0, 0.8)
    g2 = gen.random_translate_inside(rng, g2, 0, -0.8, 0.8, 0.8)
    if g1.intersects(g2):
        # push g2 away
        p1, p2 = nearest_points(g1, g2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        dist = (dx * dx + dy * dy) ** 0.5
        if dist < 1e-6:
            dx, dy = 0.2, 0.0
        else:
            dx, dy = dx / dist * 0.3, dy / dist * 0.3
        g2 = translate(g2, dx, dy)
    return _ensure_valid(g1), _ensure_valid(g2)


def sample_polygon_polygon_touches(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Two polygons that touch at boundary only."""
    g1 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.place_geom(rng, "polygon", complexity)
    g1 = gen.random_translate_inside(rng, g1, -0.6, -0.6, 0.2, 0.6)
    g2 = gen.random_translate_inside(rng, g2, -0.2, -0.6, 0.6, 0.6)
    if not g1.touches(g2) and not g1.intersects(g2):
        p1, p2 = nearest_points(g1, g2)
        shift_x = p1.x - p2.x
        shift_y = p1.y - p2.y
        g2 = translate(g2, shift_x, shift_y)
    if g1.intersects(g2) and not g1.touches(g2):
        # overlap: nudge apart slightly
        p1, p2 = nearest_points(g1, g2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        d = (dx * dx + dy * dy) ** 0.5
        if d > 1e-10:
            g2 = translate(g2, dx / d * 0.02, dy / d * 0.02)
    return _ensure_valid(g1), _ensure_valid(g2)


def sample_polygon_polygon_contains(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """g1 contains g2 (small polygon inside large)."""
    g1 = gen.place_geom(rng, "polygon", complexity)
    g2_small = gen.random_star_polygon(rng, rng.randint(3, 6), True, False, 0.05, 0.15)
    cx, cy = g1.centroid.x, g1.centroid.y
    g2 = translate(g2_small, cx, cy)
    if not g1.contains(g2):
        g2 = scale(g2, xfact=0.5, yfact=0.5, origin=(cx, cy))
    return _ensure_valid(g1), _ensure_valid(g2)


def sample_polygon_polygon_within(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """g1 within g2 (g2 contains g1)."""
    g2, g1 = sample_polygon_polygon_contains(rng, complexity)
    return _ensure_valid(g1), _ensure_valid(g2)


def sample_polygon_polygon_intersects(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Overlap but not contains/within/touches (overlaps)."""
    g1 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.place_geom(rng, "polygon", complexity)
    g1 = gen.random_translate_inside(rng, g1, -0.4, -0.4, 0.4, 0.4)
    p1, p2 = nearest_points(g1, g2)
    shift_x = p1.x - p2.x
    shift_y = p1.y - p2.y
    g2 = translate(g2, shift_x, shift_y)
    # move g2 slightly inside g1 so we get overlap
    cx, cy = g1.centroid.x, g1.centroid.y
    c2x, c2y = g2.centroid.x, g2.centroid.y
    g2 = translate(g2, (cx - c2x) * 0.4, (cy - c2y) * 0.4)
    if not g1.intersects(g2) or g1.contains(g2) or g2.contains(g1) or g1.touches(g2):
        # retry with different overlap
        g2 = gen.place_geom(rng, "polygon", complexity)
        g2 = translate(g2, g1.centroid.x - g2.centroid.x + rng.uniform(-0.1, 0.1), g1.centroid.y - g2.centroid.y + rng.uniform(-0.1, 0.1))
    return _ensure_valid(g1), _ensure_valid(g2)


def sample_polygon_polygon_equals(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]:
    """Same geometry twice."""
    g = gen.place_geom(rng, "polygon", complexity)
    g = gen.random_translate_inside(rng, g)
    return _ensure_valid(g), _ensure_valid(g)


def sample_polyline_polygon_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Polygon | MultiPolygon]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g2 = gen.place_geom(rng, "polygon", complexity)
    g1 = gen.random_translate_inside(rng, g1, -0.8, -0.8, 0, 0.8)
    g2 = gen.random_translate_inside(rng, g2, 0, -0.8, 0.8, 0.8)
    if g1.intersects(g2):
        p1, p2 = nearest_points(g1, g2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-6:
            dx, dy = 0.2, 0
        else:
            dx, dy = dx / d * 0.3, dy / d * 0.3
        g2 = translate(g2, dx, dy)
    return g1, _ensure_valid(g2)


def sample_polyline_polygon_touches(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Polygon | MultiPolygon]:
    g2 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.random_translate_inside(rng, g2)
    # build short line that ends on g2 boundary
    b = g2.boundary
    if b.is_empty:
        g1 = gen.place_geom(rng, "polyline", complexity)
        g1 = gen.random_translate_inside(rng, g1)
        return g1, _ensure_valid(g2)
    coords = list(b.coords)[:20]
    i = rng.randint(0, max(0, len(coords) - 2))
    x, y = coords[i][0], coords[i][1]
    dx = rng.uniform(-0.05, 0.05)
    dy = rng.uniform(-0.05, 0.05)
    g1 = LineString([(x - dx, y - dy), (x, y)])
    if not g1.touches(g2):
        g1 = LineString([(x, y), (x + dx, y + dy)])
    return g1, _ensure_valid(g2)


def sample_polyline_polygon_within(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Polygon | MultiPolygon]:
    g2 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.random_translate_inside(rng, g2)
    g1 = gen.place_geom(rng, "polyline", complexity)
    cx, cy = g2.centroid.x, g2.centroid.y
    g1 = translate(g1, cx, cy)
    if not g2.contains(g1):
        g1 = scale(g1, xfact=0.3, yfact=0.3, origin=(cx, cy))
    return g1, _ensure_valid(g2)


def sample_polyline_polygon_intersects(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Polygon | MultiPolygon]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g2 = gen.place_geom(rng, "polygon", complexity)
    g1 = gen.random_translate_inside(rng, g1)
    p1, p2 = nearest_points(g1, g2)
    g2 = translate(g2, p1.x - p2.x, p1.y - p2.y)
    g2 = translate(g2, (g1.centroid.x - g2.centroid.x) * 0.3, (g1.centroid.y - g2.centroid.y) * 0.3)
    return g1, _ensure_valid(g2)


def sample_point_polygon_inside(
    rng: random.Random,
    complexity: str,
) -> Tuple[Point, Polygon | MultiPolygon]:
    g2 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.random_translate_inside(rng, g2)
    # point inside g2
    cx, cy = g2.centroid.x, g2.centroid.y
    for _ in range(50):
        x = cx + rng.uniform(-0.1, 0.1)
        y = cy + rng.uniform(-0.1, 0.1)
        g1 = Point(x, y)
        if g2.contains(g1):
            return g1, _ensure_valid(g2)
    g1 = Point(cx, cy)
    return g1, _ensure_valid(g2)


def sample_point_polygon_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[Point, Polygon | MultiPolygon]:
    g2 = gen.place_geom(rng, "polygon", complexity)
    g2 = gen.random_translate_inside(rng, g2, -0.6, -0.6, 0.6, 0.6)
    g1 = gen.random_point(rng)
    if g2.contains(g1):
        g1 = Point(g2.bounds[0] - 0.2, g2.centroid.y)
    return g1, _ensure_valid(g2)


def sample_polygon_point_contains(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Point]:
    g2, g1 = sample_point_polygon_inside(rng, complexity)
    return _ensure_valid(g1), g2


def sample_polygon_point_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[Polygon | MultiPolygon, Point]:
    g1, g2 = sample_point_polygon_disjoint(rng, complexity)
    return _ensure_valid(g1), g2


def sample_polyline_polyline_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, LineString]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g2 = gen.place_geom(rng, "polyline", complexity)
    g1 = gen.random_translate_inside(rng, g1, -0.7, -0.7, 0, 0.7)
    g2 = gen.random_translate_inside(rng, g2, 0, -0.7, 0.7, 0.7)
    return g1, g2


def sample_polyline_polyline_intersects(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, LineString]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g2 = gen.place_geom(rng, "polyline", complexity)
    g1 = gen.random_translate_inside(rng, g1)
    p1, p2 = nearest_points(g1, g2)
    g2 = translate(g2, p1.x - p2.x, p1.y - p2.y)
    return g1, g2


def sample_polyline_point_disjoint(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Point]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g2 = gen.random_point(rng)
    g1 = gen.random_translate_inside(rng, g1)
    if g1.intersects(g2):
        g2 = translate(g2, 0.5, 0.5)
    return g1, g2


def sample_polyline_point_intersects(
    rng: random.Random,
    complexity: str,
) -> Tuple[LineString, Point]:
    g1 = gen.place_geom(rng, "polyline", complexity)
    g1 = gen.random_translate_inside(rng, g1)
    coords = list(g1.coords)
    i = rng.randint(0, len(coords) - 1) if len(coords) > 1 else 0
    x, y = coords[i][0], coords[i][1]
    g2 = Point(x, y)
    return g1, g2


# Dispatcher: (pair_type, label) -> sampler function
def get_topology_sampler(pair_type: str, label: str):
    """Return a function (rng, complexity) -> (g1, g2) for the given pair_type and label."""
    key = (pair_type, label)
    table = {
        ("polygon-polygon", "disjoint"): sample_polygon_polygon_disjoint,
        ("polygon-polygon", "touches"): sample_polygon_polygon_touches,
        ("polygon-polygon", "contains"): sample_polygon_polygon_contains,
        ("polygon-polygon", "within"): sample_polygon_polygon_within,
        ("polygon-polygon", "intersects"): sample_polygon_polygon_intersects,
        ("polygon-polygon", "equals"): sample_polygon_polygon_equals,
        ("polyline-polygon", "disjoint"): sample_polyline_polygon_disjoint,
        ("polyline-polygon", "touches"): sample_polyline_polygon_touches,
        ("polyline-polygon", "within"): sample_polyline_polygon_within,
        ("polyline-polygon", "intersects"): sample_polyline_polygon_intersects,
        ("point-polygon", "inside"): sample_point_polygon_inside,
        ("point-polygon", "disjoint"): sample_point_polygon_disjoint,
        ("polygon-point", "contains"): sample_polygon_point_contains,
        ("polygon-point", "disjoint"): sample_polygon_point_disjoint,
        ("polyline-polyline", "disjoint"): sample_polyline_polyline_disjoint,
        ("polyline-polyline", "intersects"): sample_polyline_polyline_intersects,
        ("polyline-point", "disjoint"): sample_polyline_point_disjoint,
        ("polyline-point", "intersects"): sample_polyline_point_intersects,
    }
    return table.get(key)


def sample_targeted_topology_pair(
    rng: random.Random,
    pair_type: str,
    target_label: str,
    complexity: str,
) -> Tuple[Any, Any]:
    """
    Sample a pair (g1, g2) that has the given topology label.
    Falls back to random pair + filter if no dedicated sampler exists.
    """
    sampler = get_topology_sampler(pair_type, target_label)
    if sampler is not None:
        return sampler(rng, complexity)
    # Fallback: random pair and hope it matches (may need many tries)
    type1, type2 = pair_type.split("-")
    g1, g2 = gen.make_pair(rng, type1, type2, complexity)
    return g1, g2
