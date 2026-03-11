"""
Geometry generators for the unified spatial dataset.
Generates: point, polyline, polygon, multipolygon with complexity control.
"""

from __future__ import annotations

import math
import random
from typing import Any, Tuple

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, scale
from shapely.ops import unary_union


def random_point(rng: random.Random) -> Point:
    return Point(rng.uniform(-1, 1), rng.uniform(-1, 1))


def random_polyline(
    rng: random.Random,
    n_vertices: int,
    step_min: float = 0.05,
    step_max: float = 0.2,
    max_extent: float = 0.45,
) -> LineString:
    """Random walk polyline, recentered and scaled to fit in [-max_extent, max_extent]."""
    x, y = 0.0, 0.0
    pts = [(x, y)]
    for _ in range(n_vertices - 1):
        angle = rng.uniform(0, 2 * math.pi)
        step = rng.uniform(step_min, step_max)
        x += step * math.cos(angle)
        y += step * math.sin(angle)
        pts.append((x, y))

    mx = sum(px for px, _ in pts) / len(pts)
    my = sum(py for _, py in pts) / len(pts)
    pts = [(px - mx, py - my) for px, py in pts]

    line = LineString(pts)
    coords = list(line.coords)
    max_abs = max(max(abs(x), abs(y)) for x, y in coords)
    if max_abs > max_extent and max_abs > 0:
        s = max_extent / max_abs
        line = scale(line, xfact=s, yfact=s, origin=(0, 0))
    return line


def random_star_polygon(
    rng: random.Random,
    n_vertices: int,
    nonconvex: bool = True,
    with_hole: bool = False,
    radius_min: float = 0.08,
    radius_max: float = 0.35,
) -> Polygon:
    """Star-like polygon; nonconvex varies radii, with_hole adds interior ring."""
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n_vertices))

    if nonconvex:
        radii = [rng.uniform(radius_min, radius_max) for _ in range(n_vertices)]
    else:
        base = rng.uniform(0.15, 0.3)
        radii = [rng.uniform(base * 0.85, base * 1.15) for _ in range(n_vertices)]

    shell = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
    poly = Polygon(shell).buffer(0)

    if poly.is_empty or not poly.is_valid or poly.area < 1e-4:
        return random_star_polygon(rng, n_vertices, nonconvex, with_hole)

    if hasattr(poly, "geoms"):  # MultiPolygon
        poly = max(poly.geoms, key=lambda g: g.area)

    if with_hole and poly.area > 0.05:
        cx, cy = poly.centroid.x, poly.centroid.y
        hole_scale = rng.uniform(0.15, 0.35)
        hole = scale(poly, xfact=hole_scale, yfact=hole_scale, origin=(cx, cy))
        if hasattr(hole, "exterior"):
            if poly.contains(hole):
                poly = Polygon(
                    poly.exterior.coords,
                    [hole.exterior.coords],
                ).buffer(0)
        if poly.is_empty or not poly.is_valid:
            poly = Polygon(shell).buffer(0)

    return poly


def make_pair(
    rng: random.Random,
    type1: str,
    type2: str,
    complexity: str,
) -> Tuple[Point | LineString | Polygon | MultiPolygon, Point | LineString | Polygon | MultiPolygon]:
    """Create a random pair of geometries, each translated inside [-1,1]^2."""
    g1 = random_translate_inside(rng, place_geom(rng, type1, complexity))
    g2 = random_translate_inside(rng, place_geom(rng, type2, complexity))
    return g1, g2


def random_multipolygon(rng: random.Random) -> MultiPolygon:
    """Two disjoint star polygons as MultiPolygon."""
    p1 = translate(
        random_star_polygon(rng, rng.randint(5, 10), True, False),
        -0.3, 0.0,
    )
    p2 = translate(
        random_star_polygon(rng, rng.randint(5, 10), True, False),
        0.3, 0.0,
    )
    mp = unary_union([p1, p2])
    if hasattr(mp, "exterior"):  # single Polygon
        mp = MultiPolygon([p1, p2])
    return mp


def complexity_params(complexity: str, rng: random.Random) -> dict[str, Any]:
    """Return vertex counts and probabilities for the given complexity."""
    if complexity == "low":
        return {
            "pl_n": rng.randint(2, 4),
            "pg_n": rng.randint(3, 6),
            "hole_prob": 0.0,
            "multi_prob": 0.0,
        }
    if complexity == "mid":
        return {
            "pl_n": rng.randint(5, 8),
            "pg_n": rng.randint(7, 12),
            "hole_prob": 0.2,
            "multi_prob": 0.1,
        }
    # high
    return {
        "pl_n": rng.randint(9, 18),
        "pg_n": rng.randint(13, 24),
        "hole_prob": 0.4,
        "multi_prob": 0.2,
    }


def place_geom(
    rng: random.Random,
    gtype: str,
    complexity: str,
) -> Point | LineString | Polygon | MultiPolygon:
    """Create one geometry of type gtype with given complexity."""
    params = complexity_params(complexity, rng)

    if gtype == "point":
        return random_point(rng)
    if gtype == "polyline":
        return random_polyline(rng, params["pl_n"])
    if gtype == "polygon":
        if rng.random() < params["multi_prob"]:
            return random_multipolygon(rng)
        return random_star_polygon(
            rng,
            params["pg_n"],
            nonconvex=True,
            with_hole=(rng.random() < params["hole_prob"]),
        )
    raise ValueError(f"Unknown gtype: {gtype}")


def random_translate_inside(
    rng: random.Random,
    geom: Point | LineString | Polygon | MultiPolygon,
    x_min: float = -1.0,
    y_min: float = -1.0,
    x_max: float = 1.0,
    y_max: float = 1.0,
    max_attempts: int = 50,
) -> Point | LineString | Polygon | MultiPolygon:
    """Translate geometry so its bounds stay inside the given box."""
    for _ in range(max_attempts):
        dx = rng.uniform(x_min + 0.3, x_max - 0.3)
        dy = rng.uniform(y_min + 0.3, y_max - 0.3)
        g = translate(geom, dx, dy)
        minx, miny, maxx, maxy = g.bounds
        if minx >= x_min and miny >= y_min and maxx <= x_max and maxy <= y_max:
            return g
    return geom
