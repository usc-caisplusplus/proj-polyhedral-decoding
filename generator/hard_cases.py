"""
Hard case generation: perturb or construct pairs that are difficult to classify.
- very close but disjoint
- almost contained
- thin polygons
- near boundary / self-near boundaries
"""

from __future__ import annotations

import random
from typing import Tuple, Any

from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.affinity import translate, scale
from shapely.ops import nearest_points

from . import geometry_generators as gen


def _ensure_valid(g: Any) -> Any:
    """Make geometry valid (buffer(0) for polygon-like) to avoid TopologyException."""
    if g is None or (hasattr(g, "is_empty") and g.is_empty):
        return g
    if hasattr(g, "buffer"):
        try:
            g = g.buffer(0)
        except Exception:
            pass
    if hasattr(g, "geoms") and hasattr(g, "area"):
        g = max(g.geoms, key=lambda x: x.area)
    return g


def _ensure_polygon(g):
    if hasattr(g, "buffer"):
        g = g.buffer(0)
    if hasattr(g, "geoms"):
        g = max(g.geoms, key=lambda x: x.area)
    return g


def make_near_disjoint(
    rng: random.Random,
    g1: Any,
    g2: Any,
    gap_scale: float = 0.02,
) -> Tuple[Any, Any]:
    """
    Move g2 so it is barely disjoint from g1 (very small gap).
    """
    g1 = _ensure_valid(g1)
    g2 = _ensure_valid(g2)
    try:
        if not g1.intersects(g2):
            return g1, g2
    except Exception:
        return g1, g2
    try:
        p1, p2 = nearest_points(g1, g2)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        d = (dx * dx + dy * dy) ** 0.5
        if d < 1e-10:
            dx, dy = gap_scale, 0.0
        else:
            dx = dx / d * gap_scale
            dy = dy / d * gap_scale
        g2 = translate(g2, dx, dy)
    except Exception:
        pass
    return g1, g2


def make_almost_contained(
    rng: random.Random,
    g_outer: Any,
    g_inner: Any,
    nudge_scale: float = 0.05,
) -> Tuple[Any, Any]:
    """
    g_inner almost fully inside g_outer but slightly crossing boundary.
    Assumes g_outer contains g_inner or they overlap; nudge g_inner so it's almost contained.
    """
    if not g_outer.contains(g_inner):
        # scale down g_inner and place near centroid of g_outer
        cx, cy = g_outer.centroid.x, g_outer.centroid.y
        g_inner = scale(g_inner, xfact=0.3, yfact=0.3, origin=(g_inner.centroid.x, g_inner.centroid.y))
        g_inner = translate(g_inner, cx - g_inner.centroid.x, cy - g_inner.centroid.y)
    # nudge toward boundary so "almost" contained
    bx, by = g_outer.centroid.x, g_outer.centroid.y
    ix, iy = g_inner.centroid.x, g_inner.centroid.y
    dx = (bx - ix) * nudge_scale
    dy = (by - iy) * nudge_scale
    g_inner = translate(g_inner, -dx, -dy)
    return g_outer, g_inner


def make_thin_polygon(
    rng: random.Random,
    n_vertices: int = 8,
    aspect: float = 0.08,
) -> Polygon:
    """
    Long thin polygon (high aspect ratio).
    """
    pts = []
    for i in range(n_vertices):
        t = i / max(1, n_vertices - 1)
        x = (t - 0.5) * 0.4
        y = rng.uniform(-aspect, aspect)
        pts.append((x, y))
    poly = Polygon(pts).buffer(0)
    if poly.is_empty or not poly.is_valid:
        return gen.random_star_polygon(rng, n_vertices, True, False)
    if hasattr(poly, "geoms"):
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def make_near_boundary_case(
    rng: random.Random,
    geom: Any,
    box_min: float = -1.0,
    box_max: float = 1.0,
    margin: float = 0.05,
) -> Any:
    """
    Translate geometry so it sits close to the bounding box boundary (e.g. [-1,1]^2).
    """
    side = rng.randint(0, 3)  # 0=left, 1=right, 2=bottom, 3=top
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    if side == 0:
        dx = box_min + margin - minx
        dy = 0
    elif side == 1:
        dx = box_max - margin - maxx
        dy = 0
    elif side == 2:
        dx = 0
        dy = box_min + margin - miny
    else:
        dx = 0
        dy = box_max - margin - maxy
    return translate(geom, dx, dy)


def make_thin_polyline(
    rng: random.Random,
    n_vertices: int = 15,
    curvature: float = 0.3,
) -> LineString:
    """Long, winding polyline."""
    return gen.random_polyline(rng, n_vertices, step_min=curvature * 0.1, step_max=curvature * 0.25, max_extent=0.45)


def apply_hard_case(
    rng: random.Random,
    g1: Any,
    g2: Any,
    mode: str,
    pair_type: str,
) -> Tuple[Any, Any]:
    """
    Apply one hard-case transformation to the pair (g1, g2).
    mode: none, near_disjoint, almost_contained, thin_polygon, near_boundary, mixed
    """
    if mode == "none":
        return g1, g2

    if mode == "near_disjoint":
        return make_near_disjoint(rng, g1, g2)

    if mode == "almost_contained":
        # only makes sense for polygon-polygon or polyline-polygon
        if "polygon" in pair_type and "polygon" in pair_type:
            if g1.area >= g2.area:
                return make_almost_contained(rng, g1, g2)
            return make_almost_contained(rng, g2, g1)
        if pair_type == "polyline-polygon":
            return make_almost_contained(rng, g2, g1)
        return g1, g2

    if mode == "thin_polygon":
        # replace one polygon with thin one
        type1, type2 = pair_type.split("-")
        if type1 == "polygon":
            g1 = make_thin_polygon(rng)
            g1 = gen.random_translate_inside(rng, g1)
        if type2 == "polygon":
            g2 = make_thin_polygon(rng)
            g2 = gen.random_translate_inside(rng, g2)
        return g1, g2

    if mode == "near_boundary":
        g1 = make_near_boundary_case(rng, g1)
        g2 = make_near_boundary_case(rng, g2)
        return g1, g2

    if mode == "mixed":
        choice = rng.choice(["near_disjoint", "almost_contained", "thin_polygon", "near_boundary"])
        return apply_hard_case(rng, g1, g2, choice, pair_type)

    return g1, g2
