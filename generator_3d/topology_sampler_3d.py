"""
Targeted topology sampling for 3D pairs.

Lifted from `generator/topology_sampler.py`. Uses the prism decomposition
(2D footprint × 1D z-range) to construct pairs with a specified 3D topology
label exactly — avoiding Poisson-rare topology classes that random placement
almost never produces.

All samplers return a (geom1, geom2) tuple where each geom is one of the
types in `geometry_generators_3d.Geom3D`.
"""

from __future__ import annotations

import math
import random
from typing import Tuple, Optional, Any

import numpy as np
from shapely.geometry import Point as ShpPoint
from shapely.affinity import translate as shp_translate, scale as shp_scale
from shapely.ops import nearest_points

from . import geometry_generators_3d as gen
from .geometry_generators_3d import Polyhedron3D, translate_3d, scale_3d, bounds_3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_into_unit_box(g: gen.Geom3D) -> gen.Geom3D:
    """Translate g so its bounds fit in [-1, 1]^3 (best effort)."""
    minx, miny, minz, maxx, maxy, maxz = bounds_3d(g)
    dx = dy = dz = 0.0
    if minx < -1.0:
        dx = -1.0 - minx
    elif maxx > 1.0:
        dx = 1.0 - maxx
    if miny < -1.0:
        dy = -1.0 - miny
    elif maxy > 1.0:
        dy = 1.0 - maxy
    if minz < -1.0:
        dz = -1.0 - minz
    elif maxz > 1.0:
        dz = 1.0 - maxz
    if dx or dy or dz:
        return translate_3d(g, dx, dy, dz)
    return g


def _random_small_polyhedron(rng: random.Random) -> Polyhedron3D:
    return gen.random_polyhedron(rng, rng.randint(3, 6), height_min=0.05, height_max=0.15)


# ---------------------------------------------------------------------------
# Polyhedron-polyhedron
# ---------------------------------------------------------------------------

def sample_polyhedron_polyhedron_disjoint(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """Two prisms with either disjoint footprints or disjoint z-ranges."""
    g1 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    # Shove g2 away from g1 horizontally: place g1 on the left, g2 on the right.
    g1 = translate_3d(g1, rng.uniform(-0.8, -0.2), rng.uniform(-0.6, 0.6), 0.0)
    g2 = translate_3d(g2, rng.uniform(0.2, 0.8), rng.uniform(-0.6, 0.6), 0.0)
    # If somehow their 2D footprints still intersect, push g2 further.
    if g1.footprint.intersects(g2.footprint):
        p1, p2 = nearest_points(g1.footprint, g2.footprint)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        d = math.hypot(dx, dy)
        step = 0.3 if d < 1e-9 else 0.3 / d
        g2 = translate_3d(g2, dx * step + 0.05, dy * step, 0.0)
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


def sample_polyhedron_polyhedron_equals(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """Two identical prisms (same footprint, same z-range)."""
    g = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g = gen.random_translate_inside(rng, g)
    # Return two independent copies so downstream mutation doesn't alias them.
    g_copy = Polyhedron3D(footprint=g.footprint, z_bot=g.z_bot, z_top=g.z_top)
    return g, g_copy


def sample_polyhedron_polyhedron_contains(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """g1 strictly contains g2 (footprint and z-range)."""
    g1 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"],
                                height_min=0.4, height_max=0.8)
    g1 = gen.random_translate_inside(rng, g1)
    # Build a small prism centered on g1 and shrink until contained.
    small_fp = gen.random_star_polygon_2d(rng, rng.randint(3, 5),
                                            nonconvex=False,
                                            radius_min=0.04, radius_max=0.10)
    cx, cy = g1.footprint.centroid.x, g1.footprint.centroid.y
    small_fp = shp_translate(small_fp, cx, cy)
    # shrink further if not inside
    for _ in range(6):
        if g1.footprint.contains(small_fp):
            break
        small_fp = shp_scale(small_fp, xfact=0.5, yfact=0.5, origin=(cx, cy))
    z_gap = 0.05
    z_bot = g1.z_bot + z_gap
    z_top = g1.z_top - z_gap
    if z_top <= z_bot:
        z_mid = 0.5 * (g1.z_bot + g1.z_top)
        z_bot = z_mid - 0.02
        z_top = z_mid + 0.02
    g2 = Polyhedron3D(footprint=small_fp, z_bot=z_bot, z_top=z_top)
    return g1, g2


def sample_polyhedron_polyhedron_within(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    g2, g1 = sample_polyhedron_polyhedron_contains(rng, complexity)
    return g1, g2


def sample_polyhedron_polyhedron_touches(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """Two prisms sharing a z-face exactly (top of g1 = bottom of g2) with footprint overlap."""
    g1 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g1 = gen.random_translate_inside(rng, g1)
    # g2 has the same footprint shape (translated slightly) and stacks on g1's top face.
    g2_fp = gen.random_star_polygon_2d(rng, rng.randint(3, 6), nonconvex=True)
    cx, cy = g1.footprint.centroid.x, g1.footprint.centroid.y
    g2_fp = shp_translate(g2_fp, cx, cy)
    height2 = rng.uniform(0.1, 0.4)
    g2 = Polyhedron3D(footprint=g2_fp, z_bot=g1.z_top, z_top=g1.z_top + height2)
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


def sample_polyhedron_polyhedron_intersects(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """Footprints overlap AND z-ranges overlap, but neither is contained in the other."""
    g1 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g1 = gen.random_translate_inside(rng, g1)
    # Offset g2 slightly from g1 so their footprints overlap but don't coincide.
    cx1, cy1 = g1.footprint.centroid.x, g1.footprint.centroid.y
    cx2, cy2 = g2.footprint.centroid.x, g2.footprint.centroid.y
    shift_x = (cx1 - cx2) + rng.uniform(-0.05, 0.05)
    shift_y = (cy1 - cy2) + rng.uniform(-0.05, 0.05)
    # z overlap: put g2's z-range overlapping but not enclosing/enclosed in g1's.
    z_shift = rng.uniform(0.1, 0.3) * rng.choice([-1.0, 1.0])
    g2 = translate_3d(g2, shift_x, shift_y, (g1.z_bot - g2.z_bot) + z_shift)
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


# ---------------------------------------------------------------------------
# Point / polyhedron
# ---------------------------------------------------------------------------

def sample_point_polyhedron_inside(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_translate_inside(rng, g2)
    cx, cy = g2.footprint.centroid.x, g2.footprint.centroid.y
    z_mid = 0.5 * (g2.z_bot + g2.z_top)
    for _ in range(50):
        x = cx + rng.uniform(-0.05, 0.05)
        y = cy + rng.uniform(-0.05, 0.05)
        z = z_mid + rng.uniform(-0.02, 0.02)
        if g2.footprint.contains(ShpPoint(x, y)) and g2.z_bot <= z <= g2.z_top:
            return np.array([x, y, z]), g2
    return np.array([cx, cy, z_mid]), g2


def sample_point_polyhedron_disjoint(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_translate_inside(rng, g2)
    # Put the point above g2's top face with a safe margin.
    g1 = np.array([
        g2.footprint.centroid.x + rng.uniform(-0.2, 0.2),
        g2.footprint.centroid.y + rng.uniform(-0.2, 0.2),
        min(0.95, g2.z_top + 0.1 + rng.uniform(0, 0.2)),
    ])
    if g2.footprint.contains(ShpPoint(float(g1[0]), float(g1[1]))) and g2.z_bot <= g1[2] <= g2.z_top:
        # fallback: shove outside horizontally
        g1[0] = min(max(g2.footprint.bounds[2] + 0.2, -1.0), 1.0)
    return g1, g2


def sample_polyhedron_point_contains(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, np.ndarray]:
    p, g2 = sample_point_polyhedron_inside(rng, complexity)
    return g2, p


def sample_polyhedron_point_disjoint(
    rng: random.Random, complexity: str
) -> Tuple[Polyhedron3D, np.ndarray]:
    p, g2 = sample_point_polyhedron_disjoint(rng, complexity)
    return g2, p


# ---------------------------------------------------------------------------
# Polyline / polyhedron
# ---------------------------------------------------------------------------

def sample_polyline_polyhedron_disjoint(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    g1 = gen.random_polyline_3d(rng, gen.complexity_params(complexity, rng)["pl_n"])
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    # g1 on the left, g2 on the right
    g1 = translate_3d(g1, rng.uniform(-0.8, -0.3), rng.uniform(-0.5, 0.5), 0.0)
    g2 = translate_3d(g2, rng.uniform(0.3, 0.7), rng.uniform(-0.5, 0.5), 0.0)
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


def sample_polyline_polyhedron_within(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    """Polyline fully inside prism: shrink polyline and center on prism centroid."""
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"],
                                height_min=0.4, height_max=0.8)
    g2 = gen.random_translate_inside(rng, g2)
    g1 = gen.random_polyline_3d(rng, gen.complexity_params(complexity, rng)["pl_n"],
                                 step_min=0.01, step_max=0.04, max_extent=0.06)
    cx, cy = g2.footprint.centroid.x, g2.footprint.centroid.y
    z_mid = 0.5 * (g2.z_bot + g2.z_top)
    g1 = translate_3d(g1, cx, cy, z_mid)
    # Defensive shrink if any vertex escaped.
    for _ in range(4):
        ok = all(
            g2.footprint.contains(ShpPoint(float(p[0]), float(p[1])))
            and g2.z_bot <= float(p[2]) <= g2.z_top
            for p in g1
        )
        if ok:
            break
        g1 = scale_3d(g1, 0.5, 0.5, 0.5, (cx, cy, z_mid))
    return g1, g2


def sample_polyline_polyhedron_intersects(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    """Polyline has some vertices inside and some outside the prism."""
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_translate_inside(rng, g2)
    cx, cy = g2.footprint.centroid.x, g2.footprint.centroid.y
    z_mid = 0.5 * (g2.z_bot + g2.z_top)
    # Build a polyline that starts inside the prism and walks out.
    n = gen.complexity_params(complexity, rng)["pl_n"]
    pts = [np.array([cx, cy, z_mid])]
    direction = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
    direction /= max(np.linalg.norm(direction), 1e-9)
    step = 0.12
    for _ in range(n - 1):
        pts.append(pts[-1] + step * direction)
    g1 = np.array(pts)
    return _clamp_into_unit_box(g1), g2


def sample_polyline_polyhedron_touches(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, Polyhedron3D]:
    """Polyline endpoint lies on prism boundary (approx via footprint boundary + a matching z)."""
    g2 = gen.random_polyhedron(rng, gen.complexity_params(complexity, rng)["ph_n"])
    g2 = gen.random_translate_inside(rng, g2)
    # pick a boundary vertex of g2's footprint
    coords = list(g2.footprint.exterior.coords)[:-1] or [(0.0, 0.0)]
    x_b, y_b = coords[rng.randint(0, len(coords) - 1)]
    z_b = rng.choice([g2.z_bot, g2.z_top, 0.5 * (g2.z_bot + g2.z_top)])
    # build short polyline ending at (x_b, y_b, z_b), starting outside
    outward = np.array([x_b, y_b]) - np.array([g2.footprint.centroid.x, g2.footprint.centroid.y])
    nrm = np.linalg.norm(outward)
    if nrm < 1e-9:
        outward = np.array([1.0, 0.0])
        nrm = 1.0
    outward = outward / nrm * 0.15
    start = np.array([x_b + outward[0], y_b + outward[1], z_b + 0.02])
    mid = np.array([x_b + 0.5 * outward[0], y_b + 0.5 * outward[1], z_b + 0.01])
    end = np.array([x_b, y_b, z_b])
    g1 = np.vstack([start, mid, end])
    return _clamp_into_unit_box(g1), g2


# ---------------------------------------------------------------------------
# Polyline / polyline
# ---------------------------------------------------------------------------

def sample_polyline_polyline_disjoint(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, np.ndarray]:
    g1 = gen.random_polyline_3d(rng, gen.complexity_params(complexity, rng)["pl_n"])
    g2 = gen.random_polyline_3d(rng, gen.complexity_params(complexity, rng)["pl_n"])
    g1 = translate_3d(g1, rng.uniform(-0.8, -0.2), rng.uniform(-0.5, 0.5), 0.0)
    g2 = translate_3d(g2, rng.uniform(0.2, 0.8), rng.uniform(-0.5, 0.5), 0.0)
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


def sample_polyline_polyline_intersects(
    rng: random.Random, complexity: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Build two polylines sharing a common midpoint."""
    pivot = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)])
    d1 = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
    d2 = np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])
    d1 = d1 / max(np.linalg.norm(d1), 1e-9) * 0.2
    d2 = d2 / max(np.linalg.norm(d2), 1e-9) * 0.2
    g1 = np.vstack([pivot - d1, pivot, pivot + d1])
    g2 = np.vstack([pivot - d2, pivot, pivot + d2])
    return _clamp_into_unit_box(g1), _clamp_into_unit_box(g2)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_TABLE = {
    ("polyhedron-polyhedron", "disjoint"):   sample_polyhedron_polyhedron_disjoint,
    ("polyhedron-polyhedron", "equals"):     sample_polyhedron_polyhedron_equals,
    ("polyhedron-polyhedron", "contains"):   sample_polyhedron_polyhedron_contains,
    ("polyhedron-polyhedron", "within"):     sample_polyhedron_polyhedron_within,
    ("polyhedron-polyhedron", "touches"):    sample_polyhedron_polyhedron_touches,
    ("polyhedron-polyhedron", "intersects"): sample_polyhedron_polyhedron_intersects,
    ("point-polyhedron", "inside"):          sample_point_polyhedron_inside,
    ("point-polyhedron", "disjoint"):        sample_point_polyhedron_disjoint,
    ("polyhedron-point", "contains"):        sample_polyhedron_point_contains,
    ("polyhedron-point", "disjoint"):        sample_polyhedron_point_disjoint,
    ("polyline-polyhedron", "disjoint"):     sample_polyline_polyhedron_disjoint,
    ("polyline-polyhedron", "within"):       sample_polyline_polyhedron_within,
    ("polyline-polyhedron", "intersects"):   sample_polyline_polyhedron_intersects,
    ("polyline-polyhedron", "touches"):      sample_polyline_polyhedron_touches,
    ("polyline-polyline", "disjoint"):       sample_polyline_polyline_disjoint,
    ("polyline-polyline", "intersects"):     sample_polyline_polyline_intersects,
}


def get_topology_sampler_3d(pair_type: str, label: str):
    return _TABLE.get((pair_type, label))


def sample_targeted_topology_pair_3d(
    rng: random.Random,
    pair_type: str,
    target_label: str,
    complexity: str,
) -> Tuple[Any, Any]:
    """Return a pair that *should* satisfy `target_label` (verified by caller)."""
    sampler = get_topology_sampler_3d(pair_type, target_label)
    if sampler is not None:
        return sampler(rng, complexity)
    # Fallback — hope the random pair matches.
    type1, type2 = pair_type.split("-")
    return gen.make_pair(rng, type1, type2, complexity)
