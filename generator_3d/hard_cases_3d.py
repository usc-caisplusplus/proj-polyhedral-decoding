"""
Hard-case perturbations for 3D pairs.

Mirrors `generator/hard_cases.py`; introduces one genuinely 3D-specific
mode: `vertical_near_touching` — two prisms aligned vertically with an
arbitrarily small z-gap. That class cannot exist in 2D and is interesting
for the topology probe.
"""

from __future__ import annotations

import math
import random
from dataclasses import replace
from typing import Tuple, Any

import numpy as np
from shapely.geometry import Point as ShpPoint
from shapely.affinity import translate as shp_translate
from shapely.ops import nearest_points

from . import geometry_generators_3d as gen
from .geometry_generators_3d import Polyhedron3D, translate_3d, bounds_3d


# ---------------------------------------------------------------------------
# Near disjoint: barely apart
# ---------------------------------------------------------------------------

def make_near_disjoint_3d(
    rng: random.Random,
    g1: Any, g2: Any,
    gap: float = 0.02,
) -> Tuple[Any, Any]:
    """Push g2 so it is barely disjoint from g1 (footprint or z)."""
    if isinstance(g1, Polyhedron3D) and isinstance(g2, Polyhedron3D):
        # Try z-axis gap first (easy): stack g2 just above g1 with `gap`.
        if rng.random() < 0.5:
            return g1, Polyhedron3D(
                footprint=g2.footprint,
                z_bot=g1.z_top + gap,
                z_top=g1.z_top + gap + (g2.z_top - g2.z_bot),
            )
        # Otherwise: horizontal nudge
        if g1.footprint.intersects(g2.footprint):
            p1, p2 = nearest_points(g1.footprint, g2.footprint)
            dx, dy = (p2.x - p1.x), (p2.y - p1.y)
            d = math.hypot(dx, dy)
            if d < 1e-9:
                dx, dy = gap, 0.0
            else:
                dx, dy = dx / d * gap, dy / d * gap
            return g1, translate_3d(g2, dx, dy, 0.0)
        return g1, g2
    # Point / polyline / mixed: fall back to a small 3D shove.
    return g1, translate_3d(g2, gap, 0.0, 0.0)


def make_vertical_near_touching(
    rng: random.Random,
    g1: Polyhedron3D,
    g2: Polyhedron3D,
    gap: float = 0.005,
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """Align two prisms so their z-faces are separated by a very small gap."""
    # g2 directly above g1, matching footprint shift not required for the hard case.
    height2 = g2.z_top - g2.z_bot
    z_bot2 = g1.z_top + gap
    return g1, Polyhedron3D(footprint=g2.footprint, z_bot=z_bot2, z_top=z_bot2 + height2)


# ---------------------------------------------------------------------------
# Almost contained
# ---------------------------------------------------------------------------

def make_almost_contained_3d(
    rng: random.Random,
    g_outer: Polyhedron3D,
    g_inner: Polyhedron3D,
    breach: float = 0.04,
) -> Tuple[Polyhedron3D, Polyhedron3D]:
    """g_inner mostly inside g_outer but slightly breaching boundary.

    Achieves the breach by nudging g_inner's z-range upward so its top
    peeks above g_outer's top face.
    """
    height_inner = g_inner.z_top - g_inner.z_bot
    # anchor g_inner to g_outer's center so it's mostly inside
    cx, cy = g_outer.footprint.centroid.x, g_outer.footprint.centroid.y
    inner_c = g_inner.footprint.centroid
    fp_inner = shp_translate(g_inner.footprint, cx - inner_c.x, cy - inner_c.y)
    z_center = 0.5 * (g_outer.z_bot + g_outer.z_top)
    z_bot = z_center - 0.5 * height_inner
    z_top = z_center + 0.5 * height_inner
    # nudge above
    z_bot += breach
    z_top += breach
    return g_outer, Polyhedron3D(footprint=fp_inner, z_bot=z_bot, z_top=z_top)


# ---------------------------------------------------------------------------
# Thin prism
# ---------------------------------------------------------------------------

def make_thin_prism(
    rng: random.Random,
    orientation: str = "tall",
) -> Polyhedron3D:
    """Very tall-thin or very flat prism — stress-tests moment stability."""
    if orientation == "tall":
        fp = gen.random_star_polygon_2d(rng, rng.randint(3, 5), nonconvex=False,
                                          radius_min=0.03, radius_max=0.08)
        h = rng.uniform(0.7, 0.95)
    else:  # flat
        fp = gen.random_star_polygon_2d(rng, rng.randint(5, 10), nonconvex=True,
                                          radius_min=0.25, radius_max=0.45)
        h = rng.uniform(0.02, 0.06)
    z_bot = rng.uniform(-0.8, 0.8 - h)
    return Polyhedron3D(footprint=fp, z_bot=z_bot, z_top=z_bot + h)


# ---------------------------------------------------------------------------
# Near-boundary placement
# ---------------------------------------------------------------------------

def make_near_boundary_3d(
    rng: random.Random,
    geom: Any,
    margin: float = 0.03,
) -> Any:
    """Translate geom so it abuts one face of [-1, 1]^3."""
    minx, miny, minz, maxx, maxy, maxz = bounds_3d(geom)
    face = rng.randint(0, 5)  # 0..5 for ±x ±y ±z
    dx = dy = dz = 0.0
    if face == 0:
        dx = (-1.0 + margin) - minx
    elif face == 1:
        dx = (1.0 - margin) - maxx
    elif face == 2:
        dy = (-1.0 + margin) - miny
    elif face == 3:
        dy = (1.0 - margin) - maxy
    elif face == 4:
        dz = (-1.0 + margin) - minz
    else:
        dz = (1.0 - margin) - maxz
    return translate_3d(geom, dx, dy, dz)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def apply_hard_case_3d(
    rng: random.Random,
    g1: Any, g2: Any,
    mode: str,
    pair_type: str,
) -> Tuple[Any, Any]:
    if mode == "none":
        return g1, g2

    if mode == "near_disjoint":
        return make_near_disjoint_3d(rng, g1, g2)

    if mode == "almost_contained":
        if isinstance(g1, Polyhedron3D) and isinstance(g2, Polyhedron3D):
            # the larger one is the outer
            vol1 = g1.footprint.area * (g1.z_top - g1.z_bot)
            vol2 = g2.footprint.area * (g2.z_top - g2.z_bot)
            if vol1 >= vol2:
                return make_almost_contained_3d(rng, g1, g2)
            return make_almost_contained_3d(rng, g2, g1)
        return g1, g2

    if mode == "thin_prism":
        t1, t2 = pair_type.split("-")
        if t1 == "polyhedron":
            g1 = gen.random_translate_inside(rng, make_thin_prism(rng, rng.choice(["tall", "flat"])))
        if t2 == "polyhedron":
            g2 = gen.random_translate_inside(rng, make_thin_prism(rng, rng.choice(["tall", "flat"])))
        return g1, g2

    if mode == "near_boundary":
        return make_near_boundary_3d(rng, g1), make_near_boundary_3d(rng, g2)

    if mode == "vertical_near_touching":
        if isinstance(g1, Polyhedron3D) and isinstance(g2, Polyhedron3D):
            return make_vertical_near_touching(rng, g1, g2)
        return g1, g2

    if mode == "mixed":
        choice = rng.choice([
            "near_disjoint", "almost_contained", "thin_prism",
            "near_boundary", "vertical_near_touching",
        ])
        return apply_hard_case_3d(rng, g1, g2, choice, pair_type)

    return g1, g2
