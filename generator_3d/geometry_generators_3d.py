"""
3D geometry generators.

Primitives (MVP):
  - point:       np.ndarray shape (3,)
  - polyline:    np.ndarray shape (N, 3)
  - polyhedron:  Polyhedron3D (extruded 2D star-polygon, LoD1-like prism)

Design choice: polyhedra are restricted to vertical extrusions of 2D footprints.
This keeps topology/distance exact via 2D Shapely + 1D z-range tests, and
matches how OSM/CityGML LoD1 buildings are usually stored. More general meshes
(non-vertical faces, full trimesh objects) can be added later without breaking
the Geom3D union.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate as shp_translate, scale as shp_scale


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class Polyhedron3D:
    """Extruded prism: 2D footprint extruded vertically from z_bot to z_top."""
    footprint: Polygon
    z_bot: float
    z_top: float

    def bounds_3d(self) -> Tuple[float, float, float, float, float, float]:
        minx, miny, maxx, maxy = self.footprint.bounds
        return minx, miny, self.z_bot, maxx, maxy, self.z_top

    def centroid_3d(self) -> np.ndarray:
        c = self.footprint.centroid
        return np.array([c.x, c.y, 0.5 * (self.z_bot + self.z_top)])


# Point: ndarray of shape (3,). Polyline: ndarray of shape (N, 3).
Geom3D = Union[np.ndarray, Polyhedron3D]


# ---------------------------------------------------------------------------
# Random primitives
# ---------------------------------------------------------------------------

def random_point_3d(rng: random.Random) -> np.ndarray:
    return np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)])


def random_polyline_3d(
    rng: random.Random,
    n_vertices: int,
    step_min: float = 0.05,
    step_max: float = 0.2,
    max_extent: float = 0.45,
) -> np.ndarray:
    """Random walk in 3D (uniform directions on the sphere), recentered and scaled."""
    pts = [np.zeros(3)]
    for _ in range(n_vertices - 1):
        theta = rng.uniform(0, 2 * math.pi)
        # Uniform on sphere: phi = acos(1 - 2*u)
        phi = math.acos(1.0 - 2.0 * rng.uniform(0.0, 1.0))
        step = rng.uniform(step_min, step_max)
        dx = step * math.sin(phi) * math.cos(theta)
        dy = step * math.sin(phi) * math.sin(theta)
        dz = step * math.cos(phi)
        pts.append(pts[-1] + np.array([dx, dy, dz]))
    arr = np.array(pts)
    arr = arr - arr.mean(axis=0, keepdims=True)
    mx = float(np.max(np.abs(arr)))
    if mx > max_extent and mx > 0:
        arr = arr * (max_extent / mx)
    return arr


def random_star_polygon_2d(
    rng: random.Random,
    n_vertices: int,
    nonconvex: bool = True,
    radius_min: float = 0.08,
    radius_max: float = 0.35,
) -> Polygon:
    """2D star polygon footprint (same recipe as 2D generator, kept local)."""
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n_vertices))
    if nonconvex:
        radii = [rng.uniform(radius_min, radius_max) for _ in range(n_vertices)]
    else:
        base = rng.uniform(0.15, 0.3)
        radii = [rng.uniform(base * 0.85, base * 1.15) for _ in range(n_vertices)]
    shell = [(r * math.cos(a), r * math.sin(a)) for r, a in zip(radii, angles)]
    poly = Polygon(shell).buffer(0)
    if poly.is_empty or not poly.is_valid or poly.area < 1e-4:
        return random_star_polygon_2d(rng, n_vertices, nonconvex, radius_min, radius_max)
    if hasattr(poly, "geoms"):  # MultiPolygon from buffer(0) self-intersection fix
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def random_polyhedron(
    rng: random.Random,
    n_vertices: int,
    height_min: float = 0.1,
    height_max: float = 0.6,
) -> Polyhedron3D:
    """Extruded prism with random 2D footprint and vertical extent."""
    footprint = random_star_polygon_2d(rng, n_vertices, nonconvex=True)
    height = rng.uniform(height_min, height_max)
    z_bot = rng.uniform(-1.0, 1.0 - height)
    z_top = z_bot + height
    return Polyhedron3D(footprint=footprint, z_bot=z_bot, z_top=z_top)


# ---------------------------------------------------------------------------
# Complexity + pair construction
# ---------------------------------------------------------------------------

def complexity_params(complexity: str, rng: random.Random) -> dict:
    if complexity == "low":
        return {"pl_n": rng.randint(2, 4), "ph_n": rng.randint(3, 6)}
    if complexity == "mid":
        return {"pl_n": rng.randint(5, 8), "ph_n": rng.randint(7, 12)}
    return {"pl_n": rng.randint(9, 18), "ph_n": rng.randint(13, 24)}


def place_geom_3d(rng: random.Random, gtype: str, complexity: str) -> Geom3D:
    params = complexity_params(complexity, rng)
    if gtype == "point":
        return random_point_3d(rng)
    if gtype == "polyline":
        return random_polyline_3d(rng, params["pl_n"])
    if gtype == "polyhedron":
        return random_polyhedron(rng, params["ph_n"])
    raise ValueError(f"Unknown 3D gtype: {gtype}")


# ---------------------------------------------------------------------------
# Transforms + bounds
# ---------------------------------------------------------------------------

def translate_3d(geom: Geom3D, dx: float, dy: float, dz: float) -> Geom3D:
    if isinstance(geom, np.ndarray):
        return geom + np.array([dx, dy, dz])
    if isinstance(geom, Polyhedron3D):
        new_fp = shp_translate(geom.footprint, dx, dy)
        return Polyhedron3D(new_fp, geom.z_bot + dz, geom.z_top + dz)
    raise ValueError(f"Unsupported geom type: {type(geom)}")


def scale_3d(
    geom: Geom3D,
    sx: float, sy: float, sz: float,
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Geom3D:
    if isinstance(geom, np.ndarray):
        o = np.array(origin)
        s = np.array([sx, sy, sz])
        return (geom - o) * s + o
    if isinstance(geom, Polyhedron3D):
        ox, oy, oz = origin
        new_fp = shp_scale(geom.footprint, xfact=sx, yfact=sy, origin=(ox, oy))
        new_zb = (geom.z_bot - oz) * sz + oz
        new_zt = (geom.z_top - oz) * sz + oz
        return Polyhedron3D(new_fp, new_zb, new_zt)
    raise ValueError(f"Unsupported geom type: {type(geom)}")


def bounds_3d(geom: Geom3D) -> Tuple[float, float, float, float, float, float]:
    if isinstance(geom, np.ndarray):
        if geom.ndim == 1:
            x, y, z = float(geom[0]), float(geom[1]), float(geom[2])
            return x, y, z, x, y, z
        mins = geom.min(axis=0)
        maxs = geom.max(axis=0)
        return (
            float(mins[0]), float(mins[1]), float(mins[2]),
            float(maxs[0]), float(maxs[1]), float(maxs[2]),
        )
    if isinstance(geom, Polyhedron3D):
        return geom.bounds_3d()
    raise ValueError(f"Unsupported geom type: {type(geom)}")


def _inside_unit_box(b: Tuple[float, float, float, float, float, float]) -> bool:
    return all(-1.0 <= v <= 1.0 for v in b)


def random_translate_inside(
    rng: random.Random,
    geom: Geom3D,
    max_attempts: int = 50,
) -> Geom3D:
    """Translate so bounds fit inside [-1, 1]^3. Caller supplies geom already near origin."""
    for _ in range(max_attempts):
        dx = rng.uniform(-0.7, 0.7)
        dy = rng.uniform(-0.7, 0.7)
        dz = rng.uniform(-0.7, 0.7)
        g = translate_3d(geom, dx, dy, dz)
        if _inside_unit_box(bounds_3d(g)):
            return g
    return geom


def make_pair(
    rng: random.Random,
    type1: str,
    type2: str,
    complexity: str,
) -> Tuple[Geom3D, Geom3D]:
    g1 = random_translate_inside(rng, place_geom_3d(rng, type1, complexity))
    g2 = random_translate_inside(rng, place_geom_3d(rng, type2, complexity))
    return g1, g2
