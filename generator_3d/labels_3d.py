"""
Labels for 3D spatial tasks.

  - distance:    continuous float
  - directional: 48-class (16 azimuth bins × 3 elevation bins) or named pair
  - topology:    3D generalization of the 2D DE-9IM subset, exact for prisms
                 via (2D footprint relation) × (1D z-range relation).
"""

from __future__ import annotations

import math
from typing import Tuple, Optional

import numpy as np
from shapely.geometry import Point as ShpPoint, LineString

from .geometry_generators_3d import Geom3D, Polyhedron3D


# ---------------------------------------------------------------------------
# Direction: (azimuth, elevation)
# ---------------------------------------------------------------------------

AZ_LABELS = [
    "E", "ENE", "NE", "NNE",
    "N", "NNW", "NW", "WNW",
    "W", "WSW", "SW", "SSW",
    "S", "SSE", "SE", "ESE",
]
AZ_BIN_W = 2.0 * math.pi / 16.0

# Elevation bins — 3 classes: below, level (|phi| < 15°), above.
EL_LABELS = ["below", "level", "above"]
EL_THRESH = math.radians(15.0)


def _wrap_0_2pi(t: float) -> float:
    t = t % (2.0 * math.pi)
    return t if t >= 0 else t + 2.0 * math.pi


def angle_to_az(theta: float) -> int:
    t = _wrap_0_2pi(theta)
    return int((t + AZ_BIN_W / 2.0) // AZ_BIN_W) % 16


def elevation_to_bin(phi: float) -> int:
    if phi > EL_THRESH:
        return 2
    if phi < -EL_THRESH:
        return 0
    return 1


def representative_xyz(geom: Geom3D) -> np.ndarray:
    if isinstance(geom, np.ndarray):
        if geom.ndim == 1:
            return geom.astype(float)
        return geom.mean(axis=0).astype(float)
    if isinstance(geom, Polyhedron3D):
        return geom.centroid_3d()
    raise ValueError(f"Unsupported geom type: {type(geom)}")


def directional_label_3d(g1: Geom3D, g2: Geom3D) -> Tuple[int, str]:
    p1 = representative_xyz(g1)
    p2 = representative_xyz(g2)
    d = p2 - p1
    theta = math.atan2(d[1], d[0])
    horiz = math.hypot(d[0], d[1])
    if horiz > 1e-12:
        phi = math.atan2(d[2], horiz)
    else:
        phi = math.pi / 2.0 if d[2] > 0 else -math.pi / 2.0
    az = angle_to_az(theta)
    el = elevation_to_bin(phi)
    label_id = el * 16 + az  # 0..47
    label_name = f"{EL_LABELS[el]}-{AZ_LABELS[az]}"
    return label_id, label_name


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def _footprint_and_zrange(geom: Geom3D):
    """Project a geom to (shapely-in-xy, (z_lo, z_hi))."""
    if isinstance(geom, np.ndarray):
        if geom.ndim == 1:
            return ShpPoint(float(geom[0]), float(geom[1])), (float(geom[2]), float(geom[2]))
        line = LineString([(float(p[0]), float(p[1])) for p in geom])
        zs = geom[:, 2]
        return line, (float(zs.min()), float(zs.max()))
    if isinstance(geom, Polyhedron3D):
        return geom.footprint, (geom.z_bot, geom.z_top)
    raise ValueError(f"Unsupported geom type: {type(geom)}")


def distance_label_3d(g1: Geom3D, g2: Geom3D) -> float:
    """Exact 3D distance for prism pairs; 2D-XY distance + 1D z-gap for other cases."""
    f1, (z1_lo, z1_hi) = _footprint_and_zrange(g1)
    f2, (z2_lo, z2_hi) = _footprint_and_zrange(g2)
    d_xy = float(f1.distance(f2))
    if z1_hi < z2_lo:
        dz = z2_lo - z1_hi
    elif z2_hi < z1_lo:
        dz = z1_lo - z2_hi
    else:
        dz = 0.0
    return math.hypot(d_xy, dz)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

TOPOLOGY_LABELS_POLYHEDRON_POLYHEDRON = [
    "disjoint", "touches", "intersects", "contains", "within", "equals",
]
TOPOLOGY_LABELS_POINT_POLYHEDRON = ["disjoint", "inside"]
TOPOLOGY_LABELS_POLYHEDRON_POINT = ["disjoint", "contains"]
TOPOLOGY_LABELS_POLYLINE_POLYHEDRON = ["disjoint", "touches", "within", "intersects"]
TOPOLOGY_LABELS_POLYLINE_POLYLINE = ["disjoint", "intersects"]
TOPOLOGY_LABELS_POINT_POINT = ["disjoint", "equals"]


def _shapely_2d_rel(a, b) -> str:
    if a.equals(b):
        return "equals"
    if a.contains(b):
        return "contains"
    if a.within(b):
        return "within"
    if a.touches(b):
        return "touches"
    if a.intersects(b):
        return "intersects"
    return "disjoint"


def _z_rel(a_lo: float, a_hi: float, b_lo: float, b_hi: float, tol: float = 1e-9) -> str:
    if abs(a_lo - b_lo) < tol and abs(a_hi - b_hi) < tol:
        return "equals"
    if a_hi < b_lo - tol or b_hi < a_lo - tol:
        return "disjoint"
    if abs(a_hi - b_lo) < tol or abs(b_hi - a_lo) < tol:
        return "touches"
    if a_lo <= b_lo + tol and a_hi >= b_hi - tol:
        return "contains"
    if b_lo <= a_lo + tol and b_hi >= a_hi - tol:
        return "within"
    return "overlap"


def _combine_3d_topology(xy: str, z: str) -> str:
    """Combine a 2D footprint relation with a 1D z relation into a 3D relation."""
    if xy == "disjoint" or z == "disjoint":
        return "disjoint"
    if xy == "equals" and z == "equals":
        return "equals"
    if xy == "touches" or z == "touches":
        return "touches"
    # contains: XY contains AND z contains (or equal)
    if (xy == "contains" and z in ("contains", "equals")) or \
       (xy == "equals" and z == "contains"):
        return "contains"
    if (xy == "within" and z in ("within", "equals")) or \
       (xy == "equals" and z == "within"):
        return "within"
    return "intersects"


def topology_label_3d(pair_type: str, g1: Geom3D, g2: Geom3D) -> Optional[str]:
    if pair_type == "point-point":
        assert isinstance(g1, np.ndarray) and g1.ndim == 1
        assert isinstance(g2, np.ndarray) and g2.ndim == 1
        return "equals" if float(np.linalg.norm(g1 - g2)) < 1e-9 else "disjoint"

    if pair_type == "point-polyhedron":
        assert isinstance(g1, np.ndarray) and g1.ndim == 1
        assert isinstance(g2, Polyhedron3D)
        in_xy = g2.footprint.contains(ShpPoint(float(g1[0]), float(g1[1])))
        in_z = (g2.z_bot <= float(g1[2]) <= g2.z_top)
        return "inside" if (in_xy and in_z) else "disjoint"

    if pair_type == "polyhedron-point":
        lab = topology_label_3d("point-polyhedron", g2, g1)
        return "contains" if lab == "inside" else "disjoint"

    if pair_type == "polyhedron-polyhedron":
        assert isinstance(g1, Polyhedron3D) and isinstance(g2, Polyhedron3D)
        xy = _shapely_2d_rel(g1.footprint, g2.footprint)
        z = _z_rel(g1.z_bot, g1.z_top, g2.z_bot, g2.z_top)
        return _combine_3d_topology(xy, z)

    if pair_type == "polyline-polyhedron":
        assert isinstance(g1, np.ndarray) and g1.ndim == 2
        assert isinstance(g2, Polyhedron3D)
        inside_flags = [
            g2.footprint.contains(ShpPoint(float(p[0]), float(p[1])))
            and (g2.z_bot <= float(p[2]) <= g2.z_top)
            for p in g1
        ]
        if all(inside_flags):
            return "within"
        line_xy = LineString([(float(p[0]), float(p[1])) for p in g1])
        if not line_xy.intersects(g2.footprint):
            return "disjoint"
        z_lo, z_hi = float(g1[:, 2].min()), float(g1[:, 2].max())
        zr = _z_rel(z_lo, z_hi, g2.z_bot, g2.z_top)
        if zr == "disjoint":
            return "disjoint"
        if zr == "touches":
            return "touches"
        return "intersects"

    if pair_type == "polyline-polyline":
        assert isinstance(g1, np.ndarray) and g1.ndim == 2
        assert isinstance(g2, np.ndarray) and g2.ndim == 2
        # 3D polyline-polyline exact intersection is annoying without trimesh;
        # MVP: project to XY and require z-range overlap (necessary but not sufficient).
        l1 = LineString([(float(p[0]), float(p[1])) for p in g1])
        l2 = LineString([(float(p[0]), float(p[1])) for p in g2])
        if not l1.intersects(l2):
            return "disjoint"
        z1 = (float(g1[:, 2].min()), float(g1[:, 2].max()))
        z2 = (float(g2[:, 2].min()), float(g2[:, 2].max()))
        if z1[1] < z2[0] - 1e-9 or z2[1] < z1[0] - 1e-9:
            return "disjoint"
        return "intersects"

    return None


def get_topology_labels_for_pair_type(pair_type: str) -> list:
    m = {
        "point-point": TOPOLOGY_LABELS_POINT_POINT,
        "point-polyhedron": TOPOLOGY_LABELS_POINT_POLYHEDRON,
        "polyhedron-point": TOPOLOGY_LABELS_POLYHEDRON_POINT,
        "polyhedron-polyhedron": TOPOLOGY_LABELS_POLYHEDRON_POLYHEDRON,
        "polyline-polyhedron": TOPOLOGY_LABELS_POLYLINE_POLYHEDRON,
        "polyline-polyline": TOPOLOGY_LABELS_POLYLINE_POLYLINE,
    }
    return m.get(pair_type, [])
