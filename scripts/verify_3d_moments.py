#!/usr/bin/env python3
"""
Numerical verification of the 3D moment formulas in
`utils/moments_compute_3d.py` against closed-form reference values.

Runs five sanity checks, each of which the theory doc predicts exactly:

  1. Point at origin — delta sifting: C_mnp = phi_mnp(0, 0, 0)
  2. Axis-aligned unit segment — integrate x^m along [-0.5, 0.5]
  3. Right triangle in xy-plane — classical first-order surface moments
  4. Unit cube centered at origin — volume, 2nd-order monomial, Fourier DC
  5. Regular tetrahedron — volume check

Each row reports (formula, computed, reference, abs_error).

Usage:  py scripts/verify_3d_moments.py
"""

from __future__ import annotations

import math
import sys
import os

# Make repo root importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from utils.moments_compute_3d import (
    moment_of_point_monomial_3d,
    moment_of_point_fourier_3d,
    moment_of_line_segment_monomial_3d,
    moment_of_line_segment_fourier_3d,
    moment_of_polygon_monomial_3d,
    moment_of_polyhedron_monomial_3d,
    moment_of_polyhedron_fourier_3d,
)
from utils.moments_compute_3d_vectorized import (
    MomentTables3D,
    batch_moments_polyhedra_monomial_3d,
    batch_moments_polyhedra_fourier_3d,
    batch_moments_points_monomial_3d,
    batch_moments_points_fourier_3d,
)


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def unit_cube_faces(side: float = 1.0) -> np.ndarray:
    """Axis-aligned cube centered at origin, triangulated with outward normals.

    Returns faces: ndarray of shape (12, 3, 3) — two triangles per face, six faces.
    Vertex ordering is CCW as viewed from outside (so cross(e1, e2) points outward).
    """
    s = side / 2.0
    # 8 cube vertices (signs of ±s in x, y, z)
    V = np.array([
        [-s, -s, -s],  # 0
        [ s, -s, -s],  # 1
        [ s,  s, -s],  # 2
        [-s,  s, -s],  # 3
        [-s, -s,  s],  # 4
        [ s, -s,  s],  # 5
        [ s,  s,  s],  # 6
        [-s,  s,  s],  # 7
    ])
    # Each cube face as two CCW triangles (viewed from outside).
    quads = [
        (0, 3, 2, 1),   # z = -s (bottom), outward normal (0,0,-1)
        (4, 5, 6, 7),   # z = +s (top),    outward normal (0,0,+1)
        (0, 1, 5, 4),   # y = -s,          outward normal (0,-1,0)
        (2, 3, 7, 6),   # y = +s,          outward normal (0,+1,0)
        (0, 4, 7, 3),   # x = -s,          outward normal (-1,0,0)
        (1, 2, 6, 5),   # x = +s,          outward normal (+1,0,0)
    ]
    faces = []
    for (a, b, c, d) in quads:
        faces.append([V[a], V[b], V[c]])
        faces.append([V[a], V[c], V[d]])
    return np.array(faces)


def tetrahedron_faces(scale: float = 1.0) -> np.ndarray:
    """Regular tetrahedron centered near origin with edge length ~scale.

    Uses the 4 alternating vertices of a cube: {(+,+,+), (+,-,-), (-,+,-), (-,-,+)} * scale/2.
    Edge length = sqrt(2) * scale.
    Outward-CCW faces as seen from outside.
    """
    s = scale / 2.0
    V = np.array([
        [ s,  s,  s],   # 0
        [ s, -s, -s],   # 1
        [-s,  s, -s],   # 2
        [-s, -s,  s],   # 3
    ])
    # CCW outward faces (verified: cross(e1, e2) points away from the opposite vertex):
    faces = np.array([
        [V[0], V[1], V[2]],   # opposite V3
        [V[0], V[3], V[1]],   # opposite V2
        [V[0], V[2], V[3]],   # opposite V1
        [V[1], V[3], V[2]],   # opposite V0
    ])
    return faces


def right_triangle_xy() -> np.ndarray:
    """Right triangle (0,0,0) -> (1,0,0) -> (0,1,0)."""
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])


# ---------------------------------------------------------------------------
# Reference (analytic) values
# ---------------------------------------------------------------------------

def cube_monomial_reference(side: float, m: int, n: int, p: int) -> float:
    """C_mnp = ∫∫∫_cube x^m y^n z^p dV   (cube centered at origin, edge = side)."""
    def axis(k: int) -> float:
        if k % 2 == 1:
            return 0.0
        half = side / 2.0
        return 2.0 * (half ** (k + 1)) / (k + 1)
    return axis(m) * axis(n) * axis(p)


def cube_fourier_reference(side: float, m: int, n: int, p: int) -> complex:
    """Fourier moment of a cube centered at origin. Uses sinc identity."""
    def axis(k: int) -> complex:
        if k == 0:
            return side
        # ∫_{-s/2}^{s/2} e^{-j π k x} dx = 2 sin(π k side/2) / (π k)
        return 2.0 * math.sin(math.pi * k * side / 2.0) / (math.pi * k)
    return complex(axis(m) * axis(n) * axis(p))


def segment_monomial_reference(a: tuple, b: tuple, m: int, n: int, p: int) -> float:
    """Monomial moment of a 3D line segment by direct 1D integration."""
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    L = math.sqrt(dx * dx + dy * dy + dz * dz)

    # ∫_0^1 (ax + t dx)^m (ay + t dy)^n (az + t dz)^p dt — adaptive Simpson on a fine grid.
    N = 20001
    ts = np.linspace(0.0, 1.0, N)
    ys = ((ax + ts * dx) ** m) * ((ay + ts * dy) ** n) * ((az + ts * dz) ** p)
    # Simpson's rule
    h = ts[1] - ts[0]
    s = ys[0] + ys[-1] + 4.0 * ys[1:-1:2].sum() + 2.0 * ys[2:-2:2].sum()
    integral = s * h / 3.0
    return L * integral


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def fmt(z) -> str:
    if isinstance(z, complex):
        return f"{z.real:+.6f} {z.imag:+.6f}j"
    return f"{float(z):+.6f}"


def check(name: str, computed, reference, atol: float = 1e-6):
    if isinstance(computed, complex) or isinstance(reference, complex):
        err = abs(complex(computed) - complex(reference))
        status = "OK " if err < atol else "FAIL"
        print(f"  [{status}] {name:50s} computed={fmt(computed):22s} ref={fmt(reference):22s} err={err:.2e}")
    else:
        err = abs(float(computed) - float(reference))
        status = "OK " if err < atol else "FAIL"
        print(f"  [{status}] {name:50s} computed={fmt(computed):18s} ref={fmt(reference):18s} err={err:.2e}")
    return err < atol


def main() -> int:
    all_ok = True

    print("=" * 80)
    print("§1. Point at origin")
    print("=" * 80)
    # C^pt,geo_000 = 0^0 * 0^0 * 0^0 = 1; for higher orders where any index > 0 → 0.
    all_ok &= check("point geo (0,0,0)", moment_of_point_monomial_3d(0, 0, 0, 0, 0, 0), 1.0)
    all_ok &= check("point geo (2,0,0) at origin", moment_of_point_monomial_3d(0, 0, 0, 2, 0, 0), 0.0)
    all_ok &= check("point geo (1,1,1) at (0.5,0.2,-0.3)",
                     moment_of_point_monomial_3d(0.5, 0.2, -0.3, 1, 1, 1),
                     0.5 * 0.2 * (-0.3))
    all_ok &= check("point Fourier (1,0,0) at (0.5,0,0)",
                     moment_of_point_fourier_3d(0.5, 0.0, 0.0, 1, 0, 0),
                     complex(math.cos(math.pi * 0.5), -math.sin(math.pi * 0.5)))

    print()
    print("=" * 80)
    print("§2. Axis-aligned unit segment [-0.5,0,0] -> [0.5,0,0]")
    print("=" * 80)
    a, b = (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)
    # length = 1, so C_000 = 1
    all_ok &= check("segment geo (0,0,0)",
                     moment_of_line_segment_monomial_3d(*a, *b, 0, 0, 0), 1.0)
    # ∫_{-0.5}^{0.5} x^2 dx = 1/12
    all_ok &= check("segment geo (2,0,0)",
                     moment_of_line_segment_monomial_3d(*a, *b, 2, 0, 0),
                     1.0 / 12.0)
    # Reference via direct numerical integration
    all_ok &= check("segment geo (3,0,0) [numeric ref]",
                     moment_of_line_segment_monomial_3d(*a, *b, 3, 0, 0),
                     segment_monomial_reference(a, b, 3, 0, 0))
    # Fourier: ∫_{-0.5}^{0.5} e^{-jπx} dx = 2 sin(π/2)/π = 2/π
    all_ok &= check("segment Fourier (1,0,0)",
                     moment_of_line_segment_fourier_3d(*a, *b, 1, 0, 0),
                     complex(2.0 / math.pi, 0.0))

    print()
    print("=" * 80)
    print("§3. Right triangle (0,0,0)-(1,0,0)-(0,1,0) in z=0 plane")
    print("=" * 80)
    tri = right_triangle_xy()
    # Area = 0.5, so C^tri,geo_000 = 0.5 (zero z, 2A * ∫∫_simplex 1 du dv = 2*0.5*0.5 = 0.5)
    all_ok &= check("triangle geo (0,0,0)",
                     moment_of_polygon_monomial_3d(tri, 0, 0, 0), 0.5)
    # ∫∫ x dS over T: reduces to ∫_0^1 ∫_0^(1-y) x dx dy = ∫_0^1 (1-y)^2/2 dy = 1/6
    all_ok &= check("triangle geo (1,0,0)",
                     moment_of_polygon_monomial_3d(tri, 1, 0, 0), 1.0 / 6.0)
    all_ok &= check("triangle geo (0,1,0)",
                     moment_of_polygon_monomial_3d(tri, 0, 1, 0), 1.0 / 6.0)
    all_ok &= check("triangle geo (0,0,1) [z=0 everywhere]",
                     moment_of_polygon_monomial_3d(tri, 0, 0, 1), 0.0)
    # ∫∫ x^2 dS = ∫_0^1 ∫_0^(1-y) x^2 dx dy = ∫_0^1 (1-y)^3/3 dy = 1/12
    all_ok &= check("triangle geo (2,0,0)",
                     moment_of_polygon_monomial_3d(tri, 2, 0, 0), 1.0 / 12.0)

    print()
    print("=" * 80)
    print("§4. Unit cube centered at origin (side = 1)")
    print("=" * 80)
    cube = unit_cube_faces(side=1.0)
    # Volume
    all_ok &= check("cube vol, geo (0,0,0) [volume]",
                     moment_of_polyhedron_monomial_3d(cube, 0, 0, 0), 1.0)
    # Odd-power coordinate → 0 by symmetry
    all_ok &= check("cube vol, geo (1,0,0)",
                     moment_of_polyhedron_monomial_3d(cube, 1, 0, 0), 0.0, atol=1e-9)
    # ∫∫∫ x^2 dV = 1/12
    all_ok &= check("cube vol, geo (2,0,0)",
                     moment_of_polyhedron_monomial_3d(cube, 2, 0, 0), 1.0 / 12.0)
    # ∫∫∫ x^2 y^2 dV = 1/12 * 1/12 * 1 = 1/144
    all_ok &= check("cube vol, geo (2,2,0)",
                     moment_of_polyhedron_monomial_3d(cube, 2, 2, 0), 1.0 / 144.0)
    # ∫∫∫ x^4 dV = 1/80
    all_ok &= check("cube vol, geo (4,0,0)",
                     moment_of_polyhedron_monomial_3d(cube, 4, 0, 0), 1.0 / 80.0)
    # Fourier DC = volume
    all_ok &= check("cube vol, Fourier (0,0,0) [DC branch]",
                     moment_of_polyhedron_fourier_3d(cube, 0, 0, 0), 1.0 + 0.0j)
    # Fourier (1,0,0) = 2/π (real, even)
    all_ok &= check("cube vol, Fourier (1,0,0)",
                     moment_of_polyhedron_fourier_3d(cube, 1, 0, 0),
                     cube_fourier_reference(1.0, 1, 0, 0),
                     atol=1e-5)
    # Fourier (2,0,0) = 2 sin(π) / (2π) = 0
    all_ok &= check("cube vol, Fourier (2,0,0)",
                     moment_of_polyhedron_fourier_3d(cube, 2, 0, 0),
                     cube_fourier_reference(1.0, 2, 0, 0),
                     atol=1e-5)

    print()
    print("=" * 80)
    print("§5. Regular tetrahedron (vertices from alternating cube corners at scale 1)")
    print("=" * 80)
    tet = tetrahedron_faces(scale=1.0)
    # Volume of a tetrahedron with vertices (+,+,+),(+,-,-),(-,+,-),(-,-,+) at scale 1/2
    # = (1/3) det([v1-v0, v2-v0, v3-v0]) / volume factor
    # Directly: |det| / 6 where det is the 3x3 of edge vectors from v0.
    v = tet[0]
    v0 = v[0]
    # Take the edges from the first face's first vertex — but we want the full tet volume.
    # Easier: use all 4 unique vertices.
    all_verts = np.unique(tet.reshape(-1, 3), axis=0)
    assert all_verts.shape[0] == 4
    v0, v1, v2, v3 = all_verts[0], all_verts[1], all_verts[2], all_verts[3]
    ref_vol = abs(np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
    all_ok &= check("tetra vol, geo (0,0,0) [volume]",
                     moment_of_polyhedron_monomial_3d(tet, 0, 0, 0),
                     ref_vol)

    print()
    print("=" * 80)
    print("§6. Scalar vs. vectorized cross-check (cube + tetrahedron)")
    print("=" * 80)
    tables = MomentTables3D(max_order=3)  # (O, O, O) = (4, 4, 4) grid
    # Stack cube + tet as a batch with padding
    max_F = max(cube.shape[0], tet.shape[0])
    batch_faces = np.zeros((2, max_F, 3, 3), dtype=np.float64)
    batch_faces[0, :cube.shape[0]] = cube
    batch_faces[1, :tet.shape[0]] = tet
    face_counts = np.array([cube.shape[0], tet.shape[0]], dtype=int)

    vec_geo = batch_moments_polyhedra_monomial_3d(batch_faces, face_counts, tables)
    vec_F   = batch_moments_polyhedra_fourier_3d (batch_faces, face_counts, tables)

    # Compare every (m, n, p) in the 4x4x4 grid against the scalar implementation.
    max_err_geo = 0.0
    max_err_F   = 0.0
    for b, (faces, name) in enumerate([(cube, "cube"), (tet, "tetra")]):
        for m in range(4):
            for n in range(4):
                for p in range(4):
                    ref_g = moment_of_polyhedron_monomial_3d(faces, m, n, p)
                    ref_f = moment_of_polyhedron_fourier_3d(faces, m, n, p)
                    err_g = abs(vec_geo[b, m, n, p] - ref_g)
                    err_f = abs(vec_F[b, m, n, p]   - ref_f)
                    if err_g > max_err_geo:
                        max_err_geo = err_g
                    if err_f > max_err_F:
                        max_err_F = err_f
    all_ok &= check("vectorized polyhedra geo — max error over 4^3 * 2 cases",
                     max_err_geo, 0.0, atol=1e-9)
    all_ok &= check("vectorized polyhedra Fourier — max error over 4^3 * 2 cases",
                     max_err_F, 0.0, atol=1e-9)

    # Also check vectorized point path against scalar.
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, -0.3]])
    vec_pt_geo = batch_moments_points_monomial_3d(pts, tables)
    vec_pt_F   = batch_moments_points_fourier_3d(pts, tables)
    max_err_pt_g = 0.0
    max_err_pt_f = 0.0
    for b, pt in enumerate(pts):
        for m in range(4):
            for n in range(4):
                for p in range(4):
                    ref_g = moment_of_point_monomial_3d(pt[0], pt[1], pt[2], m, n, p)
                    ref_f = moment_of_point_fourier_3d (pt[0], pt[1], pt[2], m, n, p)
                    max_err_pt_g = max(max_err_pt_g, abs(vec_pt_geo[b, m, n, p] - ref_g))
                    max_err_pt_f = max(max_err_pt_f, abs(vec_pt_F  [b, m, n, p] - ref_f))
    all_ok &= check("vectorized points geo — max error over 4^3 * 2 cases",
                     max_err_pt_g, 0.0, atol=1e-12)
    all_ok &= check("vectorized points Fourier — max error over 4^3 * 2 cases",
                     max_err_pt_f, 0.0, atol=1e-12)

    print()
    print("=" * 80)
    print("Overall:", "ALL OK" if all_ok else "FAILURES DETECTED")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
