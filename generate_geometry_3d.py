"""
Random 3D geometry generator (numpy only).

Output arrays match the input contracts of moments_compute_3d_vectorized.py:

    kind         geom shape         lengths
    -----------  -----------------  ----------------------------
    points       (B, 3)             ones (unused by the encoder)
    polylines    (B, maxS, 3)       number of valid points (>= 2)
    polyhedra    (B, maxF, 3, 3)    number of valid triangles

Polyhedra are closed triangle meshes with OUTWARD, counter-clockwise faces
(what batch_moments_polyhedra_* expects). Two styles are generated:
    prism : extruded star-shaped polygon (LoD1 building style, non-convex)
    blob  : radially perturbed icosphere (star-shaped, non-convex, faces in
            arbitrary directions)

Every geometry lies inside [-1, 1]^3, the domain assumed by the Legendre and
Fourier bases.

Usage with the moment code:

    from moments_compute_3d_vectorized import MomentTables3D, batch_encode_3d
    from generate_geometry_3d import generate_batch

    batch = generate_batch("polyhedra", 32, complexity="mid", seed=0)
    tables = MomentTables3D(max_order=4)
    moments = batch_encode_3d(tables=tables, basis="legendre", **batch.moment_kwargs())
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import numpy as np

KINDS = ("points", "polylines", "polyhedra")
COMPLEXITIES = ("low", "mid", "high")
POLYHEDRON_STYLES = ("prism", "blob", "mixed")

# pl_n: polyline vertex count range; prism_n: footprint vertices; subdiv: icosphere level
# (faces: 0 -> 20, 1 -> 80, 2 -> 320)
_PARAMS = {
    "low":  dict(pl_n=(2, 4),  prism_n=(3, 6),   subdiv=0),
    "mid":  dict(pl_n=(5, 8),  prism_n=(7, 12),  subdiv=1),
    "high": dict(pl_n=(9, 18), prism_n=(13, 24), subdiv=2),
}


# ---------------------------------------------------------------------------
# Placement helper
# ---------------------------------------------------------------------------

def _place_inside(rng: np.random.Generator, v: np.ndarray, max_half: float = 0.9) -> np.ndarray:
    """
    Recenter the bounding box on the origin, shrink uniformly if any half-extent
    exceeds max_half, then translate randomly so everything stays in [-1, 1]^3.
    Uniform scaling and translation keep face orientation intact.
    """
    flat = v.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    v = v - (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    m = float(half.max())
    if m > max_half:
        v = v * (max_half / m)
        half = half * (max_half / m)
    slack = 1.0 - half
    return v + rng.uniform(-slack, slack)


# ---------------------------------------------------------------------------
# Points and polylines
# ---------------------------------------------------------------------------

def random_point(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=3)


def random_polyline(
    rng: np.random.Generator,
    n_vertices: int,
    step_min: float = 0.05,
    step_max: float = 0.2,
    max_extent: float = 0.45,
) -> np.ndarray:
    """3D random walk with uniformly distributed directions. Returns (n_vertices, 3)."""
    d = rng.normal(size=(n_vertices - 1, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    steps = rng.uniform(step_min, step_max, size=(n_vertices - 1, 1))
    pts = np.vstack([np.zeros((1, 3)), np.cumsum(d * steps, axis=0)])
    return _place_inside(rng, pts, max_half=max_extent)


# ---------------------------------------------------------------------------
# Polyhedra: extruded star-polygon prisms
# ---------------------------------------------------------------------------

def _star_polygon(rng: np.random.Generator, n: int,
                  r_min: float = 0.08, r_max: float = 0.35) -> np.ndarray:
    """
    CCW star-shaped polygon around the origin, shape (n, 2). Angular gaps are
    kept below pi, so the origin sees every edge and a triangle fan from the
    origin is a valid triangulation even though the polygon is non-convex.
    """
    while True:
        ang = np.sort(rng.uniform(0.0, 2.0 * np.pi, n))
        gaps = np.diff(np.append(ang, ang[0] + 2.0 * np.pi))
        if gaps.max() < 0.9 * np.pi and gaps.min() > 0.02:
            break
    r = rng.uniform(r_min, r_max, n)
    return np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)


def random_prism_faces(rng: np.random.Generator, n_vertices: int,
                       h_min: float = 0.1, h_max: float = 0.6) -> np.ndarray:
    """Closed prism as 4*n outward-CCW triangles, shape (4n, 3, 3), centered at origin."""
    ring = _star_polygon(rng, n_vertices)
    n = len(ring)
    h = rng.uniform(h_min, h_max)
    z0, z1 = -h / 2.0, h / 2.0

    def lift(z):
        return np.concatenate([ring, np.full((n, 1), z)], axis=1)

    bot, top = lift(z0), lift(z1)
    cb = np.broadcast_to(np.array([0.0, 0.0, z0]), (n, 3))
    ct = np.broadcast_to(np.array([0.0, 0.0, z1]), (n, 3))
    i = np.arange(n)
    j = (i + 1) % n

    cap_top = np.stack([ct, top[i], top[j]], axis=1)       # normal +z
    cap_bot = np.stack([cb, bot[j], bot[i]], axis=1)       # normal -z
    side_a = np.stack([bot[i], bot[j], top[j]], axis=1)    # outward for CCW ring
    side_b = np.stack([bot[i], top[j], top[i]], axis=1)
    return np.concatenate([cap_top, cap_bot, side_a, side_b], axis=0)


# ---------------------------------------------------------------------------
# Polyhedra: radially perturbed icosphere ("blob")
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _icosphere(subdiv: int):
    """Unit icosphere. Returns (verts (V,3), faces (F,3)) with outward CCW faces."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [np.array(p, dtype=float) / np.linalg.norm(p) for p in [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(subdiv):
        cache = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        new = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new
    return np.array(verts), np.array(faces, dtype=int)


def random_blob_faces(rng: np.random.Generator, subdiv: int,
                      base_min: float = 0.15, base_max: float = 0.4,
                      roughness: Optional[float] = None, n_waves: int = 4) -> np.ndarray:
    """
    Star-shaped non-convex polyhedron: icosphere with radius exp(sum of low-frequency
    waves), then anisotropic scaling. Returns (F, 3, 3), outward CCW.
    Radial displacement keeps every triangle inside its original cone, so the mesh
    cannot self-intersect.
    """
    if roughness is None:
        roughness = rng.uniform(0.2, 0.7)
    v, f = _icosphere(subdiv)
    s = np.zeros(len(v))
    for _ in range(n_waves):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        freq = rng.uniform(1.0, 3.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amp = rng.uniform(0.3, 1.0) * roughness / n_waves
        s += amp * np.cos(freq * np.pi * (v @ u) + phase)
    radius = rng.uniform(base_min, base_max) * np.exp(s)
    aniso = rng.uniform(0.6, 1.4, size=3)
    verts = v * radius[:, None] * aniso[None, :]
    return verts[f]


# ---------------------------------------------------------------------------
# Single geometry + batching
# ---------------------------------------------------------------------------

def polyhedron_signed_volume(faces: np.ndarray) -> float:
    """Signed volume of a triangle mesh (positive when faces point outward)."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def generate_one(rng: np.random.Generator, kind: str, complexity: str = "mid",
                 polyhedron_style: str = "mixed") -> np.ndarray:
    """One unpadded geometry: (3,), (n, 3) or (F, 3, 3)."""
    p = _PARAMS[complexity]
    if kind == "points":
        return random_point(rng)
    if kind == "polylines":
        n = int(rng.integers(p["pl_n"][0], p["pl_n"][1] + 1))
        return random_polyline(rng, n)
    if kind == "polyhedra":
        style = polyhedron_style
        if style == "mixed":
            style = "prism" if rng.random() < 0.5 else "blob"
        if style == "prism":
            n = int(rng.integers(p["prism_n"][0], p["prism_n"][1] + 1))
            faces = random_prism_faces(rng, n)
        elif style == "blob":
            faces = random_blob_faces(rng, p["subdiv"])
        else:
            raise ValueError(f"Unknown polyhedron_style: {polyhedron_style}")
        return _place_inside(rng, faces)
    raise ValueError(f"Unknown kind: {kind}")


@dataclass
class GeomBatch:
    kind: str
    geom: np.ndarray      # padded, see module docstring
    lengths: np.ndarray   # (B,) int

    def __len__(self) -> int:
        return len(self.geom)

    def get(self, i: int) -> np.ndarray:
        """Unpadded geometry i: (3,), (n, 3) or (F, 3, 3)."""
        if self.kind == "points":
            return self.geom[i]
        return self.geom[i, : int(self.lengths[i])]

    def slice(self, start: int, stop: int) -> "GeomBatch":
        """Sub-batch with padding trimmed to the longest geometry in the slice."""
        lengths = self.lengths[start:stop]
        g = self.geom[start:stop]
        if self.kind != "points":
            g = g[:, : int(lengths.max())]
        return GeomBatch(self.kind, g, lengths)

    def moment_kwargs(self) -> dict:
        """Keyword args for batch_encode_3d (add tables= and basis=)."""
        return dict(
            geom=self.geom,
            lengths=self.lengths,
            dataset_type=self.kind,
            face_counts=self.lengths if self.kind == "polyhedra" else None,
        )

    def save(self, path: str) -> None:
        np.savez_compressed(path, kind=self.kind, geom=self.geom, lengths=self.lengths)

    @staticmethod
    def load(path: str) -> "GeomBatch":
        d = np.load(path, allow_pickle=False)
        return GeomBatch(str(d["kind"]), d["geom"], d["lengths"])


def _collate(kind: str, items: List[np.ndarray]) -> GeomBatch:
    B = len(items)
    if kind == "points":
        return GeomBatch(kind, np.stack(items).astype(np.float64), np.ones(B, dtype=int))
    lengths = np.array([len(x) for x in items], dtype=int)
    padded = np.zeros((B, int(lengths.max())) + items[0].shape[1:], dtype=np.float64)
    for b, x in enumerate(items):
        padded[b, : len(x)] = x
    return GeomBatch(kind, padded, lengths)


def generate_batch(kind: str, batch_size: int, complexity: str = "mid",
                   seed: Optional[int] = None, polyhedron_style: str = "mixed") -> GeomBatch:
    """
    complexity: "low" | "mid" | "high" | "mixed" (random per sample)
    polyhedron_style: "prism" | "blob" | "mixed"
    """
    rng = np.random.default_rng(seed)
    items = []
    for _ in range(batch_size):
        c = str(rng.choice(COMPLEXITIES)) if complexity == "mixed" else complexity
        items.append(generate_one(rng, kind, c, polyhedron_style))
    return _collate(kind, items)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a random 3D geometry batch (.npz)")
    ap.add_argument("--kind", choices=KINDS, default="polyhedra")
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--complexity", choices=COMPLEXITIES + ("mixed",), default="mid")
    ap.add_argument("--polyhedron-style", choices=POLYHEDRON_STYLES, default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="batch.npz")
    args = ap.parse_args()

    batch = generate_batch(args.kind, args.n, args.complexity, args.seed, args.polyhedron_style)
    batch.save(args.out)
    print(f"{args.kind}: geom {batch.geom.shape}, lengths [{batch.lengths.min()}..{batch.lengths.max()}] -> {args.out}")


if __name__ == "__main__":
    main()
