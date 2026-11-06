"""
Snapshot renderer for points, polylines and triangle-mesh polyhedra.

Pure numpy (z-buffer rasterizer, orthographic camera), so occlusion is exact per
pixel and framing is identical for every image. Takes the same unpadded arrays
that generate_geometry_3d.GeomBatch.get(i) returns:

    points     (3,)
    polylines  (n, 3)
    polyhedra  (F, 3, 3)

Images contain only the geometry on a plain background: no axes, no text.
Lit with a light attached to the camera, so face brightness encodes orientation.

Camera: elev/azim in degrees. (elev=0, azim=-90) looks along +y with x to the right
and z up; elev=90 looks straight down with x to the right and y up.

By default the framing is fitted to the [-1, 1]^3 cube *for each view* (the same
for every sample), so nothing can be clipped and absolute position stays
recoverable. Pass view_extent=<float> (half-width in world units) to zoom.
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

VIEW_PRESETS: Dict[str, Tuple[float, float]] = {
    "iso":      (30, -60),
    "iso_back": (30, 120),
    "top":      (90, -90),
    "bottom":   (-90, -90),
    "front":    (0, -90),
    "right":    (0, 0),
    "back":     (0, 90),
    "left":     (0, 180),
}
DEFAULT_VIEWS = ("iso", "iso_back", "top", "front")

DEFAULT_COLORS = {
    "polyhedra": (70, 120, 200),
    "polylines": (220, 60, 50),
    "points":    (40, 150, 60),
}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

def camera_basis(elev: float, azim: float):
    """Unit vectors (right, up, toward_camera); right x up = toward_camera."""
    e, a = np.radians(elev), np.radians(azim)
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(c, right)
    return right, up, c


class _Camera:
    def __init__(self, elev, azim, size, ssaa, view_extent):
        self.right, self.up, self.c = camera_basis(elev, azim)
        if view_extent is None:  # projected half-size of the [-1,1]^3 cube
            view_extent = 1.03 * max(np.abs(self.right).sum(), np.abs(self.up).sum())
        self.S = size * ssaa
        self.ssaa = ssaa
        self.scale = self.S / (2.0 * view_extent)

    def project(self, P: np.ndarray):
        sx = (P @ self.right) * self.scale + self.S / 2.0
        sy = self.S / 2.0 - (P @ self.up) * self.scale
        return sx, sy, P @ self.c   # depth: larger = closer to camera


# ---------------------------------------------------------------------------
# Rasterization primitives
# ---------------------------------------------------------------------------

def _fill_triangles(img, zbuf, xy, z, rgb) -> None:
    """xy (T,3,2) pixel coords, z (T,3) depth per vertex, rgb (T,3)."""
    H, W = zbuf.shape
    for t in range(len(xy)):
        p = xy[t]
        x0 = max(int(np.floor(p[:, 0].min())), 0)
        x1 = min(int(np.ceil(p[:, 0].max())), W - 1)
        y0 = max(int(np.floor(p[:, 1].min())), 0)
        y1 = min(int(np.ceil(p[:, 1].max())), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        (ax, ay), (bx, by), (cx, cy) = p
        det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(det) < 1e-12:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Y - cy)) / det
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Y - cy)) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        zz = w0 * z[t, 0] + w1 * z[t, 1] + w2 * z[t, 2]
        sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
        upd = inside & (zz > sub_z)
        if upd.any():
            sub_z[upd] = zz[upd]
            img[y0:y1 + 1, x0:x1 + 1][upd] = rgb[t]


def _fill_disc(img, zbuf, cx, cy, r, z, rgb) -> None:
    H, W = zbuf.shape
    x0, x1 = max(int(np.floor(cx - r)), 0), min(int(np.ceil(cx + r)), W - 1)
    y0, y1 = max(int(np.floor(cy - r)), 0), min(int(np.ceil(cy + r)), H - 1)
    if x0 > x1 or y0 > y1:
        return
    X, Y = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
    inside = (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    sub_z = zbuf[y0:y1 + 1, x0:x1 + 1]
    upd = inside & (z > sub_z)
    if upd.any():
        sub_z[upd] = z
        img[y0:y1 + 1, x0:x1 + 1][upd] = rgb


# ---------------------------------------------------------------------------
# Per-kind drawing
# ---------------------------------------------------------------------------

def _draw_polyhedron(img, zbuf, cam, tris, rgb, ambient):
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1)
    keep = ln > 1e-12
    tris, n, ln = tris[keep], n[keep], ln[keep]
    n = n / ln[:, None]
    facing = n @ cam.c
    n = n * np.where(facing < 0, -1.0, 1.0)[:, None]      # two-sided shading
    light = cam.c + 0.6 * cam.right + 0.8 * cam.up
    light /= np.linalg.norm(light)
    k = ambient + (1.0 - ambient) * np.clip(n @ light, 0.0, 1.0)
    colors = np.asarray(rgb, dtype=np.float64)[None, :] * k[:, None]

    sx, sy, z = cam.project(tris.reshape(-1, 3))
    xy = np.stack([sx, sy], axis=-1).reshape(-1, 3, 2)
    _fill_triangles(img, zbuf, xy, z.reshape(-1, 3), colors)


def _draw_polyline(img, zbuf, cam, pts, rgb, width_px):
    sx, sy, z = cam.project(pts)
    xy = np.stack([sx, sy], axis=-1)
    hw = width_px * cam.ssaa / 2.0
    quads_xy, quads_z = [], []
    for i in range(len(pts) - 1):
        d = xy[i + 1] - xy[i]
        L = float(np.hypot(*d))
        if L < 1e-9:
            continue
        nrm = np.array([-d[1], d[0]]) / L * hw
        v = np.array([xy[i] + nrm, xy[i] - nrm, xy[i + 1] - nrm, xy[i + 1] + nrm])
        zz = np.array([z[i], z[i], z[i + 1], z[i + 1]])
        quads_xy += [v[[0, 1, 2]], v[[0, 2, 3]]]
        quads_z += [zz[[0, 1, 2]], zz[[0, 2, 3]]]
    if quads_xy:
        colors = np.tile(np.asarray(rgb, dtype=np.float64), (len(quads_xy), 1))
        _fill_triangles(img, zbuf, np.array(quads_xy), np.array(quads_z), colors)
    for i in range(len(pts)):   # round joints
        _fill_disc(img, zbuf, sx[i], sy[i], hw, z[i], np.asarray(rgb, dtype=np.float64))


def _draw_point(img, zbuf, cam, p, rgb, radius_px):
    sx, sy, z = cam.project(np.asarray(p, dtype=np.float64).reshape(1, 3))
    _fill_disc(img, zbuf, sx[0], sy[0], radius_px * cam.ssaa, z[0],
               np.asarray(rgb, dtype=np.float64))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_snapshot(
    geom: np.ndarray,
    kind: str,
    elev: float,
    azim: float,
    image_px: int = 256,
    view_extent: Optional[float] = None,
    ssaa: int = 2,
    background: Tuple[int, int, int] = (255, 255, 255),
    color: Optional[Tuple[int, int, int]] = None,
    line_width_px: float = 3.0,
    point_radius_px: float = 4.0,
    ambient: float = 0.35,
) -> np.ndarray:
    """Render one geometry. Returns uint8 array (image_px, image_px, 3)."""
    cam = _Camera(elev, azim, image_px, ssaa, view_extent)
    img = np.empty((cam.S, cam.S, 3), dtype=np.float32)
    img[:] = np.asarray(background, dtype=np.float32)
    zbuf = np.full((cam.S, cam.S), -np.inf)
    rgb = color if color is not None else DEFAULT_COLORS[kind]

    if kind == "polyhedra":
        _draw_polyhedron(img, zbuf, cam, np.asarray(geom, dtype=np.float64), rgb, ambient)
    elif kind == "polylines":
        _draw_polyline(img, zbuf, cam, np.asarray(geom, dtype=np.float64), rgb, line_width_px)
    elif kind == "points":
        _draw_point(img, zbuf, cam, geom, rgb, point_radius_px)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if ssaa > 1:
        img = img.reshape(image_px, ssaa, image_px, ssaa, 3).mean(axis=(1, 3))
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


ViewSpec = Union[str, Tuple[float, float]]


def render_views(geom: np.ndarray, kind: str, views: Iterable[ViewSpec] = DEFAULT_VIEWS,
                 **kwargs) -> Dict[str, np.ndarray]:
    """Render several views. A view is a preset name or an (elev, azim) tuple."""
    out = {}
    for v in views:
        if isinstance(v, str):
            name, (elev, azim) = v, VIEW_PRESETS[v]
        else:
            elev, azim = v
            name = f"e{elev:g}_a{azim:g}"
        out[name] = render_snapshot(geom, kind, elev, azim, **kwargs)
    return out


def save_png(path: str, img: np.ndarray) -> None:
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        import matplotlib.image as mpimg
        mpimg.imsave(path, img)


def montage(rows: Sequence[Sequence[np.ndarray]]) -> np.ndarray:
    """Tile images into a contact sheet (rows x cols)."""
    return np.concatenate([np.concatenate(list(r), axis=1) for r in rows], axis=0)


# ---------------------------------------------------------------------------
# CLI: quick contact sheet (rows = samples, columns = views)
# ---------------------------------------------------------------------------

def main() -> None:
    from generate_geometry_3d import KINDS, COMPLEXITIES, generate_batch

    ap = argparse.ArgumentParser(description="Render a preview contact sheet")
    ap.add_argument("--kind", choices=KINDS, default="polyhedra")
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--complexity", choices=COMPLEXITIES + ("mixed",), default="mid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS),
                    choices=list(VIEW_PRESETS))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--view-extent", type=float, default=None)
    ap.add_argument("--out", default="preview.png")
    args = ap.parse_args()

    batch = generate_batch(args.kind, args.n, args.complexity, args.seed)
    rows = [list(render_views(batch.get(i), args.kind, args.views, image_px=args.size,
                                 view_extent=args.view_extent).values())
            for i in range(len(batch))]
    save_png(args.out, montage(rows))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
