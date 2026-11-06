"""
Build an image dataset: random geometries -> snapshots (PNG) + geometry arrays,
optionally with moment labels computed by moments_compute_3d_vectorized.py.

    python build_dataset.py --kind polyhedra -n 1000 --complexity mixed \
        --views iso iso_back top front --image-size 256 --out-dir data/polyhedra \
        --max-order 4 --basis legendre

Output layout:
    out_dir/images/000000_iso.png ...   one PNG per sample per view
    out_dir/geometry.npz                kind, geom (padded), lengths  (GeomBatch.load)
    out_dir/moments_<basis>_o<K>.npy    (N, O, O, O) labels, if --max-order is given
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from generate_geometry_3d import COMPLEXITIES, KINDS, POLYHEDRON_STYLES, generate_batch
from snapshot_renderer_3d import VIEW_PRESETS, DEFAULT_VIEWS, render_snapshot, save_png


def compute_moments(batch, max_order: int, basis: str, chunk: int) -> np.ndarray:
    from moments_compute_3d_vectorized import MomentTables3D, batch_encode_3d

    tables = MomentTables3D(max_order)
    parts = []
    for s in range(0, len(batch), chunk):   # chunked: polyhedron tensors get large
        sub = batch.slice(s, s + chunk)
        parts.append(batch_encode_3d(tables=tables, basis=basis, **sub.moment_kwargs()))
        print(f"  moments {min(s + chunk, len(batch))}/{len(batch)}", file=sys.stderr)
    return np.concatenate(parts, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=KINDS, default="polyhedra")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--complexity", choices=COMPLEXITIES + ("mixed",), default="mid")
    ap.add_argument("--polyhedron-style", choices=POLYHEDRON_STYLES, default="mixed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS), choices=list(VIEW_PRESETS))
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--view-extent", type=float, default=None,
                    help="Half-width of the view in world units. Default: fit the whole "
                         "[-1,1]^3 cube per view (nothing clipped). Smaller = zoom in.")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--max-order", type=int, default=None,
                    help="If set, also compute moment labels up to this order")
    ap.add_argument("--basis", choices=["monomial", "fourier", "legendre"], default="legendre")
    ap.add_argument("--moment-batch", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    batch = generate_batch(args.kind, args.n, args.complexity, args.seed, args.polyhedron_style)
    for i in range(len(batch)):
        geom = batch.get(i)
        for view in args.views:
            elev, azim = VIEW_PRESETS[view]
            img = render_snapshot(geom, args.kind, elev, azim, image_px=args.image_size,
                                   view_extent=args.view_extent)
            save_png(str(out / "images" / f"{i:06d}_{view}.png"), img)
        if (i + 1) % 100 == 0 or i + 1 == len(batch):
            print(f"  rendered {i + 1}/{len(batch)}", file=sys.stderr)

    batch.save(str(out / "geometry.npz"))

    if args.max_order is not None:
        m = compute_moments(batch, args.max_order, args.basis, args.moment_batch)
        np.save(out / f"moments_{args.basis}_o{args.max_order}.npy", m)
        print(f"moments {m.shape} {m.dtype}")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
