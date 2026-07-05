import torch
from shapely import Polygon
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.moments_compute_vectorized import (
    MomentTables,
    batch_moments_points_monomial, batch_moments_points_fourier, batch_moments_points_legendre,
    batch_moments_polylines,
    batch_moments_polygons_monomial, batch_moments_polygons_fourier, batch_moments_polygons_legendre,
)


class GeometryMomentEncoder():

    def __init__(self, args, device='cpu'):
        self.device = torch.device(device)
        self.basis_type = args.moment_params.get('basis_type', 'monomial')
        max_order = args.moment_params.get('max_order', 10)
        self.tables = MomentTables(max_order)

    @property
    def _moment_dtype(self):
        return torch.complex64 if self.basis_type == 'fourier' else torch.float32

    def encode(self, geom_dataset: torch.Tensor, lengths: torch.Tensor, dataset_type: str):
        geom_np = geom_dataset.cpu().numpy()
        lengths_np = lengths.cpu().numpy() if lengths is not None else None

        if dataset_type == "points":
            result = self._encode_points(geom_np)
        elif dataset_type == "lines":
            # Treat each line (B, 2, 2) as a 2-point polyline
            result = batch_moments_polylines(
                geom_np, np.full(geom_np.shape[0], 2, dtype=int), self.tables, basis=self.basis_type
            )
        elif dataset_type == "polylines":
            result = batch_moments_polylines(geom_np, lengths_np, self.tables, basis=self.basis_type)
        elif dataset_type == "polygons":
            result = self._encode_polygons(geom_np, lengths_np)
        else:
            raise ValueError(f"Invalid input dataset type: {dataset_type}")

        return torch.tensor(result, dtype=self._moment_dtype, device=self.device)

    def _encode_points(self, points_np: np.ndarray) -> np.ndarray:
        # points may arrive as (B, 1, 2) from the padded dataloader — squeeze to (B, 2)
        if points_np.ndim == 3:
            points_np = points_np[:, 0, :]
        if self.basis_type == 'monomial':
            return batch_moments_points_monomial(points_np, self.tables)
        elif self.basis_type == 'fourier':
            return batch_moments_points_fourier(points_np, self.tables)
        elif self.basis_type == 'legendre':
            return batch_moments_points_legendre(points_np, self.tables)

    def _encode_polygons(self, polygons_np: np.ndarray, lengths_np: np.ndarray) -> np.ndarray:
        B, max_K = polygons_np.shape[0], polygons_np.shape[1]
        processed = np.zeros_like(polygons_np)
        # signs[b, k] = +1 for exterior vertices, -1 for hole vertices, 0 for padding
        signs = np.zeros((B, max_K), dtype=np.float64)
        new_lengths = np.zeros(B, dtype=int)

        for b in range(B):
            K = int(lengths_np[b])
            if K < 3:
                new_lengths[b] = K
                continue
            verts = self.preprocess_polygon(polygons_np[b, :K])
            n = min(len(verts), max_K)
            processed[b, :n] = verts[:n]
            new_lengths[b] = n

            # Build per-vertex sign array: +1 exterior, -1 holes (NaN rows are separators)
            sign = 1.0
            for k in range(n):
                if np.isnan(verts[k, 0]):
                    sign = -1.0  # flip sign after each NaN separator
                    signs[b, k] = 0.0  # sentinel vertex itself has no contribution
                else:
                    signs[b, k] = sign

        if self.basis_type == 'monomial':
            return batch_moments_polygons_monomial(processed, new_lengths, self.tables, signs=signs)
        elif self.basis_type == 'fourier':
            return batch_moments_polygons_fourier(processed, new_lengths, self.tables, signs=signs)
        elif self.basis_type == 'legendre':
            return batch_moments_polygons_legendre(processed, new_lengths, self.tables, signs=signs)

    def preprocess_polygon(self, polygon_coords):
        """
        Returns a flat coordinate array encoding exterior and hole rings.

        Rings are separated by a [NaN, NaN] sentinel row.
        The exterior ring comes first (sign +1), followed by each interior
        ring (holes, sign -1), each preceded by a NaN sentinel.

        Layout:  [ext_verts..., NaN, hole0_verts..., NaN, hole1_verts..., ...]
        """
        polygon_coords = np.asarray(polygon_coords, dtype=np.float64)
        separator_indices = np.flatnonzero(np.isnan(polygon_coords).any(axis=1))
        if separator_indices.size:
            rings = []
            start = 0
            for stop in separator_indices:
                if stop - start >= 3:
                    rings.append(polygon_coords[start:stop])
                start = stop + 1
            if len(polygon_coords) - start >= 3:
                rings.append(polygon_coords[start:])
            if not rings:
                raise ValueError("Polygon input contains no valid rings")
            polygon = Polygon(rings[0], holes=rings[1:])
        else:
            polygon = Polygon(polygon_coords)

        fixed_polygon = polygon.buffer(0)

        # Handle MultiPolygon produced by buffer(0) — take the largest piece
        if fixed_polygon.geom_type == 'MultiPolygon':
            fixed_polygon = max(fixed_polygon.geoms, key=lambda g: g.area)
        if fixed_polygon.is_empty or fixed_polygon.geom_type != 'Polygon':
            raise ValueError("Polygon could not be repaired")

        def counterclockwise(coords):
            coords = np.asarray(coords, dtype=np.float64)
            x = coords[:, 0]
            y = coords[:, 1]
            signed_area_twice = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
            return coords if signed_area_twice >= 0 else coords[::-1].copy()

        # The vectorized routines explicitly subtract hole rings, so all rings
        # must use the same orientation.
        exterior_coords = counterclockwise(
            np.array(list(fixed_polygon.exterior.coords)[:-1])
        )
        rings = [exterior_coords]

        for interior in fixed_polygon.interiors:
            hole_coords = counterclockwise(
                np.array(list(interior.coords)[:-1])
            )
            rings.append(np.array([[np.nan, np.nan]]))
            rings.append(hole_coords)

        return np.concatenate(rings, axis=0)
