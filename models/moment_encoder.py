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
        if self.basis_type == 'monomial':
            return batch_moments_points_monomial(points_np, self.tables)
        elif self.basis_type == 'fourier':
            return batch_moments_points_fourier(points_np, self.tables)
        elif self.basis_type == 'legendre':
            return batch_moments_points_legendre(points_np, self.tables)

    def _encode_polygons(self, polygons_np: np.ndarray, lengths_np: np.ndarray) -> np.ndarray:
        B, max_K = polygons_np.shape[0], polygons_np.shape[1]
        processed = np.zeros_like(polygons_np)
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

        if self.basis_type == 'monomial':
            return batch_moments_polygons_monomial(processed, new_lengths, self.tables)
        elif self.basis_type == 'fourier':
            return batch_moments_polygons_fourier(processed, new_lengths, self.tables)
        elif self.basis_type == 'legendre':
            return batch_moments_polygons_legendre(processed, new_lengths, self.tables)

    def preprocess_polygon(self, polygon_coords):
        polygon = Polygon(polygon_coords)
        fixed_polygon = polygon.buffer(0)
        exterior_coords = np.array(list(fixed_polygon.exterior.coords)[:-1])
        return exterior_coords
