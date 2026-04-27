import torch
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.moments_compute_3d_vectorized import (
    MomentTables3D,
    batch_encode_3d,
)


class GeometryMomentEncoder3D:
    """
    3D moment encoder analogous to GeometryMomentEncoder (2D).

    Supported dataset_type values:
        "points"     — (B, 3) or (B, 1, 3)
        "polylines"  — (B, maxS, 3) with lengths (B,)
        "polygons"   — flat 3D polygons, (B, maxK, 3) with lengths (B,)
        "polyhedra"  — triangulated meshes, (B, maxF, 3, 3) with face_counts (B,)

    The encoder returns a torch.Tensor of shape (B, O, O, O) where O = max_order + 1.
    """

    def __init__(self, args, device='cpu'):
        self.device = torch.device(device)
        self.basis_type = args.moment_params.get('basis_type', 'legendre')
        max_order = args.moment_params.get('max_order', 5)
        self.tables = MomentTables3D(max_order)

    @property
    def _moment_dtype(self):
        return torch.complex64 if self.basis_type == 'fourier' else torch.float32

    def encode(self, geom: torch.Tensor, lengths: torch.Tensor,
               dataset_type: str, face_counts: torch.Tensor = None) -> torch.Tensor:
        """
        Encode a batch of 3D geometries into moment tensors.

        Args:
            geom:         (B, ..., 3) coordinate tensor
            lengths:      (B,) actual point/vertex counts (ignored for points)
            dataset_type: one of "points", "polylines", "polygons", "polyhedra"
            face_counts:  (B,) number of triangular faces — required for polyhedra

        Returns:
            torch.Tensor of shape (B, O, O, O), dtype float32 or complex64
        """
        geom_np = geom.cpu().numpy().astype(np.float64)
        lengths_np = lengths.cpu().numpy().astype(int) if lengths is not None else None

        if dataset_type == 'points':
            # accept (B, 1, 3) or (B, 3)
            if geom_np.ndim == 3:
                geom_np = geom_np[:, 0, :]
            result = batch_encode_3d(geom_np, lengths_np, 'points', self.tables,
                                     basis=self.basis_type)

        elif dataset_type == 'polylines':
            result = batch_encode_3d(geom_np, lengths_np, 'polylines', self.tables,
                                     basis=self.basis_type)

        elif dataset_type == 'polygons':
            result = batch_encode_3d(geom_np, lengths_np, 'polygons', self.tables,
                                     basis=self.basis_type)

        elif dataset_type == 'polyhedra':
            if face_counts is None:
                raise ValueError("face_counts required for polyhedra dataset_type")
            fc_np = face_counts.cpu().numpy().astype(int)
            result = batch_encode_3d(geom_np, lengths_np, 'polyhedra', self.tables,
                                     basis=self.basis_type, face_counts=fc_np)

        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")

        return torch.tensor(result, dtype=self._moment_dtype, device=self.device)
