import torch
import triangle
from shapely import Polygon
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.moments_compute import (
    moment_of_point_monomial, moment_of_point_legendre,
    moment_of_line_segment_monomial, moment_of_line_segment_legendre,
    moment_of_polygon_monomial, moment_of_polygon_legendre,
    moment_of_point_fourier, moment_of_line_segment_fourier,
    moment_of_polygon_fourier
)


class GeometryMomentEncoder():

    def __init__(self, args, device='cpu'):
        self.device = torch.device(device)
        self.basis_type = args.moment_params.get('basis_type', 'monomial')  # 'monomial' or 'legendre'

        # Create integer meshgrid for (m, n) pairs
        self.M, self.N = self.create_moment_meshgrid(args.moment_params)
        # Store as numpy arrays since moment computations use numpy
        self.M_np = self.M.cpu().numpy()
        self.N_np = self.N.cpu().numpy()

        # Get moment computation functions based on basis type
        if self.basis_type == 'monomial':
            self.point_moment_fn = moment_of_point_monomial
            self.line_moment_fn = moment_of_line_segment_monomial
            self.polygon_moment_fn = moment_of_polygon_monomial
        elif self.basis_type == 'legendre':
            self.point_moment_fn = moment_of_point_legendre
            self.line_moment_fn = moment_of_line_segment_legendre
            self.polygon_moment_fn = moment_of_polygon_legendre
        elif self.basis_type == 'fourier':
            self.point_moment_fn = moment_of_point_fourier
            self.line_moment_fn = moment_of_line_segment_fourier
            self.polygon_moment_fn = moment_of_polygon_fourier
        else:
            raise ValueError(f"Unsupported basis type: {self.basis_type}")

    def create_moment_meshgrid(self, params: dict):
        """
        Create a meshgrid of integer indices (m, n) for moment computation.
        """
        max_order = params.get('max_order', 10)

        # Create indices from 0 to max_order
        m_indices = torch.arange(0, max_order + 1, dtype=torch.long)
        n_indices = torch.arange(0, max_order + 1, dtype=torch.long)

        # Create meshgrid
        M, N = torch.meshgrid(m_indices, n_indices, indexing='ij')

        return M, N

    @property
    def _moment_dtype(self):
        return torch.complex64 if self.basis_type == 'fourier' else torch.float32

    def encode(self, geom_dataset: torch.Tensor, lengths: torch.Tensor, dataset_type: str):
        geom_dataset = geom_dataset.to(self.device)

        if dataset_type == "points":
            return self.point_encoder(geom_dataset)

        elif dataset_type == "lines":
            return self.line_encoder(geom_dataset)

        elif dataset_type == "polylines":
            return self.polyline_encoder(geom_dataset, lengths)

        elif dataset_type == "polygons":
            return self.polygon_encoder(geom_dataset, lengths)

        else:
            raise ValueError(f"Invalid input dataset type: {dataset_type}")

    def point_encoder(self, points: torch.Tensor):
        n_geoms = points.shape[0]
        moment_shape = self.M.shape
        moments = torch.zeros((n_geoms, *moment_shape), dtype=self._moment_dtype, device=self.device)

        points_np = points.cpu().numpy()

        for i in range(n_geoms):
            x, y = float(points_np[i, 0]), float(points_np[i, 1])

            for idx_m in range(moment_shape[0]):
                for idx_n in range(moment_shape[1]):
                    m = int(self.M_np[idx_m, idx_n])
                    n = int(self.N_np[idx_m, idx_n])
                    moment_val = self.point_moment_fn(x, y, m, n)
                    moments[i, idx_m, idx_n] = moment_val

        return moments

    def line_encoder(self, lines: torch.Tensor):
        """
        Encode line segment dataset.

        Args:
            lines: Tensor of shape (n_geoms, 2, 2) where first dim is number of segments,
                   second dim is 2 endpoints, third dim is (x, y) coordinates

        Returns:
            Tensor of shape (n_geoms, n_m, n_n) containing moment values
        """
        n_geoms = lines.shape[0]
        moment_shape = self.M.shape
        moments = torch.zeros((n_geoms, *moment_shape), dtype=self._moment_dtype, device=self.device)

        lines_np = lines.cpu().numpy()

        for i in range(n_geoms):
            ax, ay = float(lines_np[i, 0, 0]), float(lines_np[i, 0, 1])
            bx, by = float(lines_np[i, 1, 0]), float(lines_np[i, 1, 1])

            for idx_m in range(moment_shape[0]):
                for idx_n in range(moment_shape[1]):
                    m = int(self.M_np[idx_m, idx_n])
                    n = int(self.N_np[idx_m, idx_n])
                    moment_val = self.line_moment_fn(ax, ay, bx, by, m, n)
                    moments[i, idx_m, idx_n] = moment_val

        return moments

    def polyline_encoder(self, polylines: torch.Tensor, lengths: torch.Tensor):
        """
        Encode polyline dataset by summing moments of constituent line segments.

        Args:
            polylines: Tensor of shape (n_geoms, M, 2) where M is max number of points
            lengths: Tensor of shape (n_geoms,) containing actual length of each polyline

        Returns:
            Tensor of shape (n_geoms, n_m, n_n) containing moment values
        """
        n_geoms = polylines.shape[0]
        moment_shape = self.M.shape
        moments = torch.zeros((n_geoms, *moment_shape), dtype=self._moment_dtype, device=self.device)

        polylines_np = polylines.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

        for geom_idx in range(n_geoms):
            actual_length = int(lengths_np[geom_idx])
            if actual_length < 2:  # Need at least 2 points to form a line
                continue

            # Sum moments over all line segments in the polyline
            for seg_idx in range(actual_length - 1):
                ax, ay = float(polylines_np[geom_idx, seg_idx, 0]), float(polylines_np[geom_idx, seg_idx, 1])
                bx, by = float(polylines_np[geom_idx, seg_idx + 1, 0]), float(polylines_np[geom_idx, seg_idx + 1, 1])

                # Compute moments for all (m, n) pairs
                for idx_m in range(moment_shape[0]):
                    for idx_n in range(moment_shape[1]):
                        m = int(self.M_np[idx_m, idx_n])
                        n = int(self.N_np[idx_m, idx_n])
                        moment_val = self.line_moment_fn(ax, ay, bx, by, m, n)
                        moments[geom_idx, idx_m, idx_n] += moment_val

        return moments

    def polygon_encoder(self, polygons: torch.Tensor, lengths: torch.Tensor, hole=None):
        """
        Encode polygon dataset using area moments.

        Args:
            polygons: Tensor of shape (n_geoms, M, 2) where M is max number of vertices
            lengths: Tensor of shape (n_geoms,) containing actual number of vertices
            hole: Optional holes in polygons (not currently supported)

        Returns:
            Tensor of shape (n_geoms, n_m, n_n) containing moment values
        """
        n_geoms = polygons.shape[0]
        moment_shape = self.M.shape
        moments = torch.zeros((n_geoms, *moment_shape), dtype=self._moment_dtype, device=self.device)

        polygons_np = polygons.cpu().numpy()
        lengths_np = lengths.cpu().numpy()

        for geom_idx in range(n_geoms):
            actual_length = int(lengths_np[geom_idx])
            if actual_length < 3:  # Need at least 3 points to form a polygon
                continue

            # Extract vertices for this polygon
            vertices = polygons_np[geom_idx, :actual_length, :]

            # Preprocess polygon to handle self-intersections
            vertices = self.preprocess_polygon(vertices)

            # Compute moments for all (m, n) pairs
            for idx_m in range(moment_shape[0]):
                for idx_n in range(moment_shape[1]):
                    m = int(self.M_np[idx_m, idx_n])
                    n = int(self.N_np[idx_m, idx_n])
                    moment_val = self.polygon_moment_fn(vertices, m, n)
                    moments[geom_idx, idx_m, idx_n] = moment_val

        return moments

    def preprocess_polygon(self, polygon_coords):
        """
        Preprocess polygon using Shapely to handle self-intersections and collinear points.

        Args:
            polygon_coords: Numpy array of shape (K, 2) containing vertex coordinates

        Returns:
            Numpy array of preprocessed vertices
        """
        # Create a Shapely polygon
        polygon = Polygon(polygon_coords)

        # Fix any self-intersections and simplify the polygon
        fixed_polygon = polygon.buffer(0)  # Buffering with 0 fixes self-intersections

        # Extract exterior coordinates and return as numpy array
        exterior_coords = np.array(list(fixed_polygon.exterior.coords)[:-1])  # Remove duplicate closing vertex

        return exterior_coords
