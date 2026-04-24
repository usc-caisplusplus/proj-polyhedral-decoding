"""
3D spatial dataset generator package.

Mirrors the 2D `generator/` package, lifted to 3D with polyhedron (extruded
prism) as the solid primitive. No trimesh dependency for the MVP — 2D Shapely
handles footprints, z-ranges are 1D, and topology/distance are exact for prisms.
"""

from . import geometry_generators_3d
from . import labels_3d
from . import normalization_3d
from . import io_utils_3d
from . import topology_sampler_3d
from . import hard_cases_3d

__all__ = [
    "geometry_generators_3d",
    "labels_3d",
    "normalization_3d",
    "io_utils_3d",
    "topology_sampler_3d",
    "hard_cases_3d",
]
