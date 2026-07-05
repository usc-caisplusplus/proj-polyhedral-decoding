"""
Convert a Geo2Vec entity GeoPackage into a Poly2Moment distance-pair dataset.

The output dataset is the five-tensor tuple expected by ``run.py``:

    (X1, X2, L1, L2, Y)

Coordinates are normalized once over the complete source layer, not separately
for each geometry type. Distance labels are then computed in that shared
normalized frame. A sidecar file stores deterministic 60/20/20 split indices
and the source-row positions used for every pair.

Example:

    python scripts/convert_geo2vec_distance.py \
        --input Geo2Vec/data/NYC_total_data.gpkg \
        --output data/Geo2VecNYC/point_polygon_dis_data.pt \
        --type1 point --type2 polygon --num-pairs 50000 --seed 2024
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)


GEOMETRY_TYPES = ("point", "polyline", "polygon")
SQLITE_HEADER = b"SQLite format 3\x00"


@dataclass(frozen=True)
class Entity:
    """One source geometry after conversion to a Poly2Moment-supported type."""

    source_position: int
    geometry: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic Poly2Moment distance-pair .pt file from "
            "a Geo2Vec GeoPackage."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--type1", required=True, choices=GEOMETRY_TYPES)
    parser.add_argument("--type2", required=True, choices=GEOMETRY_TYPES)
    parser.add_argument("--num-pairs", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--layer",
        default=None,
        help="GeoPackage layer name; by default GeoPandas reads the first layer",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("shared_bounds", "none"),
        default="shared_bounds",
        help=(
            "shared_bounds maps the complete source layer to [-1, 1]^2 before "
            "computing labels; none preserves source coordinates"
        ),
    )
    parser.add_argument(
        "--simplify-tolerance",
        type=float,
        default=0.0,
        help=(
            "Optional simplification tolerance applied after normalization. "
            "Zero preserves all vertices."
        ),
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=None,
        help="Defaults to <output stem>_splits.pt",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Defaults to <output stem>_metadata.json",
    )
    return parser.parse_args()


def require_geopackage(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Input does not exist: {path}")
    with path.open("rb") as handle:
        header = handle.read(len(SQLITE_HEADER))
    if header != SQLITE_HEADER:
        raise ValueError(
            f"{path} is not a valid SQLite GeoPackage. Some files in the "
            "Geo2Vec bundle use a .gpkg suffix for a torch model checkpoint; "
            "those files contain weights rather than source geometries."
        )


def read_geometries(path: Path, layer: str | None) -> Any:
    require_geopackage(path)
    try:
        import geopandas as gpd
    except ImportError as error:
        raise RuntimeError(
            "GeoPandas is required for conversion. Install the repository "
            "requirements, including geopandas."
        ) from error

    frame = gpd.read_file(path, layer=layer)
    if frame.empty:
        raise ValueError(f"No rows found in {path}")
    if frame.geometry.name not in frame:
        raise ValueError(f"No active geometry column found in {path}")
    return frame


def geometry_family(geometry: Any) -> str | None:
    if isinstance(geometry, (Point, MultiPoint)):
        return "point"
    if isinstance(geometry, (LineString, MultiLineString)):
        return "polyline"
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return "polygon"
    return None


def _collection_candidates(geometry: Any, geometry_type: str) -> list[Any]:
    if not isinstance(geometry, GeometryCollection):
        return []
    return [
        part
        for part in geometry.geoms
        if geometry_family(part) == geometry_type
    ]


def coerce_simple_geometry(geometry: Any, geometry_type: str) -> Any | None:
    """
    Convert multipart entities to the largest Poly2Moment-supported component.

    The current 2D tensor format represents one point, one coordinate sequence,
    or one polygon exterior. Polygon holes are removed consistently before both
    label computation and tensorization.
    """
    if geometry is None or geometry.is_empty:
        return None

    if isinstance(geometry, GeometryCollection):
        candidates = _collection_candidates(geometry, geometry_type)
        if not candidates:
            return None
        geometry = max(
            candidates,
            key=lambda item: item.area
            if geometry_type == "polygon"
            else item.length,
        )

    if geometry_type == "point":
        if isinstance(geometry, MultiPoint):
            geometry = next(iter(geometry.geoms), None)
        if not isinstance(geometry, Point) or geometry is None:
            return None
        return Point(float(geometry.x), float(geometry.y))

    if geometry_type == "polyline":
        if isinstance(geometry, MultiLineString):
            geometry = max(geometry.geoms, key=lambda item: item.length)
        if not isinstance(geometry, LineString) or geometry.is_empty:
            return None
        coordinates = [(float(c[0]), float(c[1])) for c in geometry.coords]
        if len(coordinates) < 2:
            return None
        result = LineString(coordinates)
        return result if result.length > 0 else None

    if geometry_type == "polygon":
        if isinstance(geometry, MultiPolygon):
            geometry = max(geometry.geoms, key=lambda item: item.area)
        if not isinstance(geometry, Polygon) or geometry.is_empty:
            return None
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
            if isinstance(geometry, MultiPolygon):
                geometry = max(geometry.geoms, key=lambda item: item.area)
        if not isinstance(geometry, Polygon) or geometry.is_empty:
            return None
        # Poly2Moment's pair tensor contains one exterior sequence. Strip holes
        # here so the geometry used for labels exactly matches the tensor.
        result = Polygon([(float(c[0]), float(c[1])) for c in geometry.exterior.coords])
        return result if result.is_valid and result.area > 0 else None

    raise ValueError(f"Unsupported geometry type: {geometry_type}")


def normalized_geometry(
    geometry: Any,
    bounds: tuple[float, float, float, float],
) -> Any:
    min_x, min_y, max_x, max_y = bounds
    width = max_x - min_x
    height = max_y - min_y
    if width <= 0 or height <= 0:
        raise ValueError(f"Cannot normalize degenerate dataset bounds: {bounds}")
    x_scale = 2.0 / width
    y_scale = 2.0 / height
    return affinity.affine_transform(
        geometry,
        [x_scale, 0.0, 0.0, y_scale, -1.0 - x_scale * min_x, -1.0 - y_scale * min_y],
    )


def build_entity_pool(
    frame: Any,
    geometry_type: str,
    coordinate_mode: str,
    bounds: tuple[float, float, float, float],
    simplify_tolerance: float,
) -> list[Entity]:
    entities: list[Entity] = []
    for source_position, geometry in enumerate(frame.geometry):
        simple = coerce_simple_geometry(geometry, geometry_type)
        if simple is None:
            continue
        if coordinate_mode == "shared_bounds":
            simple = normalized_geometry(simple, bounds)
        if simplify_tolerance > 0 and geometry_type != "point":
            simple = simple.simplify(
                simplify_tolerance,
                preserve_topology=geometry_type == "polygon",
            )
            simple = coerce_simple_geometry(simple, geometry_type)
            if simple is None:
                continue
        entities.append(Entity(source_position, simple))
    return entities


def maximum_unique_pairs(size1: int, size2: int, same_pool: bool) -> int:
    if same_pool:
        return size1 * (size1 - 1) // 2
    return size1 * size2


def sample_pair_offsets(
    size1: int,
    size2: int,
    num_pairs: int,
    same_pool: bool,
    seed: int,
) -> list[tuple[int, int]]:
    if num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    available = maximum_unique_pairs(size1, size2, same_pool)
    if num_pairs > available:
        raise ValueError(
            f"Requested {num_pairs} unique pairs, but only {available} are "
            f"available from pools of size {size1} and {size2}"
        )

    rng = random.Random(seed)
    selected: set[tuple[int, int]] = set()
    while len(selected) < num_pairs:
        first = rng.randrange(size1)
        second = rng.randrange(size2)
        if same_pool:
            if first == second:
                continue
            if first > second:
                first, second = second, first
        selected.add((first, second))
    # Sorting removes dependence on set iteration order across Python builds.
    pairs = sorted(selected)
    rng.shuffle(pairs)
    return pairs


def geometry_coordinates(geometry: Any, geometry_type: str) -> np.ndarray:
    if geometry_type == "point":
        return np.asarray([geometry.x, geometry.y], dtype=np.float32)
    if geometry_type == "polyline":
        values = [(coordinate[0], coordinate[1]) for coordinate in geometry.coords]
    elif geometry_type == "polygon":
        values = [
            (coordinate[0], coordinate[1])
            for coordinate in geometry.exterior.coords
        ]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry_type}")
    return np.asarray(values, dtype=np.float32)


def padded_tensor(
    coordinate_arrays: list[np.ndarray],
    geometry_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if geometry_type == "point":
        values = torch.from_numpy(np.stack(coordinate_arrays)).float()
        lengths = torch.full((len(coordinate_arrays),), 2, dtype=torch.long)
        return values, lengths

    lengths = torch.tensor(
        [len(coordinates) for coordinates in coordinate_arrays],
        dtype=torch.long,
    )
    max_length = int(lengths.max().item())
    values = torch.zeros(
        (len(coordinate_arrays), max_length, 2),
        dtype=torch.float32,
    )
    for index, coordinates in enumerate(coordinate_arrays):
        values[index, : len(coordinates)] = torch.from_numpy(coordinates)
    return values, lengths


def create_split_indices(num_pairs: int, seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_pairs, generator=generator)
    train_end = int(0.6 * num_pairs)
    validation_end = train_end + int(0.2 * num_pairs)
    return {
        "train": permutation[:train_end],
        "val": permutation[train_end:validation_end],
        "test": permutation[validation_end:],
    }


def sidecar_path(output: Path, suffix: str) -> Path:
    return output.with_name(f"{output.stem}{suffix}")


def finite_bounds(frame: Any) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in frame.total_bounds)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"Source layer has invalid total bounds: {values}")
    return values


def main() -> None:
    args = parse_args()
    if args.simplify_tolerance < 0:
        raise ValueError("--simplify-tolerance cannot be negative")

    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    splits_path = (
        args.splits_output.expanduser().resolve()
        if args.splits_output is not None
        else sidecar_path(output_path, "_splits.pt")
    )
    metadata_path = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output is not None
        else sidecar_path(output_path, "_metadata.json")
    )

    frame = read_geometries(input_path, args.layer)
    bounds = finite_bounds(frame)
    pool1 = build_entity_pool(
        frame,
        args.type1,
        args.coordinate_mode,
        bounds,
        args.simplify_tolerance,
    )
    pool2 = (
        pool1
        if args.type1 == args.type2
        else build_entity_pool(
            frame,
            args.type2,
            args.coordinate_mode,
            bounds,
            args.simplify_tolerance,
        )
    )
    if not pool1:
        raise ValueError(f"No usable {args.type1} geometries found in {input_path}")
    if not pool2:
        raise ValueError(f"No usable {args.type2} geometries found in {input_path}")

    pair_offsets = sample_pair_offsets(
        len(pool1),
        len(pool2),
        args.num_pairs,
        same_pool=pool1 is pool2,
        seed=args.seed,
    )
    coordinates1: list[np.ndarray] = []
    coordinates2: list[np.ndarray] = []
    labels: list[float] = []
    source_pairs: list[tuple[int, int]] = []
    for first_offset, second_offset in pair_offsets:
        first = pool1[first_offset]
        second = pool2[second_offset]
        coordinates1.append(geometry_coordinates(first.geometry, args.type1))
        coordinates2.append(geometry_coordinates(second.geometry, args.type2))
        labels.append(float(first.geometry.distance(second.geometry)))
        source_pairs.append((first.source_position, second.source_position))

    x1, l1 = padded_tensor(coordinates1, args.type1)
    x2, l2 = padded_tensor(coordinates2, args.type2)
    y = torch.tensor(labels, dtype=torch.float32)
    if not torch.isfinite(y).all():
        raise RuntimeError("Generated distance labels contain NaN or infinity")

    split_payload = create_split_indices(args.num_pairs, args.seed)
    split_payload["pair_source_positions"] = torch.tensor(
        source_pairs,
        dtype=torch.long,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((x1, x2, l1, l2, y), output_path)
    torch.save(split_payload, splits_path)

    metadata = {
        "source": str(input_path),
        "layer": args.layer,
        "output": str(output_path),
        "splits_output": str(splits_path),
        "type1": args.type1,
        "type2": args.type2,
        "num_source_rows": len(frame),
        "num_type1_entities": len(pool1),
        "num_type2_entities": len(pool2),
        "num_pairs": args.num_pairs,
        "seed": args.seed,
        "coordinate_mode": args.coordinate_mode,
        "source_bounds": list(bounds),
        "simplify_tolerance": args.simplify_tolerance,
        "distance_min": float(y.min().item()),
        "distance_max": float(y.max().item()),
        "distance_mean": float(y.mean().item()),
        "x1_shape": list(x1.shape),
        "x2_shape": list(x2.shape),
        "split_sizes": {
            name: len(indices)
            for name, indices in split_payload.items()
            if name in ("train", "val", "test")
        },
        "conversion_notes": [
            "Multipart geometries are reduced to their largest component.",
            "Polygon holes are removed before tensorization and label computation.",
            "Distances are measured after the configured coordinate transformation.",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"Saved {args.num_pairs} {args.type1}-{args.type2} pairs to {output_path}"
    )
    print(f"Saved deterministic splits to {splits_path}")
    print(f"Saved conversion metadata to {metadata_path}")
    print(
        f"Shapes: X1={tuple(x1.shape)}, X2={tuple(x2.shape)}, "
        f"Y={tuple(y.shape)}; distance range=[{y.min():.6f}, {y.max():.6f}]"
    )


if __name__ == "__main__":
    main()
