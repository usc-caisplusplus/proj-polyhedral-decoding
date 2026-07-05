"""Deterministic checks for ``convert_geo2vec_distance.py`` helpers."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from shapely.geometry import MultiPolygon, Point, Polygon

from convert_geo2vec_distance import (
    SQLITE_HEADER,
    coerce_simple_geometry,
    create_split_indices,
    normalized_geometry,
    padded_tensor,
    require_geopackage,
    sample_pair_offsets,
)


def test_geopackage_signature() -> None:
    with TemporaryDirectory() as directory:
        valid = Path(directory) / "valid.gpkg"
        invalid = Path(directory) / "checkpoint.gpkg"
        valid.write_bytes(SQLITE_HEADER + b"placeholder")
        invalid.write_bytes(b"PK\x03\x04torch checkpoint")
        require_geopackage(valid)
        try:
            require_geopackage(invalid)
        except ValueError as error:
            assert "model checkpoint" in str(error)
        else:
            raise AssertionError("A checkpoint mislabeled as .gpkg was accepted")


def test_geometry_conversion_and_normalization() -> None:
    small = Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    large_with_hole = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
        holes=[[(1, 1), (2, 1), (2, 2), (1, 1)]],
    )
    converted = coerce_simple_geometry(
        MultiPolygon([small, large_with_hole]),
        "polygon",
    )
    assert converted is not None
    assert len(converted.interiors) == 0
    assert converted.area == 16

    point = normalized_geometry(Point(15, 35), (10, 20, 20, 40))
    np.testing.assert_allclose([point.x, point.y], [0.0, 0.5], atol=1e-7)


def test_pairs_tensors_and_splits() -> None:
    first = sample_pair_offsets(20, 20, 100, True, seed=2024)
    second = sample_pair_offsets(20, 20, 100, True, seed=2024)
    assert first == second
    assert len(first) == len(set(first)) == 100
    assert all(left < right for left, right in first)

    arrays = [
        np.asarray([[0, 0], [1, 1]], dtype=np.float32),
        np.asarray([[2, 2], [3, 3], [4, 4]], dtype=np.float32),
    ]
    values, lengths = padded_tensor(arrays, "polyline")
    assert values.shape == (2, 3, 2)
    assert lengths.tolist() == [2, 3]
    assert torch.equal(values[0, 2], torch.zeros(2))

    splits = create_split_indices(100, seed=2024)
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 60,
        "val": 20,
        "test": 20,
    }
    combined = torch.cat(list(splits.values()))
    assert sorted(combined.tolist()) == list(range(100))


if __name__ == "__main__":
    test_geopackage_signature()
    test_geometry_conversion_and_normalization()
    test_pairs_tensors_and_splits()
    print("convert_geo2vec_distance checks passed")
