"""Example usage of the moment-based encoder (run from repo root: ``python scripts/example_moment_encoder.py``)."""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import torch
import numpy as np
from types import SimpleNamespace


def example_point_encoding():
    """Example: Encode a batch of points using moment encoder."""
    print("\n" + "="*60)
    print("Example 1: Point Encoding")
    print("="*60)

    # Configure moment encoder
    args = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 5}
    )

    # Import encoder
    from models.moment_encoder import GeometryMomentEncoder

    # Initialize encoder
    encoder = GeometryMomentEncoder(args, device='cpu')

    # Create sample points (batch_size=3, 2 coordinates)
    points = torch.tensor([
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)

    print(f"Input points shape: {points.shape}")
    print(f"Points:\n{points}")

    # Encode points
    encoded = encoder.point_encoder(points)
    print(f"\nEncoded moments shape: {encoded.shape}")
    print(f"Number of moment features: {encoded.shape[1] * encoded.shape[2]}")
    print(f"\nFirst few moments for point [0.5, 0.5]:")
    print(encoded[0, :3, :3])


def example_line_encoding():
    """Example: Encode a batch of line segments."""
    print("\n" + "="*60)
    print("Example 2: Line Segment Encoding")
    print("="*60)

    # Configure moment encoder
    args = SimpleNamespace(
        moment_params={'basis_type': 'legendre', 'max_order': 5}
    )

    from models.moment_encoder import GeometryMomentEncoder

    encoder = GeometryMomentEncoder(args, device='cpu')

    # Create sample line segments (batch_size=2, 2 points, 2 coords)
    lines = torch.tensor([
        [[0.0, 0.0], [1.0, 1.0]],  # Diagonal line
        [[0.0, 0.5], [1.0, 0.5]]   # Horizontal line
    ], dtype=torch.float32)

    print(f"Input lines shape: {lines.shape}")
    print(f"Line 1: from {lines[0, 0]} to {lines[0, 1]}")
    print(f"Line 2: from {lines[1, 0]} to {lines[1, 1]}")

    # Encode lines
    encoded = encoder.line_encoder(lines)
    print(f"\nEncoded moments shape: {encoded.shape}")
    print(f"\nFirst few Legendre moments for diagonal line:")
    print(encoded[0, :3, :3])


def example_polyline_encoding():
    """Example: Encode polylines with variable lengths."""
    print("\n" + "="*60)
    print("Example 3: Polyline Encoding")
    print("="*60)

    args = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 4}
    )

    from models.moment_encoder import GeometryMomentEncoder

    encoder = GeometryMomentEncoder(args, device='cpu')

    # Create sample polylines (padded to max length)
    polylines = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    ], dtype=torch.float32)

    lengths = torch.tensor([4, 5], dtype=torch.long)

    print(f"Input polylines shape: {polylines.shape}")
    print(f"Lengths: {lengths}")

    encoded = encoder.polyline_encoder(polylines, lengths)
    print(f"\nEncoded moments shape: {encoded.shape}")
    print(f"\nMoments for triangle polyline (first 3x3):")
    print(encoded[0, :3, :3])


def example_polygon_encoding():
    """Example: Encode polygons."""
    print("\n" + "="*60)
    print("Example 4: Polygon Encoding")
    print("="*60)

    args = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 6}
    )

    from models.moment_encoder import GeometryMomentEncoder

    encoder = GeometryMomentEncoder(args, device='cpu')

    polygons = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
    ], dtype=torch.float32)

    lengths = torch.tensor([3, 4], dtype=torch.long)

    print(f"Input polygons shape: {polygons.shape}")
    print(f"Vertex counts: {lengths}")

    encoded = encoder.polygon_encoder(polygons, lengths)
    print(f"\nEncoded area moments shape: {encoded.shape}")
    print(f"\nArea moments for triangle (first 4x4):")
    print(encoded[0, :4, :4])
    print(f"\nNote: m=0, n=0 moment is the polygon area:")
    print(f"Triangle area (moment[0,0]): {encoded[0, 0, 0].item():.4f}")
    print(f"Square area (moment[0,0]): {encoded[1, 0, 0].item():.4f}")


def example_full_model():
    """Example: Use the complete Poly2Moment model."""
    print("\n" + "="*60)
    print("Example 5: Full Poly2Moment Model")
    print("="*60)

    args = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 10},
        d_input=121,
        d_hid=64,
        d_out=32,
        dropout=0.1,
        fusion='learned_fusion'
    )

    from models.poly2moment import Poly2Moment

    model = Poly2Moment(args, device='cpu')
    model.eval()

    polygons = torch.tensor([
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0], [0.0, 0.0]]
    ], dtype=torch.float32)

    lengths = torch.tensor([4, 3], dtype=torch.long)

    print(f"Input: 2 polygons (square and triangle)")
    print(f"Shapes: {polygons.shape}, lengths: {lengths}")

    with torch.no_grad():
        embeddings = model(polygons, lengths, dataset_type='polygons')

    print(f"\nOutput embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"\nSquare embedding (first 10 dims): {embeddings[0, :10]}")
    print(f"Triangle embedding (first 10 dims): {embeddings[1, :10]}")

    similarity = torch.nn.functional.cosine_similarity(
        embeddings[0:1], embeddings[1:2]
    )
    print(f"\nCosine similarity between square and triangle: {similarity.item():.4f}")


def compare_bases():
    """Example: Compare monomial vs Legendre bases."""
    print("\n" + "="*60)
    print("Example 6: Comparing Monomial vs Legendre Basis")
    print("="*60)

    from models.moment_encoder import GeometryMomentEncoder

    polygon = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    ], dtype=torch.float32)
    length = torch.tensor([4], dtype=torch.long)

    args_mono = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 5}
    )
    encoder_mono = GeometryMomentEncoder(args_mono, device='cpu')
    moments_mono = encoder_mono.polygon_encoder(polygon, length)

    args_leg = SimpleNamespace(
        moment_params={'basis_type': 'legendre', 'max_order': 5}
    )
    encoder_leg = GeometryMomentEncoder(args_leg, device='cpu')
    moments_leg = encoder_leg.polygon_encoder(polygon, length)

    print("Unit square polygon: [0,1] x [0,1]")
    print(f"\nMonomial moments (first 4x4):")
    print(moments_mono[0, :4, :4])
    print(f"\nLegendre moments (first 4x4):")
    print(moments_leg[0, :4, :4])

    print("\nObservations:")
    print("- Monomial moments grow with polynomial order")
    print("- Legendre moments are more balanced (orthogonal basis)")
    print("- Both encode the same geometric information differently")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Poly2Moment encoder examples")
    print("="*60)

    for fn in (
        example_point_encoding,
        example_line_encoding,
        example_polyline_encoding,
        example_polygon_encoding,
        example_full_model,
        compare_bases,
    ):
        try:
            fn()
        except Exception as e:
            print(f"Error in {fn.__name__}: {e}")

    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")
