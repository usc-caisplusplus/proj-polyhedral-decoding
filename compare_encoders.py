"""
Comparison script for Fourier-based vs Moment-based encoders.

This script compares:
1. Encoding output shapes
2. Feature characteristics
3. Computational considerations
"""

import torch
import numpy as np
from types import SimpleNamespace
import time


def compare_point_encoding():
    """Compare point encoding between Fourier and Moment encoders."""
    print("\n" + "="*70)
    print("Comparison 1: Point Encoding")
    print("="*70)

    # Sample points
    points = torch.tensor([
        [0.5, 0.5],
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=torch.float32)

    print(f"Input: {len(points)} points")
    print(f"Points: {points.tolist()}")

    # Fourier encoder
    print("\n--- Fourier Encoder ---")
    from models.fourier_encoder import GeometryFourierEncoder

    args_fourier = SimpleNamespace(
        gfm_params={'w_min': 0.1, 'w_max': 1.0, 'n_freqs': 10}
    )
    fourier_enc = GeometryFourierEncoder(args_fourier, device='cpu')

    start = time.time()
    fourier_encoded = fourier_enc.point_encoder(points)
    fourier_time = time.time() - start

    print(f"Output shape: {fourier_encoded.shape}")
    print(f"Output dtype: {fourier_encoded.dtype}")
    print(f"Is complex: {torch.is_complex(fourier_encoded)}")
    print(f"Encoding time: {fourier_time*1000:.3f} ms")
    print(f"Sample output (magnitude, first 3x3):")
    print(torch.abs(fourier_encoded[0, :3, :3]))

    # Moment encoder
    print("\n--- Moment Encoder (Monomial) ---")
    from models.moment_encoder import GeometryMomentEncoder

    args_moment = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 10}
    )
    moment_enc = GeometryMomentEncoder(args_moment, device='cpu')

    start = time.time()
    moment_encoded = moment_enc.point_encoder(points)
    moment_time = time.time() - start

    print(f"Output shape: {moment_encoded.shape}")
    print(f"Output dtype: {moment_encoded.dtype}")
    print(f"Is complex: {torch.is_complex(moment_encoded)}")
    print(f"Encoding time: {moment_time*1000:.3f} ms")
    print(f"Sample output (first 3x3):")
    print(moment_encoded[0, :3, :3])

    print(f"\nSpeed comparison: Fourier is {moment_time/fourier_time:.2f}x faster")


def compare_polygon_encoding():
    """Compare polygon encoding."""
    print("\n" + "="*70)
    print("Comparison 2: Polygon Encoding")
    print("="*70)

    # Sample polygons (square and triangle)
    polygons = torch.tensor([
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
        [[0.0, 0.0], [1.0, 0.0], [0.5, 1.0], [0.0, 0.0], [0.0, 0.0]]
    ], dtype=torch.float32)
    lengths = torch.tensor([4, 3], dtype=torch.long)

    print(f"Input: 2 polygons (square 2x2, triangle)")

    # Fourier encoder
    print("\n--- Fourier Encoder ---")
    from models.fourier_encoder import GeometryFourierEncoder

    args_fourier = SimpleNamespace(
        gfm_params={'w_min': 0.1, 'w_max': 1.0, 'n_freqs': 10}
    )
    fourier_enc = GeometryFourierEncoder(args_fourier, device='cpu')

    start = time.time()
    fourier_encoded = fourier_enc.polygon_encoder(polygons, lengths)
    fourier_time = time.time() - start

    print(f"Output shape: {fourier_encoded.shape}")
    print(f"Feature count: {fourier_encoded.shape[1] * fourier_encoded.shape[2]}")
    print(f"Encoding time: {fourier_time*1000:.3f} ms")
    print(f"Sample (magnitude at [0, 0, 0]): {torch.abs(fourier_encoded[0, 0, 0]):.6f}")
    print(f"  (This represents DC component - should relate to area)")

    # Moment encoder
    print("\n--- Moment Encoder (Monomial) ---")
    from models.moment_encoder import GeometryMomentEncoder

    args_moment = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 10}
    )
    moment_enc = GeometryMomentEncoder(args_moment, device='cpu')

    start = time.time()
    moment_encoded = moment_enc.polygon_encoder(polygons, lengths)
    moment_time = time.time() - start

    print(f"Output shape: {moment_encoded.shape}")
    print(f"Feature count: {moment_encoded.shape[1] * moment_encoded.shape[2]}")
    print(f"Encoding time: {moment_time*1000:.3f} ms")
    print(f"Sample (moment [0, 0, 0]): {moment_encoded[0, 0, 0]:.6f}")
    print(f"  (This is the area integral: ∬ 1 dA)")
    print(f"  Square area should be: 4.0, Triangle area should be: 0.5")
    print(f"  Computed square area: {moment_encoded[0, 0, 0]:.6f}")
    print(f"  Computed triangle area: {moment_encoded[1, 0, 0]:.6f}")

    print(f"\nSpeed comparison: Fourier is {moment_time/fourier_time:.2f}x faster")


def compare_feature_dimensions():
    """Compare feature dimensions and parameter counts."""
    print("\n" + "="*70)
    print("Comparison 3: Feature Dimensions & Model Parameters")
    print("="*70)

    # Fourier model
    print("\n--- Poly2Vec (Fourier-based) ---")
    args_fourier = SimpleNamespace(
        gfm_params={'w_min': 0.1, 'w_max': 1.0, 'n_freqs': 10},
        d_input=210,  # From default config
        d_hid=100,
        d_out=32,
        dropout=0.2,
        fusion='learned_fusion'
    )

    from models.poly2vec import Poly2Vec
    fourier_model = Poly2Vec(args_fourier, device='cpu')

    fourier_params = sum(p.numel() for p in fourier_model.parameters())
    print(f"Input dimension: {args_fourier.d_input}")
    print(f"Hidden dimension: {args_fourier.d_hid}")
    print(f"Output dimension: {args_fourier.d_out}")
    print(f"Total parameters: {fourier_params:,}")

    # Moment model
    print("\n--- Poly2Moment (Moment-based) ---")
    args_moment = SimpleNamespace(
        moment_params={'basis_type': 'monomial', 'max_order': 10},
        d_input=121,  # (10+1)^2
        d_hid=100,
        d_out=32,
        dropout=0.2,
        fusion='learned_fusion'
    )

    from models.poly2moment import Poly2Moment
    moment_model = Poly2Moment(args_moment, device='cpu')

    moment_params = sum(p.numel() for p in moment_model.parameters())
    print(f"Input dimension: {args_moment.d_input}")
    print(f"Hidden dimension: {args_moment.d_hid}")
    print(f"Output dimension: {args_moment.d_out}")
    print(f"Total parameters: {moment_params:,}")

    print(f"\nParameter difference: {abs(fourier_params - moment_params):,} parameters")
    print(f"Moment model is {fourier_params/moment_params:.2f}x larger" if fourier_params > moment_params
          else f"Fourier model is {moment_params/fourier_params:.2f}x larger")


def compare_properties():
    """Compare mathematical properties of encoders."""
    print("\n" + "="*70)
    print("Comparison 4: Mathematical Properties")
    print("="*70)

    print("\n--- Translation Invariance ---")
    from models.fourier_encoder import GeometryFourierEncoder
    from models.moment_encoder import GeometryMomentEncoder

    # Test translation invariance with a simple polygon
    polygon1 = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]], dtype=torch.float32)
    polygon2 = torch.tensor([[[5.0, 5.0], [6.0, 5.0], [5.5, 6.0]]], dtype=torch.float32)  # Translated
    lengths = torch.tensor([3], dtype=torch.long)

    args_fourier = SimpleNamespace(gfm_params={'w_min': 0.1, 'w_max': 1.0, 'n_freqs': 5})
    args_moment = SimpleNamespace(moment_params={'basis_type': 'monomial', 'max_order': 5})

    fourier_enc = GeometryFourierEncoder(args_fourier, device='cpu')
    moment_enc = GeometryMomentEncoder(args_moment, device='cpu')

    f1 = torch.abs(fourier_enc.polygon_encoder(polygon1, lengths).flatten())
    f2 = torch.abs(fourier_enc.polygon_encoder(polygon2, lengths).flatten())
    m1 = moment_enc.polygon_encoder(polygon1, lengths).flatten()
    m2 = moment_enc.polygon_encoder(polygon2, lengths).flatten()

    fourier_diff = torch.mean(torch.abs(f1 - f2))
    moment_diff = torch.mean(torch.abs(m1 - m2))

    print(f"Triangle at origin vs translated (+5, +5)")
    print(f"Fourier encoding difference (magnitude): {fourier_diff:.6f}")
    print(f"Moment encoding difference: {moment_diff:.6f}")
    print(f"\nNote: Fourier magnitude is phase-invariant (more robust to translation)")
    print(f"      Moments change significantly with translation")

    print("\n--- Rotation ---")
    print("Fourier: Phase changes with rotation (magnitude preserves some info)")
    print("Moments: Change significantly with rotation")
    print("Both require learned rotation invariance in downstream model")

    print("\n--- Scaling ---")
    print("Fourier: Frequency-based encoding changes with scale")
    print("Moments: Higher-order moments grow rapidly with scaling")
    print("Both capture scale information in the encoding")


def feature_summary():
    """Print a summary comparison table."""
    print("\n" + "="*70)
    print("Summary: Fourier vs Moment Encoders")
    print("="*70)

    table = """
| Property              | Fourier (Poly2Vec)          | Moment (Poly2Moment)        |
|-----------------------|-----------------------------|-----------------------------|
| Basis Functions       | exp(-2πi(ux + vy))          | x^m y^n or P_m(x)P_n(y)    |
| Parameters            | Continuous (w_min, w_max)   | Integer (max_order)         |
| Output Type           | Complex-valued              | Real-valued                 |
| # Features (default)  | 210 (from GFM grid)         | 121 (from 11x11 grid)       |
| Translation           | Phase changes, mag stable   | Significant change          |
| Rotation              | Phase changes               | Significant change          |
| Computational         | Fast (vectorized FFT-like)  | Slower (nested loops)       |
| Numerical Stability   | Good                        | Use Legendre for high order |
| Interpretability      | Frequency domain            | Polynomial moments          |
| Best For              | General-purpose, fast       | When moments meaningful     |
"""
    print(table)

    print("\nRecommendations:")
    print("- Use Fourier encoder for general tasks (default)")
    print("- Use Moment encoder when:")
    print("  * Polynomial moments have domain-specific meaning")
    print("  * You need real-valued features")
    print("  * You want to compare with classical moment-based methods")
    print("  * Computational cost is acceptable")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Poly2Vec Encoder Comparison: Fourier vs Moment")
    print("="*70)

    try:
        compare_point_encoding()
    except Exception as e:
        print(f"Error: {e}")

    try:
        compare_polygon_encoding()
    except Exception as e:
        print(f"Error: {e}")

    try:
        compare_feature_dimensions()
    except Exception as e:
        print(f"Error: {e}")

    try:
        compare_properties()
    except Exception as e:
        print(f"Error: {e}")

    try:
        feature_summary()
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "="*70)
    print("Comparison completed!")
    print("="*70 + "\n")
