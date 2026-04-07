# Moment-Based Encoder for Poly2Vec

This document describes the alternative moment-based encoder implementation that uses integer-based polynomial moments instead of Fourier transforms.

## Overview

The moment-based encoder provides an alternative to the Fourier-based encoding in Poly2Vec. Instead of using Geometric Fourier Meshgrid (GFM) with continuous frequency parameters, this encoder uses **integer-based polynomial moments** with two available basis types:

1. **Monomial basis**: φ_mn(x,y) = x^m y^n
2. **Legendre basis**: φ_mn(x,y) = P_m(x) P_n(y)

## Architecture

The moment-based encoder follows the same architectural pattern as Poly2Vec:

### File Structure

- **[models/moment_encoder.py](models/moment_encoder.py)**: `GeometryMomentEncoder` class that computes moments for points, lines, polylines, and polygons
- **[models/poly2moment.py](models/poly2moment.py)**: `Poly2Moment` model that learns embeddings from moment features
- **[utils/moments_compute.py](utils/moments_compute.py)**: Core moment computation functions for different bases

### Components

1. **GeometryMomentEncoder**
   - Encodes geospatial objects using polynomial moment bases
   - Supports polymorphic encoding for points, polylines, and polygons
   - Uses integer meshgrid for (m, n) moment orders
   - Key methods:
     - `point_encoder()`: Computes moments for single points
     - `line_encoder()`: Computes line segment moments using closed-form integrals
     - `polyline_encoder()`: Sums moments of constituent line segments
     - `polygon_encoder()`: Computes area moments using Green's theorem

2. **Poly2Moment**
   - Learns fixed-length embeddings from moment features
   - Uses dual-stream architecture with learned fusion
   - Two processing streams handle different aspects of moment features
   - Produces final embeddings via MLP

## Configuration

### Using Moment Encoder in config.json

To use the moment-based encoder, update your `config.json`:

```json
{
    "encoder_type": "poly2moment",

    "moment_params": {
        "basis_type": "monomial",
        "max_order": 10
    },

    "d_input": 121,
    "d_hid": 100,
    "d_out": 32
}
```

### Parameters

#### moment_params
- **`basis_type`**: Choice of polynomial basis
  - `"monomial"`: Use monomial basis x^m y^n (default)
  - `"legendre"`: Use Legendre polynomial basis P_m(x) P_n(y)

- **`max_order`**: Maximum polynomial order (default: 10)
  - Creates a meshgrid from (0,0) to (max_order, max_order)
  - Total number of moments: (max_order + 1)²
  - Example: max_order=10 gives 121 moment features

#### d_input
- Must equal (max_order + 1)² to match the flattened moment tensor size
- Example: max_order=10 → d_input=121

## Training Example

Train a model using the monomial moment encoder:

```bash
python run.py \
  -dataset_name "NewYork" \
  -dataset_type1 "polygons" \
  -dataset_type2 "polygons" \
  -task "multi-relation" \
  -data_file "./data/NewYork/polygon_polygon_intersect_data.pt" \
  -encoder_type "poly2moment" \
  -data_path "./data/NewYork" \
  -num_classes 6
```

Train using Legendre basis:

```bash
python run.py \
  -dataset_name "NewYork" \
  -dataset_type1 "polygons" \
  -dataset_type2 "polygons" \
  -task "multi-relation" \
  -encoder_type "poly2moment" \
  -data_file "./data/NewYork/polygon_polygon_intersect_data.pt"
```

And update your config.json to:
```json
{
    "moment_params": {
        "basis_type": "legendre",
        "max_order": 10
    }
}
```

## Key Differences from Fourier Encoder

| Aspect | Fourier Encoder | Moment Encoder |
|--------|----------------|----------------|
| **Basis** | Complex exponentials exp(-2πi(ux + vy)) | Polynomial monomials x^m y^n or Legendre P_m(x) P_n(y) |
| **Parameters** | Continuous frequencies (w_min, w_max, n_freqs) | Integer orders (0 to max_order) |
| **Output** | Complex-valued tensors (magnitude + phase) | Real-valued tensors (dual-stream processing) |
| **Features** | Geometric Fourier Meshgrid (GFM) | Integer meshgrid of polynomial orders |
| **Computational** | Closed-form Fourier transforms | Closed-form moment integrals via Green's theorem |

## Moment Computation Details

### Point Moments
For a point at (x, y):
- **Monomial**: m_mn = x^m y^n
- **Legendre**: m_mn = P_m(x) P_n(y)

### Line Segment Moments
For a line from (ax, ay) to (bx, by):
- Computed using parameterized arc-length integrals
- Closed-form formulas using binomial expansions

### Polygon Moments
For a polygon with vertices V:
- Area moments computed using Green's theorem
- Edge-based summation for computational efficiency
- Handles self-intersections via Shapely preprocessing

## Performance Considerations

### Computational Complexity
- **Moment computation**: O(max_order²) per geometry
- **Vectorized for batch processing**
- Real-valued operations (no complex arithmetic)

### Memory Usage
- Moment tensor size: (batch_size, max_order+1, max_order+1)
- Example: batch_size=128, max_order=10 → ~1.6M floats per batch

### Recommended Settings
- **max_order=10**: Good balance of expressiveness and efficiency (121 features)
- **max_order=15**: Higher-order features (256 features), more computational cost
- **basis_type="monomial"**: Fastest computation
- **basis_type="legendre"**: Better numerical stability for higher orders

## Mathematical Background

### Monomial Moments
Area moment of order (m,n):
```
M_mn = ∬_Ω x^m y^n dA
```

Computed via Green's theorem as edge integral:
```
M_mn = ∑_edges (boundary integral of x^(m+1) y^n / (m+1))
```

### Legendre Moments
Area moment using Legendre polynomials:
```
C_mn = ∬_Ω P_m(x) P_n(y) dA
```

Computed by expanding Legendre polynomials into monomial basis and reusing monomial moments.

## Citation

If you use the moment-based encoder in your research, please cite both:

1. The original Poly2Vec paper (for the framework)
2. The moments_compute.py implementation (if applicable)

## Troubleshooting

### Common Issues

1. **Dimension mismatch error**
   - Ensure `d_input = (max_order + 1)²`
   - Example: max_order=10 requires d_input=121

2. **Self-intersecting polygons**
   - Automatically handled by Shapely's `buffer(0)` preprocessing
   - May slightly modify polygon geometry

3. **Numerical instability for high orders**
   - Use Legendre basis instead of monomial for max_order > 15
   - Legendre polynomials are orthogonal and better conditioned

4. **Slow encoding**
   - Moment computation involves nested loops over (m,n) pairs
   - Consider reducing max_order if speed is critical
   - Future optimization: vectorize moment computations

## Future Extensions

Potential improvements:
1. GPU-accelerated moment computation
2. Support for Fourier moments (already in moments_compute.py)
3. Adaptive moment order selection
4. Moment normalization schemes
5. Integration with other polynomial bases (Chebyshev, Hermite, etc.)
