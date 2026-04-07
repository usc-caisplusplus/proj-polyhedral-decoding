"""
Geometric Fourier Meshgrid (GFM) frequency lattice — pure PyTorch (no Triangle/Shapely).

Used by ``GeometryFourierEncoder`` and by training scripts that need ``d_input`` without
loading the full encoder stack.
"""

from __future__ import annotations

import torch


def gfm_meshgrid(params: dict):
    """
    Build the (Wx, Wy) frequency mesh for GFM.

    Returns:
        U, V: 2D tensors of same shape (frequency indices in x and y).
    """
    n_freq = params["n_freqs"]
    if n_freq < 2:
        raise ValueError("gfm_params['n_freqs'] must be >= 2 (geometric ratio uses n_freqs - 1).")
    w_min = params["w_min"]
    w_max = params["w_max"]
    g = (w_max / w_min) ** (1 / (n_freq - 1))

    positive_wu = torch.tensor([w_min * g**u for u in range(n_freq)], dtype=torch.float32)

    if (2 * n_freq + 1) % 2 == 1:
        Wx = torch.cat((-torch.flip(positive_wu, dims=[0]), torch.tensor([0.0]), positive_wu))
    else:
        Wx = torch.cat((-torch.flip(positive_wu[:-1], dims=[0]), torch.tensor([0.0]), positive_wu))

    if n_freq % 2 == 1:
        Wy = torch.cat((torch.tensor([0.0]), positive_wu))
    else:
        Wy = positive_wu

    return torch.meshgrid(Wx, Wy, indexing="ij")


def gfm_feature_dim(gfm_params: dict) -> int:
    """Number of complex Fourier bins; equals ``Poly2Vec`` ``args.d_input`` for that GFM."""
    U, V = gfm_meshgrid(gfm_params)
    assert U.shape == V.shape
    return int(U.numel())
