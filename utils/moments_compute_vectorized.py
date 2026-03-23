"""
Vectorized batch moment computations.

Each function replaces its scalar counterpart in moments_compute.py but operates
on an entire batch of geometries and the full (O x O) moment grid simultaneously,
eliminating all Python-level loops over (m, n) pairs and geometry indices.

Shapes legend:
    O  = max_order + 1   (moment grid side length)
    B  = batch size      (number of geometries)
    S  = max segments    (padded)
    K  = max vertices    (padded)

Precomputed tables (built once at init time by MomentTables):
    leg_coef : (O, O)  float64  — leg_coef[m, u] = coeff of x^u in P_m(x)
    binom    : (O, O)  float64  — binom[n, k] = C(n, k)
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial


# ---------------------------------------------------------------------------
# Precomputed tables
# ---------------------------------------------------------------------------

class MomentTables:
    """
    Precomputes all order-dependent tables that are constant for a given max_order.
    Instantiate once and pass to the vectorized functions.
    """

    def __init__(self, max_order: int):
        self.O = max_order + 1
        O = self.O

        # Legendre coefficient matrix: leg_coef[m, u] = coeff of x^u in P_m(x)
        self.leg_coef = np.zeros((O, O), dtype=np.float64)
        for m in range(O):
            pm = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
            c = pm.coef
            self.leg_coef[m, : len(c)] = c

        # Binomial table: binom[n, k] = C(n, k)  for 0 <= k <= n < O
        from math import comb
        self.binom = np.zeros((O, O), dtype=np.float64)
        for n in range(O):
            for k in range(n + 1):
                self.binom[n, k] = comb(n, k)

        # Integer order arrays for broadcasting
        orders = np.arange(O, dtype=np.float64)          # (O,)
        self.M_grid, self.N_grid = np.meshgrid(orders, orders, indexing='ij')  # (O,O)

        # Power index arrays  i = 0..O-1,  j = 0..O-1
        self.i_idx = np.arange(O, dtype=np.float64)      # (O,)
        self.j_idx = np.arange(O, dtype=np.float64)      # (O,)


# ---------------------------------------------------------------------------
# Points  —  O(B * O^2)  instead of O(B * O^2 * python_overhead)
# ---------------------------------------------------------------------------

def batch_moments_points_monomial(
    points: np.ndarray,   # (B, 2)
    tables: MomentTables,
) -> np.ndarray:          # (B, O, O)  float64
    """
    M[b, m, n] = x_b^m * y_b^n
    """
    x = points[:, 0]     # (B,)
    y = points[:, 1]     # (B,)
    M = tables.M_grid    # (O, O)
    N = tables.N_grid    # (O, O)

    # (B,1,1) ** (1,O,O)  →  (B,O,O)
    return (x[:, None, None] ** M[None]) * (y[:, None, None] ** N[None])


def batch_moments_points_fourier(
    points: np.ndarray,   # (B, 2)
    tables: MomentTables,
) -> np.ndarray:          # (B, O, O)  complex128
    """
    M[b, m, n] = exp(-i pi (m x_b + n y_b))
    """
    x = points[:, 0]     # (B,)
    y = points[:, 1]     # (B,)
    M = tables.M_grid    # (O, O)
    N = tables.N_grid    # (O, O)

    # exponent:  (B,O,O)
    exponent = M[None] * x[:, None, None] + N[None] * y[:, None, None]
    return np.exp(-1j * np.pi * exponent)


def batch_moments_points_legendre(
    points: np.ndarray,   # (B, 2)
    tables: MomentTables,
) -> np.ndarray:          # (B, O, O)  float64
    """
    M[b, m, n] = P_m(x_b) * P_n(y_b)

    Uses the precomputed leg_coef matrix to evaluate all Legendre polynomials
    at all x/y values simultaneously via matrix multiply.
    """
    x = points[:, 0]     # (B,)
    y = points[:, 1]     # (B,)
    O = tables.O
    A = tables.leg_coef  # (O, O)  A[m, u] = coeff of x^u in P_m(x)

    # Power matrix:  pow_x[b, u] = x_b^u,   shape (B, O)
    pow_x = x[:, None] ** np.arange(O, dtype=np.float64)[None]  # (B, O)
    pow_y = y[:, None] ** np.arange(O, dtype=np.float64)[None]  # (B, O)

    # P_m(x_b) = sum_u A[m,u] * x_b^u  →  (B, O) via matmul
    Pm_vals = pow_x @ A.T   # (B, O)  — Pm_vals[b, m] = P_m(x_b)
    Pn_vals = pow_y @ A.T   # (B, O)  — Pn_vals[b, n] = P_n(y_b)

    # Outer product over (m, n): (B, O, 1) * (B, 1, O)  →  (B, O, O)
    return Pm_vals[:, :, None] * Pn_vals[:, None, :]


# ---------------------------------------------------------------------------
# Line segments — batch over all segments AND all (m,n) pairs simultaneously
# ---------------------------------------------------------------------------

def _batch_monomial_line_moments(
    ax: np.ndarray,   # (S,)
    ay: np.ndarray,   # (S,)
    dx: np.ndarray,   # (S,)
    dy: np.ndarray,   # (S,)
    L:  np.ndarray,   # (S,)
    tables: MomentTables,
) -> np.ndarray:      # (S, O, O)  float64
    """
    Core vectorized monomial line-segment moment for a flat list of S segments.

    Formula:
        C[s,m,n] = L[s] * sum_{i=0}^{m} sum_{j=0}^{n}
                     C(m,i)*C(n,j) * ax[s]^(m-i) * ay[s]^(n-j)
                                   * dx[s]^i * dy[s]^j / (i+j+1)
    """
    S = ax.shape[0]
    O = tables.O
    binom = tables.binom   # (O, O)

    # Power tables:  pow_a[s, p] = ax[s]^p,  shape (S, O)
    p_range = np.arange(O, dtype=np.float64)
    pow_ax = ax[:, None] ** p_range[None]   # (S, O)
    pow_ay = ay[:, None] ** p_range[None]   # (S, O)
    pow_dx = dx[:, None] ** p_range[None]   # (S, O)
    pow_dy = dy[:, None] ** p_range[None]   # (S, O)

    # For each (m, n) and each (i, j) summation term, build a 4-D tensor
    # term[m, n, i, j] = C(m,i)*C(n,j) / (i+j+1)
    #
    # Then contract with per-segment power products.

    # i runs 0..O-1, j runs 0..O-1
    i_idx = np.arange(O)   # (O,)
    j_idx = np.arange(O)   # (O,)
    m_idx = np.arange(O)   # (O,)
    n_idx = np.arange(O)   # (O,)

    # denom[i, j] = 1 / (i + j + 1)   shape (O, O)
    denom = 1.0 / (i_idx[:, None] + j_idx[None, :] + 1.0)   # (O, O)

    # C(m, i): valid only when i <= m, else 0 — already in binom table
    # binom[m, i]  shape (O, O)
    # binom[n, j]  shape (O, O)

    # coeff[m, n, i, j] = C(m,i) * C(n,j) / (i+j+1)
    # shape: (O, O, O, O)  — manageable for O<=16
    coeff = (binom[:, None, :, None] *    # C(m, i): (O,1,O,1)
             binom[None, :, None, :] *    # C(n, j): (1,O,1,O)
             denom[None, None, :, :])     # 1/(i+j+1): (1,1,O,O)
    # coeff shape: (O, O, O, O)  = (m, n, i, j)

    # ax^(m-i): need (S, O_m, O_i) where entry [s, m, i] = ax[s]^max(m-i, 0)
    # Build it: for m-i >= 0: pow_ax[s, m-i], else 0 (but binom handles this since C(m,i)=0 for i>m)
    # We can use pow_ax[s, m-i] for all i <= m, and for i > m binom is 0 so result is 0.
    # Build index table: mi_exp[m, i] = max(m - i, 0)
    mi_exp = np.maximum(m_idx[:, None] - i_idx[None, :], 0)   # (O, O)
    nj_exp = np.maximum(n_idx[:, None] - j_idx[None, :], 0)   # (O, O)

    # pow_ax_mi[s, m, i] = ax[s]^(m-i)
    pow_ax_mi = pow_ax[:, mi_exp]   # (S, O, O)
    pow_ay_nj = pow_ay[:, nj_exp]   # (S, O, O)
    pow_dx_i  = pow_dx[:, i_idx]    # (S, O)
    pow_dy_j  = pow_dy[:, j_idx]    # (S, O)

    # term[s, m, n, i, j] = coeff[m,n,i,j] * ax[s]^(m-i) * ay[s]^(n-j) * dx[s]^i * dy[s]^j
    #
    # Build this efficiently:
    # A[s, m, i] = ax[s]^(m-i) * dx[s]^i   →  pow_ax_mi * pow_dx_i  (S, O, O)
    # B[s, n, j] = ay[s]^(n-j) * dy[s]^j   →  pow_ay_nj * pow_dy_j  (S, O, O)

    A = pow_ax_mi * pow_dx_i[:, None, :]    # (S, O, O)  [s, m, i]
    B = pow_ay_nj * pow_dy_j[:, None, :]    # (S, O, O)  [s, n, j]

    # Sum over i, j:
    # result[s, m, n] = sum_{i,j} coeff[m,n,i,j] * A[s,m,i] * B[s,n,j]
    #
    # = sum_i A[s,m,i] * (sum_j coeff[m,n,i,j] * B[s,n,j])
    #
    # Step 1: inner[s, m, n, i] = sum_j coeff[m, n, i, j] * B[s, n, j]
    #         B:     (S, O, O)  →  (S, 1, O, O)  [s, _, n, j]
    #         coeff: (O, O, O, O)  [m, n, i, j]
    # Step 2: contract over j:  einsum('mnij, snj -> smni')  then contract over i

    # Use einsum for clarity and efficiency
    # inner[s, m, n, i] = sum_j coeff[m,n,i,j] * B[s,n,j]
    inner = np.einsum('mnij,snj->smni', coeff, B)    # (S, O, O, O)

    # result[s, m, n] = sum_i A[s, m, i] * inner[s, m, n, i]
    result = np.einsum('smi,smni->smn', A, inner)    # (S, O, O)

    return L[:, None, None] * result


def batch_moments_line_monomial(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    tables: MomentTables,
) -> np.ndarray:   # (S, O, O) float64
    dx = bx - ax
    dy = by - ay
    L  = np.hypot(dx, dy)
    return _batch_monomial_line_moments(ax, ay, dx, dy, L, tables)


def batch_moments_line_fourier(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    tables: MomentTables,
    eps: float = 1e-14,
) -> np.ndarray:   # (S, O, O) complex128
    """
    F[s, m, n] = L[s] * exp(-i pi (m ax + n ay)) * sinc_term(m dx + n dy)

    where sinc_term(z) = (1 - exp(-i pi z)) / (i pi z)  if |z| > eps
                       = 1                               otherwise
    """
    dx = bx - ax
    dy = by - ay
    L  = np.hypot(dx, dy)          # (S,)

    M = tables.M_grid              # (O, O)
    N = tables.N_grid              # (O, O)

    # kdot_a[s, m, n] = m*ax[s] + n*ay[s]
    kdot_a = M[None] * ax[:, None, None] + N[None] * ay[:, None, None]   # (S, O, O)
    kdot_d = M[None] * dx[:, None, None] + N[None] * dy[:, None, None]   # (S, O, O)

    phase0 = np.exp(-1j * np.pi * kdot_a)                                 # (S, O, O)

    # sinc-like term — handle near-zero denominator
    with np.errstate(invalid='ignore', divide='ignore'):
        sinc = np.where(
            np.abs(kdot_d) < eps,
            np.ones_like(kdot_d, dtype=complex),
            (1.0 - np.exp(-1j * np.pi * kdot_d)) / (1j * np.pi * kdot_d)
        )                                                                  # (S, O, O)

    return L[:, None, None] * phase0 * sinc


def batch_moments_line_legendre(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    tables: MomentTables,
) -> np.ndarray:   # (S, O, O) float64
    """
    C[s, m, n] = L[s] * P_m(x(t)) P_n(y(t)) integrated over arc length.

    Expands P_m, P_n into monomials using the precomputed leg_coef matrix,
    then reuses the monomial line segment formula.

    C_leg[s, m, n] = sum_{u,v} A[m,u] * A[n,v] * C_mono[s, u, v]
                   = (A ⊗ A) @ C_mono   (double matrix multiply)
    """
    dx = bx - ax
    dy = by - ay
    L  = np.hypot(dx, dy)

    # Compute all monomial moments at once: (S, O, O)
    C_mono = _batch_monomial_line_moments(ax, ay, dx, dy, L, tables)

    # A[m, u] = coeff of x^u in P_m(x),  shape (O, O)
    A = tables.leg_coef

    # C_leg[s, m, n] = sum_{u,v} A[m,u] * A[n,v] * C_mono[s, u, v]
    #                = einsum('mu, nv, suv -> smn', A, A, C_mono)
    # Do it in two steps for efficiency:
    #   tmp[s, m, v] = sum_u A[m, u] * C_mono[s, u, v]
    tmp = np.einsum('mu,suv->smv', A, C_mono)    # (S, O, O)
    #   C_leg[s, m, n] = sum_v A[n, v] * tmp[s, m, v]
    C_leg = np.einsum('nv,smv->smn', A, tmp)     # (S, O, O)

    return C_leg


# ---------------------------------------------------------------------------
# Polylines — batch over (B geometries × variable segments)
# ---------------------------------------------------------------------------

def batch_moments_polylines(
    polylines: np.ndarray,   # (B, maxS, 2)  padded
    lengths: np.ndarray,     # (B,)  int — actual number of points
    tables: MomentTables,
    basis: str = 'legendre',
) -> np.ndarray:             # (B, O, O)
    """
    Computes moment grids for a batch of B polylines by:

    1. Flattening all valid segments across all geometries into a single
       array of shape (total_segs, 4).
    2. Computing all moments for all segments at once (one call to a
       batch line-segment function).
    3. Scatter-adding results back to per-geometry output using np.add.at.

    This eliminates the triple Python loop over (geom, segment, m, n).
    """
    B = polylines.shape[0]
    O = tables.O
    dtype = np.complex128 if basis == 'fourier' else np.float64

    # --- collect valid segments ---
    seg_ax, seg_ay, seg_bx, seg_by = [], [], [], []
    seg_geom_idx = []     # which geometry each segment belongs to

    lengths_np = np.asarray(lengths, dtype=int)
    for b in range(B):
        n_pts = int(lengths_np[b])
        if n_pts < 2:
            continue
        pts = polylines[b, :n_pts]         # (n_pts, 2)
        a_pts = pts[:-1]                   # (n_pts-1, 2)
        b_pts = pts[1:]                    # (n_pts-1, 2)
        n_segs = n_pts - 1
        seg_ax.append(a_pts[:, 0])
        seg_ay.append(a_pts[:, 1])
        seg_bx.append(b_pts[:, 0])
        seg_by.append(b_pts[:, 1])
        seg_geom_idx.extend([b] * n_segs)

    out = np.zeros((B, O, O), dtype=dtype)
    if not seg_ax:
        return out

    ax = np.concatenate(seg_ax)
    ay = np.concatenate(seg_ay)
    bx = np.concatenate(seg_bx)
    by = np.concatenate(seg_by)
    idx = np.array(seg_geom_idx, dtype=int)   # (total_segs,)

    # --- compute all segment moments at once ---
    if basis == 'monomial':
        seg_moments = batch_moments_line_monomial(ax, ay, bx, by, tables)
    elif basis == 'fourier':
        seg_moments = batch_moments_line_fourier(ax, ay, bx, by, tables)
    elif basis == 'legendre':
        seg_moments = batch_moments_line_legendre(ax, ay, bx, by, tables)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    # --- scatter-add:  out[geom_idx[s]] += seg_moments[s] ---
    np.add.at(out, idx, seg_moments)

    return out


# ---------------------------------------------------------------------------
# Polygons — batch over (B geometries × edges) for all moment orders at once
# ---------------------------------------------------------------------------

def _batch_monomial_polygon_moments_edges(
    ax: np.ndarray,   # (E,)  edge start x
    ay: np.ndarray,   # (E,)  edge start y
    dx: np.ndarray,   # (E,)  edge delta x
    dy: np.ndarray,   # (E,)  edge delta y
    tables: MomentTables,
) -> np.ndarray:      # (E, O, O) float64
    """
    Vectorized Green's-theorem formula for monomial polygon moments over a
    flat list of E edges (from any geometry).

    M[e, m, n] = (dy[e] / (m+1)) * sum_{i=0}^{m+1} sum_{j=0}^{n}
                   C(m+1, i)*C(n, j) * ax[e]^(m+1-i) * ay[e]^(n-j)
                              * dx[e]^i * dy[e]^j / (i+j+1)

    After scatter-adding, the total moment for polygon g is sum of edge contributions.
    """
    O = tables.O
    binom = tables.binom
    E = ax.shape[0]

    # Need powers 0..O (one extra) because p = m+1 reaches O when m = O-1
    p_range = np.arange(O + 1, dtype=np.float64)
    pow_ax = ax[:, None] ** p_range[None]    # (E, O+1)
    pow_ay = ay[:, None] ** p_range[None]    # (E, O+1)
    pow_dx = dx[:, None] ** p_range[None]    # (E, O+1)
    pow_dy = dy[:, None] ** p_range[None]    # (E, O+1)

    m_idx = np.arange(O)
    n_idx = np.arange(O)
    # i sums from 0 to p = m+1; max p = O, so i needs O+1 values (0..O)
    i_idx = np.arange(O + 1)
    j_idx = np.arange(O)

    # p = m + 1,  we need binom[p, i] = C(p, i)
    # Extend binom table to size (O+1) x (O+1) to cover p up to O.
    O2 = O + 1
    from math import comb as _comb
    binom_ext = np.zeros((O2, O2), dtype=np.float64)
    for nn in range(O2):
        for kk in range(nn + 1):
            binom_ext[nn, kk] = _comb(nn, kk)

    # denom[i, j] = 1 / (i + j + 1),  shape (O+1, O)
    denom = 1.0 / (i_idx[:, None] + j_idx[None, :] + 1.0)

    # p_arr[m] = m + 1  for m in 0..O-1,  shape (O,)
    p_arr = m_idx + 1   # (O,)

    # coeff_green[m, n, i, j] = C(m+1, i) * C(n, j) / (i+j+1)
    # shape (O, O, O+1, O):
    coeff_green = (binom_ext[p_arr[:, None, None, None], i_idx[None, None, :, None]] *
                   binom[n_idx[None, :, None, None],     j_idx[None, None, None, :]] *
                   denom[None, None, :, :])

    # pow index for ax:  ax^(p - i) = ax^(m+1 - i),  clamped to 0
    pi_exp = np.maximum(p_arr[:, None] - i_idx[None, :], 0)   # (O, O+1)
    nj_exp = np.maximum(n_idx[:, None] - j_idx[None, :], 0)   # (O, O)

    pow_ax_pi = pow_ax[:, pi_exp]    # (E, O, O+1)  [e, m, i]
    pow_ay_nj = pow_ay[:, nj_exp]    # (E, O, O)    [e, n, j]
    pow_dx_i  = pow_dx[:, i_idx]     # (E, O+1)
    pow_dy_j  = pow_dy[:, j_idx]     # (E, O)

    A = pow_ax_pi * pow_dx_i[:, None, :]    # (E, O, O+1)  [e, m, i]
    B = pow_ay_nj * pow_dy_j[:, None, :]    # (E, O, O)    [e, n, j]

    # result[e, m, n] = sum_{i,j} coeff_green[m,n,i,j] * A[e,m,i] * B[e,n,j]
    inner = np.einsum('mnij,enj->emni', coeff_green, B)   # (E, O, O, O+1)
    inner_sum = np.einsum('emi,emni->emn', A, inner)      # (E, O, O)

    # Green's theorem factor: dy[e] / (m + 1)
    factor = dy[:, None, None] / (m_idx[None, :, None] + 1.0)   # (E, O, O)

    return factor * inner_sum


def batch_moments_polygons_monomial(
    polygons: np.ndarray,    # (B, maxK, 2)  padded
    lengths:  np.ndarray,    # (B,)  int — actual vertex count
    tables:   MomentTables,
    signs=None,  # (B, maxK) np.ndarray  +1 exterior, -1 hole, 0 sentinel/pad
) -> np.ndarray:             # (B, O, O) float64
    """
    Vectorized Green's-theorem monomial polygon moments for a batch.

    When `signs` is provided (hole-aware mode), NaN sentinel rows are skipped
    and each edge's contribution is multiplied by the sign of its start vertex
    (+1 for exterior edges, -1 for hole edges), implementing:
        M_polygon_with_hole = M_exterior - M_hole
    """
    B = polygons.shape[0]
    O = tables.O

    edge_ax, edge_ay, edge_dx, edge_dy, edge_signs = [], [], [], [], []
    edge_geom_idx = []
    lengths_np = np.asarray(lengths, dtype=int)
    hole_aware = signs is not None

    for b in range(B):
        K = int(lengths_np[b])
        if K < 3:
            continue
        V = polygons[b, :K]           # (K, 2)
        S = signs[b, :K] if hole_aware else np.ones(K)  # (K,)

        # Build edges, skipping any vertex that is a NaN sentinel or whose
        # sign is 0 (sentinel), and also skipping edges that would cross a
        # ring boundary (i.e. where the next vertex is a sentinel).
        for k in range(K):
            if hole_aware and S[k] == 0.0:
                continue  # skip the NaN sentinel vertex itself
            k_next = (k + 1) % K
            # Skip edge if next vertex is a sentinel or a different ring
            if hole_aware and S[k_next] == 0.0:
                continue
            # Skip closing edge that would jump from one ring to another
            if hole_aware and S[k] != S[k_next] and k_next != 0:
                continue
            ax_val, ay_val = V[k, 0], V[k, 1]
            bx_val, by_val = V[k_next, 0], V[k_next, 1]
            if np.isnan(ax_val) or np.isnan(bx_val):
                continue
            edge_ax.append(ax_val)
            edge_ay.append(ay_val)
            edge_dx.append(bx_val - ax_val)
            edge_dy.append(by_val - ay_val)
            edge_signs.append(S[k])
            edge_geom_idx.append(b)

    out = np.zeros((B, O, O), dtype=np.float64)
    if not edge_ax:
        return out

    ax  = np.array(edge_ax)
    ay  = np.array(edge_ay)
    dx  = np.array(edge_dx)
    dy  = np.array(edge_dy)
    sg  = np.array(edge_signs)        # (E,)
    idx = np.array(edge_geom_idx, dtype=int)

    edge_moments = _batch_monomial_polygon_moments_edges(ax, ay, dx, dy, tables)
    # Apply per-edge sign: hole edges contribute negatively
    edge_moments *= sg[:, None, None]
    np.add.at(out, idx, edge_moments)

    return out


def batch_moments_polygons_fourier(
    polygons: np.ndarray,    # (B, maxK, 2)  padded
    lengths:  np.ndarray,    # (B,)  int
    tables:   MomentTables,
    eps: float = 1e-14,
    signs=None,  # (B, maxK) np.ndarray  +1 exterior, -1 hole, 0 sentinel/pad
) -> np.ndarray:             # (B, O, O) complex128
    """
    Vectorized Green's-theorem Fourier polygon moments for a batch.

    For m != 0:  contrib = inv_m * dy[e] * edge_int[e, m, n]
    For m == 0, n != 0:  contrib = inv_n * (-dx[e]) * edge_int[e, 0, n]
    For m == n == 0:  contrib = shoelace area

    When `signs` is provided, hole edges are multiplied by -1 so their
    contribution is subtracted, implementing Green's theorem for polygons
    with holes.
    """
    B = polygons.shape[0]
    O = tables.O
    M = tables.M_grid   # (O, O)
    N = tables.N_grid   # (O, O)

    edge_ax, edge_ay, edge_dx, edge_dy, edge_signs = [], [], [], [], []
    edge_geom_idx = []
    lengths_np = np.asarray(lengths, dtype=int)
    hole_aware = signs is not None

    for b in range(B):
        K = int(lengths_np[b])
        if K < 3:
            continue
        V = polygons[b, :K]
        S = signs[b, :K] if hole_aware else np.ones(K)

        for k in range(K):
            if hole_aware and S[k] == 0.0:
                continue
            k_next = (k + 1) % K
            if hole_aware and S[k_next] == 0.0:
                continue
            if hole_aware and S[k] != S[k_next] and k_next != 0:
                continue
            ax_val, ay_val = V[k, 0], V[k, 1]
            bx_val, by_val = V[k_next, 0], V[k_next, 1]
            if np.isnan(ax_val) or np.isnan(bx_val):
                continue
            edge_ax.append(ax_val)
            edge_ay.append(ay_val)
            edge_dx.append(bx_val - ax_val)
            edge_dy.append(by_val - ay_val)
            edge_signs.append(S[k])
            edge_geom_idx.append(b)

    out = np.zeros((B, O, O), dtype=np.complex128)
    if not edge_ax:
        return out

    ax  = np.array(edge_ax)
    ay  = np.array(edge_ay)
    dx  = np.array(edge_dx)
    dy  = np.array(edge_dy)
    sg  = np.array(edge_signs)
    idx = np.array(edge_geom_idx, dtype=int)
    E   = ax.shape[0]

    # kdot_a[e, m, n], kdot_d[e, m, n]
    kdot_a = M[None] * ax[:, None, None] + N[None] * ay[:, None, None]   # (E, O, O)
    kdot_d = M[None] * dx[:, None, None] + N[None] * dy[:, None, None]   # (E, O, O)

    phase0 = np.exp(-1j * np.pi * kdot_a)   # (E, O, O)

    with np.errstate(invalid='ignore', divide='ignore'):
        sinc = np.where(
            np.abs(kdot_d) < eps,
            np.ones((E, O, O), dtype=complex),
            (1.0 - np.exp(-1j * np.pi * kdot_d)) / (1j * np.pi * kdot_d),
        )   # (E, O, O)

    edge_int = phase0 * sinc   # (E, O, O)

    # Contribution tensor:  contrib[e, m, n]
    # For each (m, n):
    #   m != 0:        contrib = dy[e] / (-i pi m) * edge_int
    #   m == 0, n!=0:  contrib = (-dx[e]) / (-i pi n) * edge_int
    #   m == 0, n==0:  contrib = 0.5 * (x[e]*dy_next - y[e]*dx_next)  (shoelace)

    with np.errstate(divide='ignore', invalid='ignore'):
        inv_m = np.where(M != 0, 1.0 / (-1j * np.pi * M), 0.0)   # (O, O)
        inv_n = np.where(N != 0, 1.0 / (-1j * np.pi * N), 0.0)   # (O, O)

    # mask arrays for each branch
    mask_m = (M != 0)                   # (O, O)
    mask_n0 = (M == 0) & (N != 0)      # (O, O)
    mask_00 = (M == 0) & (N == 0)      # (O, O)

    contrib = np.zeros((E, O, O), dtype=np.complex128)

    # branch 1: m != 0
    contrib += (dy[:, None, None] * inv_m[None] * edge_int) * mask_m[None]

    # branch 2: m == 0, n != 0
    contrib += (-dx[:, None, None] * inv_n[None] * edge_int) * mask_n0[None]

    # branch 3: m == n == 0  → shoelace
    shoelace = 0.5 * (ax * dy - ay * dx)   # (E,)  signed area contribution
    contrib[:, mask_00] += shoelace[:, None]

    # Apply per-edge sign: hole edges contribute negatively
    contrib *= sg[:, None, None]

    np.add.at(out, idx, contrib)

    return out


def batch_moments_polygons_legendre(
    polygons: np.ndarray,    # (B, maxK, 2)  padded
    lengths:  np.ndarray,    # (B,)  int
    tables:   MomentTables,
    signs=None,  # (B, maxK) np.ndarray  +1 exterior, -1 hole, 0 sentinel/pad
) -> np.ndarray:             # (B, O, O) float64
    """
    C_leg[b, m, n] = sum_{u,v} A[m,u] * A[n,v] * C_mono[b, u, v]

    Computes all monomial polygon moments first, then applies the Legendre
    expansion as a double matrix multiply — no per-(m,n) Python loop.
    """
    C_mono = batch_moments_polygons_monomial(polygons, lengths, tables, signs=signs)   # (B, O, O)
    A = tables.leg_coef   # (O, O)

    tmp   = np.einsum('mu,buv->bmv', A, C_mono)   # (B, O, O)
    C_leg = np.einsum('nv,bmv->bmn', A, tmp)      # (B, O, O)

    return C_leg


# ---------------------------------------------------------------------------
# Convenience dispatcher  (mirrors GeometryMomentEncoder.encode interface)
# ---------------------------------------------------------------------------

def batch_encode(
    geom: np.ndarray,       # (B, maxPts, 2)  or (B, 2) for points
    lengths: np.ndarray,    # (B,)  ignored for points
    dataset_type: str,      # "points" | "polylines" | "polygons"
    tables: MomentTables,
    basis: str = 'legendre',
    signs=None,             # (B, maxK) np.ndarray for hole-aware polygon encoding
) -> np.ndarray:            # (B, O, O)
    """
    Single entry point that dispatches to the correct vectorized encoder.
    """
    if dataset_type == 'points':
        if basis == 'monomial':
            return batch_moments_points_monomial(geom, tables)
        elif basis == 'fourier':
            return batch_moments_points_fourier(geom, tables)
        elif basis == 'legendre':
            return batch_moments_points_legendre(geom, tables)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    elif dataset_type == 'polylines':
        return batch_moments_polylines(geom, lengths, tables, basis=basis)

    elif dataset_type == 'polygons':
        if basis == 'monomial':
            return batch_moments_polygons_monomial(geom, lengths, tables, signs=signs)
        elif basis == 'fourier':
            return batch_moments_polygons_fourier(geom, lengths, tables, signs=signs)
        elif basis == 'legendre':
            return batch_moments_polygons_legendre(geom, lengths, tables, signs=signs)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
