"""
Vectorized batch 3D moment computations for points, line segments, flat polygons
(3D vertices), and polyhedra (triangular faces).

Each function replaces its scalar counterpart in moments_compute_3d.py but operates
on an entire batch of geometries and the full (O x O x O) moment grid simultaneously.

Shapes legend:
    O  = max_order + 1          (moment grid side length per axis)
    B  = batch size             (number of geometries)
    S  = number of segments     (flat list across all geometries)
    E  = number of triangles    (flat list across all geometries)

Precomputed tables (built once by MomentTables3D):
    leg_coef : (O, O)    float64  — leg_coef[m, u] = coeff of x^u in P_m(x)
    binom    : (O, O)    float64  — binom[n, k] = C(n, k)
    M_grid, N_grid, P_grid : (O, O, O) int meshgrids of order indices
"""

import numpy as np
from math import comb as _comb, factorial as _fact
from numpy.polynomial.polynomial import Polynomial


# ---------------------------------------------------------------------------
# Precomputed tables
# ---------------------------------------------------------------------------

class MomentTables3D:
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
        self.binom = np.zeros((O, O), dtype=np.float64)
        for nn in range(O):
            for kk in range(nn + 1):
                self.binom[nn, kk] = _comb(nn, kk)

        # Extended binomial table for polyhedron formula (needs up to O+1 rows)
        self.binom_ext = np.zeros((O + 2, O + 2), dtype=np.float64)
        for nn in range(O + 2):
            for kk in range(nn + 1):
                self.binom_ext[nn, kk] = _comb(nn, kk)

        # 3D integer order arrays for broadcasting — (O,O,O)
        orders = np.arange(O, dtype=np.float64)
        M, N, P = np.meshgrid(orders, orders, orders, indexing='ij')
        self.M_grid = M   # (O, O, O)
        self.N_grid = N   # (O, O, O)
        self.P_grid = P   # (O, O, O)

        # 1D index arrays
        self.i_idx = np.arange(O, dtype=np.float64)   # (O,)
        self.j_idx = np.arange(O, dtype=np.float64)   # (O,)
        self.k_idx = np.arange(O, dtype=np.float64)   # (O,)

        # Precompute simplex integral table sf3d[I, K] = I! * K! / (I+K+2)!
        # I, K range from 0 to 3*(O-1) for polygons/polyhedra
        max_IK = 3 * (O - 1) + 1
        self.sf3d = np.zeros((max_IK, max_IK), dtype=np.float64)
        for I in range(max_IK):
            for K in range(max_IK):
                self.sf3d[I, K] = _fact(I) * _fact(K) / _fact(I + K + 2)

        # Precompute sf_lookup[ix, iy, iz, kx, ky, kz] = sf3d[ix+iy+iz, kx+ky+kz]
        # Shape (O, O, O, O, O, O) — all indices 0..O-1
        # Used for the 3-step einsum contraction in polygon/polyhedron monomial
        ix_ = np.arange(O)
        iy_ = np.arange(O)
        iz_ = np.arange(O)
        kx_ = np.arange(O)
        ky_ = np.arange(O)
        kz_ = np.arange(O)
        # Build I_sum and K_sum arrays of shape (O,O,O,O,O,O)
        I_sum = (ix_[:, None, None, None, None, None] +
                 iy_[None, :, None, None, None, None] +
                 iz_[None, None, :, None, None, None])
        K_sum = (kx_[None, None, None, :, None, None] +
                 ky_[None, None, None, None, :, None] +
                 kz_[None, None, None, None, None, :])
        # Clip to valid range (entries beyond max_IK-1 will be 0 by the sf3d table)
        I_sum_clipped = np.clip(I_sum, 0, max_IK - 1)
        K_sum_clipped = np.clip(K_sum, 0, max_IK - 1)
        self.sf_lookup = self.sf3d[I_sum_clipped, K_sum_clipped]   # (O,O,O,O,O,O)

        # sf_lookup_poly: extended version for polyhedron formula where ix,kx range 0..O
        # Shape (O+1, O, O, O+1, O, O) — ix,kx can reach O (when m+1 = O)
        ix_p = np.arange(O + 1)
        kx_p = np.arange(O + 1)
        I_sum_p = (ix_p[:, None, None, None, None, None] +
                   iy_[None, :, None, None, None, None] +
                   iz_[None, None, :, None, None, None])
        K_sum_p = (kx_p[None, None, None, :, None, None] +
                   ky_[None, None, None, None, :, None] +
                   kz_[None, None, None, None, None, :])
        I_sum_p_clipped = np.clip(I_sum_p, 0, max_IK - 1)
        K_sum_p_clipped = np.clip(K_sum_p, 0, max_IK - 1)
        self.sf_lookup_poly = self.sf3d[I_sum_p_clipped, K_sum_p_clipped]  # (O+1,O,O,O+1,O,O)


# ---------------------------------------------------------------------------
# Helper: build B^q[q, i, k] = C(q,i)*C(q-i,k) for q up to max_q
# shape: (max_q+1, max_q+1, max_q+1),  zero where i+k > q
# ---------------------------------------------------------------------------

def _build_binom3_table(max_q: int) -> np.ndarray:
    """binom3[q, i, k] = C(q,i)*C(q-i,k)  for i+k<=q, else 0.  Shape (max_q+1, max_q+1, max_q+1)."""
    sz = max_q + 1
    b3 = np.zeros((sz, sz, sz), dtype=np.float64)
    for q in range(sz):
        for i in range(q + 1):
            for k in range(q - i + 1):
                b3[q, i, k] = _comb(q, i) * _comb(q - i, k)
    return b3


# ---------------------------------------------------------------------------
# Points — trivial 3D extension
# ---------------------------------------------------------------------------

def batch_moments_points_monomial_3d(
    points: np.ndarray,        # (B, 3)
    tables: MomentTables3D,
) -> np.ndarray:               # (B, O, O, O)  float64
    """
    result[b, m, n, p] = x[b]^m * y[b]^n * z[b]^p
    """
    x = points[:, 0]    # (B,)
    y = points[:, 1]    # (B,)
    z = points[:, 2]    # (B,)
    M = tables.M_grid   # (O, O, O)
    N = tables.N_grid
    P = tables.P_grid

    return (x[:, None, None, None] ** M[None] *
            y[:, None, None, None] ** N[None] *
            z[:, None, None, None] ** P[None])


def batch_moments_points_fourier_3d(
    points: np.ndarray,        # (B, 3)
    tables: MomentTables3D,
) -> np.ndarray:               # (B, O, O, O)  complex128
    """
    result[b, m, n, p] = exp(-j pi (m*x + n*y + p*z))
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    M = tables.M_grid
    N = tables.N_grid
    P = tables.P_grid

    exponent = (M[None] * x[:, None, None, None] +
                N[None] * y[:, None, None, None] +
                P[None] * z[:, None, None, None])   # (B, O, O, O)
    return np.exp(-1j * np.pi * exponent)


def batch_moments_points_legendre_3d(
    points: np.ndarray,        # (B, 3)
    tables: MomentTables3D,
) -> np.ndarray:               # (B, O, O, O)  float64
    """
    result[b, m, n, p] = P_m(x[b]) * P_n(y[b]) * P_p(z[b])
    Uses leg_coef matrix to evaluate all Legendre polynomials simultaneously.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    O = tables.O
    A = tables.leg_coef   # (O, O)

    p_range = np.arange(O, dtype=np.float64)
    pow_x = x[:, None] ** p_range[None]   # (B, O)
    pow_y = y[:, None] ** p_range[None]   # (B, O)
    pow_z = z[:, None] ** p_range[None]   # (B, O)

    # Pm_vals[b, m] = P_m(x[b])
    Pm_vals = pow_x @ A.T   # (B, O)
    Pn_vals = pow_y @ A.T   # (B, O)
    Pp_vals = pow_z @ A.T   # (B, O)

    # outer product: (B, O, O, O)
    return (Pm_vals[:, :, None, None] *
            Pn_vals[:, None, :, None] *
            Pp_vals[:, None, None, :])


# ---------------------------------------------------------------------------
# Line segments — batch over segments AND all (m,n,p) simultaneously
# ---------------------------------------------------------------------------

def _batch_monomial_line_moments_3d(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,   # (S,)
    dx: np.ndarray, dy: np.ndarray, dz: np.ndarray,   # (S,)
    L:  np.ndarray,                                    # (S,)
    tables: MomentTables3D,
) -> np.ndarray:   # (S, O, O, O)  float64
    """
    Core vectorized monomial line-segment moment for a flat list of S segments.

    C[s,m,n,p] = L[s] * sum_{i,j,k} C(m,i)*C(n,j)*C(p,k)
                           * ax^(m-i)*ay^(n-j)*az^(p-k)
                           * dx^i*dy^j*dz^k / (i+j+k+1)
    """
    S = ax.shape[0]
    O = tables.O
    binom = tables.binom   # (O, O)

    p_range = np.arange(O, dtype=np.float64)
    pow_ax = ax[:, None] ** p_range[None]   # (S, O)
    pow_ay = ay[:, None] ** p_range[None]
    pow_az = az[:, None] ** p_range[None]
    pow_dx = dx[:, None] ** p_range[None]
    pow_dy = dy[:, None] ** p_range[None]
    pow_dz = dz[:, None] ** p_range[None]

    i_idx = np.arange(O)
    j_idx = np.arange(O)
    k_idx = np.arange(O)

    # denom[i, j, k] = 1 / (i + j + k + 1)  shape (O, O, O)
    denom = 1.0 / (i_idx[:, None, None] +
                   j_idx[None, :, None] +
                   k_idx[None, None, :] + 1.0)   # (O, O, O)

    # coeff[m, n, p, i, j, k] = C(m,i)*C(n,j)*C(p,k) / (i+j+k+1)  shape (O,O,O,O,O,O)
    coeff = (binom[:, None, None, :, None, None] *   # C(m,i): (O,1,1,O,1,1)
             binom[None, :, None, None, :, None] *   # C(n,j): (1,O,1,1,O,1)
             binom[None, None, :, None, None, :] *   # C(p,k): (1,1,O,1,1,O)
             denom[None, None, None, :, :, :])       # denom:  (1,1,1,O,O,O)
    # coeff shape: (O, O, O, O, O, O)  = (m, n, p, i, j, k)

    # power index tables
    mi_exp = np.maximum(np.arange(O)[:, None] - i_idx[None, :], 0)   # (O, O)
    nj_exp = np.maximum(np.arange(O)[:, None] - j_idx[None, :], 0)   # (O, O)
    pk_exp = np.maximum(np.arange(O)[:, None] - k_idx[None, :], 0)   # (O, O)

    # A[s, m, i] = ax[s]^(m-i) * dx[s]^i
    pow_ax_mi = pow_ax[:, mi_exp]   # (S, O, O)
    pow_ay_nj = pow_ay[:, nj_exp]   # (S, O, O)
    pow_az_pk = pow_az[:, pk_exp]   # (S, O, O)

    A = pow_ax_mi * pow_dx[:, None, :]    # (S, O, O)  [s, m, i]
    B = pow_ay_nj * pow_dy[:, None, :]    # (S, O, O)  [s, n, j]
    C_z = pow_az_pk * pow_dz[:, None, :] # (S, O, O)  [s, p, k]

    # Contract in 3 steps:
    # Step 1: inner1[s, m, n, p, i, j] = sum_k coeff[m,n,p,i,j,k] * C_z[s,p,k]
    inner1 = np.einsum('mnpijk,spk->smnpij', coeff, C_z)   # (S, O, O, O, O, O)

    # Step 2: inner2[s, m, n, p, i] = sum_j inner1[s,m,n,p,i,j] * B[s,n,j]
    inner2 = np.einsum('smnpij,snj->smnpi', inner1, B)      # (S, O, O, O, O)

    # Step 3: result[s, m, n, p] = sum_i inner2[s,m,n,p,i] * A[s,m,i]
    result = np.einsum('smnpi,smi->smnp', inner2, A)        # (S, O, O, O)

    return L[:, None, None, None] * result


def batch_moments_line_monomial_3d(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
    tables: MomentTables3D,
) -> np.ndarray:   # (S, O, O, O)  float64
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    L  = np.sqrt(dx*dx + dy*dy + dz*dz)
    return _batch_monomial_line_moments_3d(ax, ay, az, dx, dy, dz, L, tables)


def batch_moments_line_fourier_3d(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
    tables: MomentTables3D,
    eps: float = 1e-14,
) -> np.ndarray:   # (S, O, O, O)  complex128
    """
    F[s, m, n, p] = L[s] * exp(-j pi (m*ax + n*ay + p*az))
                          * sinc_term(m*dx + n*dy + p*dz)
    """
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    L  = np.sqrt(dx*dx + dy*dy + dz*dz)   # (S,)

    M = tables.M_grid   # (O, O, O)
    N = tables.N_grid
    P = tables.P_grid

    kdot_a = (M[None] * ax[:, None, None, None] +
              N[None] * ay[:, None, None, None] +
              P[None] * az[:, None, None, None])   # (S, O, O, O)
    kdot_d = (M[None] * dx[:, None, None, None] +
              N[None] * dy[:, None, None, None] +
              P[None] * dz[:, None, None, None])   # (S, O, O, O)

    phase0 = np.exp(-1j * np.pi * kdot_a)   # (S, O, O, O)

    with np.errstate(invalid='ignore', divide='ignore'):
        sinc = np.where(
            np.abs(kdot_d) < eps,
            np.ones_like(kdot_d, dtype=complex),
            (1.0 - np.exp(-1j * np.pi * kdot_d)) / (1j * np.pi * kdot_d),
        )   # (S, O, O, O)

    return L[:, None, None, None] * phase0 * sinc


def batch_moments_line_legendre_3d(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    bx: np.ndarray, by: np.ndarray, bz: np.ndarray,
    tables: MomentTables3D,
) -> np.ndarray:   # (S, O, O, O)  float64
    """
    C_leg[s, m, n, p] = sum_{u,v,w} A[m,u]*A[n,v]*A[p,w] * C_mono[s,u,v,w]
    Done via three sequential matrix contractions.
    """
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    L  = np.sqrt(dx*dx + dy*dy + dz*dz)

    C_mono = _batch_monomial_line_moments_3d(ax, ay, az, dx, dy, dz, L, tables)  # (S,O,O,O)
    A = tables.leg_coef   # (O, O)

    tmp1 = np.einsum('mu,suvw->smvw', A, C_mono)   # (S, O, O, O)
    tmp2 = np.einsum('nv,smvw->smnw', A, tmp1)     # (S, O, O, O)
    C_leg = np.einsum('pw,smnw->smnp', A, tmp2)    # (S, O, O, O)

    return C_leg


# ---------------------------------------------------------------------------
# Polylines — batch over (B geometries × variable segments)
# ---------------------------------------------------------------------------

def batch_moments_polylines_3d(
    polylines: np.ndarray,   # (B, maxS, 3)  padded
    lengths: np.ndarray,     # (B,)  int — actual number of points
    tables: MomentTables3D,
    basis: str = 'legendre',
) -> np.ndarray:             # (B, O, O, O)
    """
    Computes moment grids for a batch of B 3D polylines by:
    1. Flattening all valid segments across all geometries into flat arrays.
    2. Computing all segment moments at once.
    3. Scatter-adding results back to per-geometry output.
    """
    B = polylines.shape[0]
    O = tables.O
    dtype = np.complex128 if basis == 'fourier' else np.float64

    seg_ax, seg_ay, seg_az = [], [], []
    seg_bx, seg_by, seg_bz = [], [], []
    seg_geom_idx = []

    lengths_np = np.asarray(lengths, dtype=int)
    for b in range(B):
        n_pts = int(lengths_np[b])
        if n_pts < 2:
            continue
        pts = polylines[b, :n_pts]   # (n_pts, 3)
        a_pts = pts[:-1]             # (n_pts-1, 3)
        b_pts = pts[1:]              # (n_pts-1, 3)
        n_segs = n_pts - 1
        seg_ax.append(a_pts[:, 0]);  seg_ay.append(a_pts[:, 1]);  seg_az.append(a_pts[:, 2])
        seg_bx.append(b_pts[:, 0]);  seg_by.append(b_pts[:, 1]);  seg_bz.append(b_pts[:, 2])
        seg_geom_idx.extend([b] * n_segs)

    out = np.zeros((B, O, O, O), dtype=dtype)
    if not seg_ax:
        return out

    ax = np.concatenate(seg_ax);  ay = np.concatenate(seg_ay);  az = np.concatenate(seg_az)
    bx = np.concatenate(seg_bx);  by = np.concatenate(seg_by);  bz = np.concatenate(seg_bz)
    idx = np.array(seg_geom_idx, dtype=int)

    if basis == 'monomial':
        seg_moments = batch_moments_line_monomial_3d(ax, ay, az, bx, by, bz, tables)
    elif basis == 'fourier':
        seg_moments = batch_moments_line_fourier_3d(ax, ay, az, bx, by, bz, tables)
    elif basis == 'legendre':
        seg_moments = batch_moments_line_legendre_3d(ax, ay, az, bx, by, bz, tables)
    else:
        raise ValueError(f"Unknown basis: {basis}")

    np.add.at(out, idx, seg_moments)
    return out


# ---------------------------------------------------------------------------
# Flat 3D Polygons — batch over (B geometries × triangles from fan triangulation)
# ---------------------------------------------------------------------------

def _build_triangle_Axyz(
    x0: np.ndarray, dx1: np.ndarray, dx2: np.ndarray,   # (E,) each
    binom_ext: np.ndarray,  # (O+2, O+2)
    O: int,
) -> np.ndarray:   # (E, O, O, O)  where indices are [e, q, ix, kx]
    """
    Build the per-coordinate factor table:
      Ax[e, q, ix, kx] = B^q_{ix,kx} * x0^(q-ix-kx) * dx1^ix * dx2^kx
    where q in 0..O-1, ix in 0..O-1, kx in 0..O-1 (zero when ix+kx > q).
    Shape: (E, O, O, O)  [e, q, ix, kx]
    """
    E = x0.shape[0]
    p_range = np.arange(O, dtype=np.float64)

    pow_x0  = x0[:, None]  ** p_range[None]    # (E, O)
    pow_dx1 = dx1[:, None] ** p_range[None]    # (E, O)
    pow_dx2 = dx2[:, None] ** p_range[None]    # (E, O)

    # q ranges 0..O-1, ix ranges 0..O-1, kx ranges 0..O-1
    q_idx  = np.arange(O)
    ix_idx = np.arange(O)
    kx_idx = np.arange(O)

    # exponent for x0: max(q-ix-kx, 0)
    # shape (O, O, O) for (q, ix, kx)
    exp_x0 = np.maximum(q_idx[:, None, None] - ix_idx[None, :, None] - kx_idx[None, None, :], 0)

    # B^q_{ix,kx} = binom_ext[q, ix] * binom_ext[q-ix, kx], zero when ix+kx > q
    # shape (O, O, O) for (q, ix, kx)
    # binom_ext[q, ix] for q in 0..O-1, ix in 0..O-1
    Bq_ix = binom_ext[q_idx[:, None, None],
                      np.minimum(ix_idx[None, :, None], q_idx[:, None, None])]  # (O,O,O)
    q_minus_ix = np.maximum(q_idx[:, None, None] - ix_idx[None, :, None], 0)    # (O,O,O)
    Bq_kx = binom_ext[q_minus_ix, np.minimum(kx_idx[None, None, :], q_minus_ix)]  # (O,O,O)
    Bq = Bq_ix * Bq_kx  # (O, O, O)

    # Zero out entries where ix+kx > q
    valid_mask = (ix_idx[None, :, None] + kx_idx[None, None, :]) <= q_idx[:, None, None]  # (O,O,O)
    Bq = Bq * valid_mask

    # pow_x0_ext[e, q, ix, kx] = x0[e]^exp_x0[q,ix,kx]
    pow_x0_ext = pow_x0[:, exp_x0]                           # (E, O, O, O)
    pow_dx1_ext = pow_dx1[:, ix_idx][: , :, None].repeat(O, axis=2)  # (E, O, O) -> broadcast needed
    # Use explicit indexing:
    pow_dx1_ext = pow_dx1[:, ix_idx][:, None, :, None] * np.ones((1, O, 1, O))   # (E,O,O,O)
    pow_dx2_ext = pow_dx2[:, kx_idx][:, None, None, :] * np.ones((1, O, O, 1))   # (E,O,O,O)

    # Ax[e, q, ix, kx] = Bq[q, ix, kx] * x0^exp * dx1^ix * dx2^kx
    Ax = Bq[None] * pow_x0_ext * pow_dx1_ext * pow_dx2_ext   # (E, O, O, O)
    return Ax


def _batch_monomial_triangle_moments_3d(
    x0: np.ndarray, y0: np.ndarray, z0: np.ndarray,     # (E,)
    dx1: np.ndarray, dy1: np.ndarray, dz1: np.ndarray,  # (E,)
    dx2: np.ndarray, dy2: np.ndarray, dz2: np.ndarray,  # (E,)
    A_2: np.ndarray,                                     # (E,)  = 2*area
    tables: MomentTables3D,
) -> np.ndarray:   # (E, O, O, O)  float64
    """
    Vectorized monomial surface moment for E triangles simultaneously.

    Uses the 3-step einsum contraction with the precomputed sf_lookup table:
      sf_lookup[ix, iy, iz, kx, ky, kz] = I!*K! / (I+K+2)!
        where I=ix+iy+iz, K=kx+ky+kz

    Step 1: contract z  →  tmp1[e, p, ix, kx, iy, ky]
    Step 2: contract y  →  tmp2[e, n, p, ix, kx]
    Step 3: contract x  →  result[e, m, n, p]
    """
    E = x0.shape[0]
    O = tables.O
    sf_lookup = tables.sf_lookup   # (O, O, O, O, O, O) [ix,iy,iz,kx,ky,kz]
    binom_ext = tables.binom_ext

    # Build per-coordinate A tables: shape (E, O, O, O) [e, q, i_coord, k_coord]
    Ax = _build_triangle_Axyz(x0, dx1, dx2, binom_ext, O)   # (E, O, O, O)
    Ay = _build_triangle_Axyz(y0, dy1, dy2, binom_ext, O)   # (E, O, O, O)
    Az = _build_triangle_Axyz(z0, dz1, dz2, binom_ext, O)   # (E, O, O, O)

    # Index naming convention (all single letters for einsum):
    #   sf_lookup[ix, iy, iz, kx, ky, kz]  →  sf[a, b, c, d, g, h]
    #   Ax[e, m, ix, kx]                    →  Ax[e, m, a, d]
    #   Ay[e, n, iy, ky]                    →  Ay[e, n, b, g]
    #   Az[e, p, iz, kz]                    →  Az[e, p, c, h]
    #
    # result[e, m, n, p] = sum_{a,b,c,d,g,h} Ax[e,m,a,d]*Ay[e,n,b,g]*Az[e,p,c,h]*sf[a,b,c,d,g,h]

    # Step 1: contract iz=c, kz=h (z contribution):
    # tmp1[e, p, a, b, d, g] = sum_{c,h} sf[a,b,c,d,g,h] * Az[e,p,c,h]
    tmp1 = np.einsum('abcdgh,epch->epabdg', sf_lookup, Az)   # (E, O, O, O, O, O)
    # tmp1 shape: (E, p, ix=a, iy=b, kx=d, ky=g)

    # Step 2: contract iy=b, ky=g (y contribution):
    # tmp2[e, n, p, a, d] = sum_{b,g} Ay[e,n,b,g] * tmp1[e,p,a,b,d,g]
    tmp2 = np.einsum('enbg,epabdg->enpad', Ay, tmp1)   # (E, O, O, O, O)
    # tmp2 shape: (E, n, p, ix=a, kx=d)

    # Step 3: contract ix=a, kx=d (x contribution):
    # result[e, m, n, p] = sum_{a,d} Ax[e,m,a,d] * tmp2[e,n,p,a,d]
    result = np.einsum('emad,enpad->emnp', Ax, tmp2)   # (E, O, O, O)

    return A_2[:, None, None, None] * result


def batch_moments_polygons_monomial_3d(
    polygons: np.ndarray,   # (B, maxK, 3)  padded
    lengths: np.ndarray,    # (B,)  int — actual vertex count
    tables: MomentTables3D,
) -> np.ndarray:            # (B, O, O, O)  float64
    """
    Monomial surface moments for a batch of flat 3D polygons.
    Fan-triangulates each polygon from vertex 0, collects all triangles,
    computes moments in one vectorized call, then scatter-adds.
    """
    B = polygons.shape[0]
    O = tables.O

    tri_x0, tri_y0, tri_z0 = [], [], []
    tri_dx1, tri_dy1, tri_dz1 = [], [], []
    tri_dx2, tri_dy2, tri_dz2 = [], [], []
    tri_A2 = []
    tri_geom_idx = []

    lengths_np = np.asarray(lengths, dtype=int)
    for b in range(B):
        K = int(lengths_np[b])
        if K < 3:
            continue
        V = polygons[b, :K]   # (K, 3)
        a = V[0]
        for k in range(1, K - 1):
            bv = V[k]
            cv = V[k + 1]
            e1 = bv - a
            e2 = cv - a
            cross = np.cross(e1, e2)
            A_2 = np.linalg.norm(cross)
            tri_x0.append(a[0]);  tri_y0.append(a[1]);  tri_z0.append(a[2])
            tri_dx1.append(e1[0]); tri_dy1.append(e1[1]); tri_dz1.append(e1[2])
            tri_dx2.append(e2[0]); tri_dy2.append(e2[1]); tri_dz2.append(e2[2])
            tri_A2.append(A_2)
            tri_geom_idx.append(b)

    out = np.zeros((B, O, O, O), dtype=np.float64)
    if not tri_x0:
        return out

    x0  = np.array(tri_x0,  dtype=np.float64)
    y0  = np.array(tri_y0,  dtype=np.float64)
    z0  = np.array(tri_z0,  dtype=np.float64)
    dx1 = np.array(tri_dx1, dtype=np.float64)
    dy1 = np.array(tri_dy1, dtype=np.float64)
    dz1 = np.array(tri_dz1, dtype=np.float64)
    dx2 = np.array(tri_dx2, dtype=np.float64)
    dy2 = np.array(tri_dy2, dtype=np.float64)
    dz2 = np.array(tri_dz2, dtype=np.float64)
    A2  = np.array(tri_A2,  dtype=np.float64)
    idx = np.array(tri_geom_idx, dtype=int)

    tri_moments = _batch_monomial_triangle_moments_3d(
        x0, y0, z0, dx1, dy1, dz1, dx2, dy2, dz2, A2, tables)

    np.add.at(out, idx, tri_moments)
    return out


def _fourier_simplex_I_vec(alpha: np.ndarray, mu: np.ndarray,
                            eps: float = 1e-14) -> np.ndarray:
    """
    Vectorized version of _fourier_simplex_I.
    alpha, mu: arbitrary-shaped numpy arrays (same shape).
    Returns complex array of same shape.
    """
    j = 1j
    shape = alpha.shape
    result = np.zeros(shape, dtype=np.complex128)

    ea  = np.exp(-j * alpha)
    emu = np.exp(-j * mu)

    abs_a  = np.abs(alpha)
    abs_m  = np.abs(mu)
    abs_am = np.abs(alpha - mu)

    # Case 1: alpha=0, mu=0
    mask_00 = (abs_a < eps) & (abs_m < eps)
    result = np.where(mask_00, 0.5 + 0.0j, result)

    # Case 2: alpha=0, mu!=0 → (1 - e^{-jmu} - jmu) / mu^2
    mask_a0 = (abs_a < eps) & (abs_m >= eps)
    with np.errstate(invalid='ignore', divide='ignore'):
        val_a0 = (1.0 - emu - j * mu) / (mu ** 2)
    result = np.where(mask_a0, val_a0, result)

    # Case 3: mu=0, alpha!=0 → (1 - e^{-jalpha} - jalpha) / alpha^2
    mask_m0 = (abs_m < eps) & (abs_a >= eps)
    with np.errstate(invalid='ignore', divide='ignore'):
        val_m0 = (1.0 - ea - j * alpha) / (alpha ** 2)
    result = np.where(mask_m0, val_m0, result)

    # Case 4: alpha=mu!=0
    mask_eq = (abs_am < eps) & (abs_a >= eps)
    with np.errstate(invalid='ignore', divide='ignore'):
        term1_eq = (1.0 - ea) / (j * alpha)
        term2_eq = (ea - 1.0 + j * alpha * ea) / (-alpha ** 2)
        val_eq = (1.0 / (j * alpha)) * (term1_eq + term2_eq)
    result = np.where(mask_eq, val_eq, result)

    # Case 5: general
    mask_gen = (~mask_00) & (~mask_a0) & (~mask_m0) & (~mask_eq)
    with np.errstate(invalid='ignore', divide='ignore'):
        ema = np.exp(-j * (mu - alpha))
        term1_gen = (1.0 - emu) / (j * mu)
        term2_gen = ea * (1.0 - ema) / (j * (mu - alpha))
        val_gen = (1.0 / (j * alpha)) * (term1_gen - term2_gen)
    result = np.where(mask_gen, val_gen, result)

    return result


def batch_moments_polygons_fourier_3d(
    polygons: np.ndarray,   # (B, maxK, 3)  padded
    lengths: np.ndarray,    # (B,)  int
    tables: MomentTables3D,
    eps: float = 1e-14,
) -> np.ndarray:            # (B, O, O, O)  complex128
    """
    Fourier surface moments for a batch of flat 3D polygons.
    Fan-triangulates, collects triangles, computes I(alpha,mu) vectorized.

    C[b,m,n,p] = sum_{tau in b} 2*A_tau * exp(-j*beta_tau) * I(alpha_tau, mu_tau)
    """
    B = polygons.shape[0]
    O = tables.O
    M = tables.M_grid   # (O, O, O)
    N = tables.N_grid
    P = tables.P_grid

    tri_a = []   # vertex a coords
    tri_e1 = []  # edge e1 = b - a
    tri_e2 = []  # edge e2 = c - a
    tri_A2 = []
    tri_geom_idx = []

    lengths_np = np.asarray(lengths, dtype=int)
    for b in range(B):
        K = int(lengths_np[b])
        if K < 3:
            continue
        V = polygons[b, :K]
        a = V[0]
        for k in range(1, K - 1):
            bv = V[k]
            cv = V[k + 1]
            e1 = bv - a
            e2 = cv - a
            cross = np.cross(e1, e2)
            A_2 = np.linalg.norm(cross)
            tri_a.append(a.copy())
            tri_e1.append(e1.copy())
            tri_e2.append(e2.copy())
            tri_A2.append(A_2)
            tri_geom_idx.append(b)

    out = np.zeros((B, O, O, O), dtype=np.complex128)
    if not tri_a:
        return out

    tri_a_arr  = np.array(tri_a,  dtype=np.float64)   # (E, 3)
    tri_e1_arr = np.array(tri_e1, dtype=np.float64)   # (E, 3)
    tri_e2_arr = np.array(tri_e2, dtype=np.float64)   # (E, 3)
    A2  = np.array(tri_A2, dtype=np.float64)          # (E,)
    idx = np.array(tri_geom_idx, dtype=int)

    E = tri_a_arr.shape[0]
    pi = np.pi

    # beta[e, m, n, p] = pi*(m*x0 + n*y0 + p*z0)
    beta = pi * (M[None] * tri_a_arr[:, 0, None, None, None] +
                 N[None] * tri_a_arr[:, 1, None, None, None] +
                 P[None] * tri_a_arr[:, 2, None, None, None])   # (E, O, O, O)

    # alpha[e, m, n, p] = pi*(m*dx1 + n*dy1 + p*dz1)
    alpha = pi * (M[None] * tri_e1_arr[:, 0, None, None, None] +
                  N[None] * tri_e1_arr[:, 1, None, None, None] +
                  P[None] * tri_e1_arr[:, 2, None, None, None])   # (E, O, O, O)

    # mu[e, m, n, p] = pi*(m*dx2 + n*dy2 + p*dz2)
    mu = pi * (M[None] * tri_e2_arr[:, 0, None, None, None] +
               N[None] * tri_e2_arr[:, 1, None, None, None] +
               P[None] * tri_e2_arr[:, 2, None, None, None])   # (E, O, O, O)

    I_val = _fourier_simplex_I_vec(alpha, mu, eps=eps)   # (E, O, O, O)

    # contrib[e, m, n, p] = 2*A2[e] * exp(-j*beta) * I_val
    contrib = A2[:, None, None, None] * np.exp(-1j * beta) * I_val   # (E, O, O, O)

    np.add.at(out, idx, contrib)
    return out


def batch_moments_polygons_legendre_3d(
    polygons: np.ndarray,   # (B, maxK, 3)  padded
    lengths: np.ndarray,    # (B,)  int
    tables: MomentTables3D,
) -> np.ndarray:            # (B, O, O, O)  float64
    """
    C_leg[b, m, n, p] = sum_{u,v,w} A[m,u]*A[n,v]*A[p,w] * C_mono[b,u,v,w]
    """
    C_mono = batch_moments_polygons_monomial_3d(polygons, lengths, tables)   # (B,O,O,O)
    A = tables.leg_coef   # (O, O)

    tmp1 = np.einsum('mu,buvw->bmvw', A, C_mono)
    tmp2 = np.einsum('nv,bmvw->bmnw', A, tmp1)
    C_leg = np.einsum('pw,bmnw->bmnp', A, tmp2)
    return C_leg


# ---------------------------------------------------------------------------
# Polyhedra — batch over (B polyhedra × triangular faces)
# faces: (B, maxF, 3, 3) padded
# ---------------------------------------------------------------------------

def _batch_monomial_polyhedron_face_moments_3d(
    x0: np.ndarray, y0: np.ndarray, z0: np.ndarray,     # (E,) — face vertex a
    dx1: np.ndarray, dy1: np.ndarray, dz1: np.ndarray,  # (E,) — e1 = b - a
    dx2: np.ndarray, dy2: np.ndarray, dz2: np.ndarray,  # (E,) — e2 = c - a
    A_2: np.ndarray,                                     # (E,) = 2*area
    nx: np.ndarray,                                      # (E,) = x-component of unit normal
    tables: MomentTables3D,
) -> np.ndarray:   # (E, O, O, O)  float64
    """
    Computes the face contribution to polyhedron monomial moments (Eq 28) for E faces.

    contrib[e, m, n, p] = (2*A[e]*nx[e]/(m+1)) *
        sum_{i1,i2,j1,j2,k1,k2} B^{m+1}_{i1,i2} * B^n_{j1,j2} * B^p_{k1,k2} *
        x0^(m+1-i1-i2) * ... * simplex(i1+j1+k1, i2+j2+k2)

    Uses the same 3-step einsum strategy as triangles, but with q = m+1 for x.
    We compute the "expanded" triangle monomial moments with x-polynomial degree m+1,
    then divide by (m+1):

    result[e, m, n, p] = A_2[e]*nx[e] * T[e, m+1, n, p] / (m+1)

    where T[e, q, n, p] is the triangle monomial moment with x-degree q.
    We compute T for q = 1..O (one extra degree for x), shape (E, O+1, O, O).
    """
    E = x0.shape[0]
    O = tables.O
    # For polyhedron, ix and kx can reach O (when m+1 = O), so use sf_lookup_poly
    # which has shape (O+1, O, O, O+1, O, O) — ix in 0..O, kx in 0..O, iy/iz/ky/kz in 0..O-1
    sf_lp = tables.sf_lookup_poly   # (O+1, O, O, O+1, O, O)
    binom_ext = tables.binom_ext    # (O+2, O+2)

    p_range_ext = np.arange(O + 1, dtype=np.float64)   # 0..O (for x powers)
    p_range     = np.arange(O,     dtype=np.float64)   # 0..O-1

    # Power tables for x: degree up to O
    pow_x0_ext  = x0[:, None]  ** p_range_ext[None]   # (E, O+1)
    pow_dx1_ext = dx1[:, None] ** p_range_ext[None]   # (E, O+1)
    pow_dx2_ext = dx2[:, None] ** p_range_ext[None]   # (E, O+1)

    # Build Ay and Az for y,z (degree 0..O-1 each)
    Ay = _build_triangle_Axyz(y0, dy1, dy2, binom_ext, O)   # (E, O, O, O)
    Az = _build_triangle_Axyz(z0, dz1, dz2, binom_ext, O)   # (E, O, O, O)

    # Build Ax_ext[e, q, ix, kx] for q in 0..O, ix in 0..O, kx in 0..O
    # B^q_{ix,kx} * x0^(q-ix-kx) * dx1^ix * dx2^kx
    # Shape: (E, O+1, O+1, O+1)
    Oext = O + 1
    q_ext  = np.arange(Oext)   # 0..O
    ix_ext = np.arange(Oext)   # 0..O
    kx_ext = np.arange(Oext)   # 0..O

    exp_x0_ext = np.maximum(q_ext[:, None, None] - ix_ext[None, :, None] - kx_ext[None, None, :], 0)  # (O+1,O+1,O+1)
    Bqx_ix = binom_ext[q_ext[:, None, None],
                        np.minimum(ix_ext[None, :, None], q_ext[:, None, None])]   # (O+1,O+1,O+1)
    q_minus_ix_ext = np.maximum(q_ext[:, None, None] - ix_ext[None, :, None], 0)
    Bqx_kx = binom_ext[q_minus_ix_ext, np.minimum(kx_ext[None, None, :], q_minus_ix_ext)]  # (O+1,O+1,O+1)
    Bqx = Bqx_ix * Bqx_kx
    valid_x_ext = (ix_ext[None, :, None] + kx_ext[None, None, :]) <= q_ext[:, None, None]
    Bqx = Bqx * valid_x_ext   # (O+1, O+1, O+1)

    pow_x0_tab  = pow_x0_ext[:, exp_x0_ext]                                         # (E, O+1, O+1, O+1)
    pow_dx1_tab = pow_dx1_ext[:, ix_ext][: , None, :, None] * np.ones((1, Oext, 1, Oext))  # (E,O+1,O+1,O+1)
    pow_dx2_tab = pow_dx2_ext[:, kx_ext][:, None, None, :]  * np.ones((1, Oext, Oext, 1))  # (E,O+1,O+1,O+1)

    Ax_ext = Bqx[None] * pow_x0_tab * pow_dx1_tab * pow_dx2_tab   # (E, O+1, O+1, O+1)

    # Contract using sf_lookup_poly[ix=a, iy=b, iz=c, kx=d, ky=g, kz=h]
    # with ix,kx in 0..O (size O+1), iy,iz,ky,kz in 0..O-1 (size O)
    # sf_lp shape: (O+1, O, O, O+1, O, O)
    #
    # Step 1: contract iz=c, kz=h (Az has indices [e,p,iz,kz] with iz,kz in 0..O-1):
    # tmp1[e, p, a, b, d, g] = sum_{c,h} sf_lp[a,b,c,d,g,h] * Az[e,p,c,h]
    # sf_lp dims: (O+1, O, O, O+1, O, O), Az dims: (E, O, O, O) [e,p,c,h]
    tmp1 = np.einsum('abcdgh,epch->epabdg', sf_lp, Az)   # (E, O, O+1, O, O+1, O)
    # tmp1 shape: [e, p, ix=a(0..O), iy=b(0..O-1), kx=d(0..O), ky=g(0..O-1)]

    # Step 2: contract iy=b, ky=g (Ay has [e,n,iy,ky] with iy,ky in 0..O-1):
    # tmp2[e, n, p, a, d] = sum_{b,g} Ay[e,n,b,g] * tmp1[e,p,a,b,d,g]
    tmp2 = np.einsum('enbg,epabdg->enpad', Ay, tmp1)   # (E, O, O, O+1, O+1)
    # tmp2 shape: [e, n, p, ix=a(0..O), kx=d(0..O)]

    # Step 3: contract ix=a, kx=d (Ax_ext has [e,q,ix,kx] with ix,kx in 0..O):
    # result_ext[e, q, n, p] = sum_{a,d} Ax_ext[e,q,a,d] * tmp2[e,n,p,a,d]
    result_ext = np.einsum('eqad,enpad->eqnp', Ax_ext, tmp2)   # (E, O+1, O, O)

    # Map back: face contribution to C[e, m, n, p]:
    # contrib[e, m, n, p] = (A_2[e]*nx[e]/(m+1)) * result_ext[e, m+1, n, p]
    m_arr = np.arange(O, dtype=np.float64)   # (O,)  m = 0..O-1
    T = result_ext[:, 1:, :, :]   # (E, O, O, O)  T[e,m,n,p] = result_ext[e,m+1,n,p]

    factor = A_2[:, None, None, None] * nx[:, None, None, None] / (m_arr[None, :, None, None] + 1.0)
    return factor * T


def batch_moments_polyhedra_monomial_3d(
    faces: np.ndarray,        # (B, maxF, 3, 3)  padded
    face_counts: np.ndarray,  # (B,)  int — actual number of faces
    tables: MomentTables3D,
) -> np.ndarray:              # (B, O, O, O)  float64
    """
    Monomial volume moments for a batch of polyhedra.
    Each polyhedron is given as triangular faces (outward CCW normals).
    """
    B = faces.shape[0]
    O = tables.O

    f_x0, f_y0, f_z0 = [], [], []
    f_dx1, f_dy1, f_dz1 = [], [], []
    f_dx2, f_dy2, f_dz2 = [], [], []
    f_A2, f_nx = [], []
    f_geom_idx = []

    fc_np = np.asarray(face_counts, dtype=int)
    for b in range(B):
        nf = int(fc_np[b])
        for f in range(nf):
            a = faces[b, f, 0]
            bv = faces[b, f, 1]
            cv = faces[b, f, 2]
            e1 = bv - a
            e2 = cv - a
            cross = np.cross(e1, e2)
            A_2 = np.linalg.norm(cross)
            if A_2 == 0.0:
                continue
            n_hat = cross / A_2
            f_x0.append(a[0]);   f_y0.append(a[1]);   f_z0.append(a[2])
            f_dx1.append(e1[0]); f_dy1.append(e1[1]); f_dz1.append(e1[2])
            f_dx2.append(e2[0]); f_dy2.append(e2[1]); f_dz2.append(e2[2])
            f_A2.append(A_2)
            f_nx.append(n_hat[0])
            f_geom_idx.append(b)

    out = np.zeros((B, O, O, O), dtype=np.float64)
    if not f_x0:
        return out

    x0  = np.array(f_x0,  dtype=np.float64)
    y0  = np.array(f_y0,  dtype=np.float64)
    z0  = np.array(f_z0,  dtype=np.float64)
    dx1 = np.array(f_dx1, dtype=np.float64)
    dy1 = np.array(f_dy1, dtype=np.float64)
    dz1 = np.array(f_dz1, dtype=np.float64)
    dx2 = np.array(f_dx2, dtype=np.float64)
    dy2 = np.array(f_dy2, dtype=np.float64)
    dz2 = np.array(f_dz2, dtype=np.float64)
    A2  = np.array(f_A2,  dtype=np.float64)
    nx  = np.array(f_nx,  dtype=np.float64)
    idx = np.array(f_geom_idx, dtype=int)

    face_moments = _batch_monomial_polyhedron_face_moments_3d(
        x0, y0, z0, dx1, dy1, dz1, dx2, dy2, dz2, A2, nx, tables)

    np.add.at(out, idx, face_moments)
    return out


def batch_moments_polyhedra_fourier_3d(
    faces: np.ndarray,        # (B, maxF, 3, 3)  padded
    face_counts: np.ndarray,  # (B,)  int
    tables: MomentTables3D,
    eps: float = 1e-14,
) -> np.ndarray:              # (B, O, O, O)  complex128
    """
    Fourier volume moments for a batch of polyhedra (Eq 31).

    For m=n=p=0: returns the volume (computed via monomial m=n=p=0).
    Otherwise:
      C[b,m,n,p] = 1/(-j pi (m^2+n^2+p^2)) *
                   sum_{f in b} 2*A_f*(m*nx+n*ny+p*nz)*exp(-j*beta_f)*I(alpha_f,mu_f)
    """
    B = faces.shape[0]
    O = tables.O
    M = tables.M_grid   # (O, O, O)
    N = tables.N_grid
    P = tables.P_grid

    f_a = []
    f_e1 = []
    f_e2 = []
    f_A2 = []
    f_nx, f_ny, f_nz = [], [], []
    f_geom_idx = []

    fc_np = np.asarray(face_counts, dtype=int)
    for b in range(B):
        nf = int(fc_np[b])
        for f in range(nf):
            a = faces[b, f, 0]
            bv = faces[b, f, 1]
            cv = faces[b, f, 2]
            e1 = bv - a
            e2 = cv - a
            cross = np.cross(e1, e2)
            A_2 = np.linalg.norm(cross)
            if A_2 == 0.0:
                continue
            n_hat = cross / A_2
            f_a.append(a.copy())
            f_e1.append(e1.copy())
            f_e2.append(e2.copy())
            f_A2.append(A_2)
            f_nx.append(n_hat[0])
            f_ny.append(n_hat[1])
            f_nz.append(n_hat[2])
            f_geom_idx.append(b)

    out = np.zeros((B, O, O, O), dtype=np.complex128)
    if not f_a:
        return out

    a_arr  = np.array(f_a,  dtype=np.float64)   # (E, 3)
    e1_arr = np.array(f_e1, dtype=np.float64)   # (E, 3)
    e2_arr = np.array(f_e2, dtype=np.float64)   # (E, 3)
    A2 = np.array(f_A2, dtype=np.float64)       # (E,)
    nx = np.array(f_nx, dtype=np.float64)       # (E,)
    ny = np.array(f_ny, dtype=np.float64)
    nz = np.array(f_nz, dtype=np.float64)
    idx = np.array(f_geom_idx, dtype=int)

    E = a_arr.shape[0]
    pi = np.pi

    beta = pi * (M[None] * a_arr[:, 0, None, None, None] +
                 N[None] * a_arr[:, 1, None, None, None] +
                 P[None] * a_arr[:, 2, None, None, None])   # (E, O, O, O)

    alpha = pi * (M[None] * e1_arr[:, 0, None, None, None] +
                  N[None] * e1_arr[:, 1, None, None, None] +
                  P[None] * e1_arr[:, 2, None, None, None])   # (E, O, O, O)

    mu = pi * (M[None] * e2_arr[:, 0, None, None, None] +
               N[None] * e2_arr[:, 1, None, None, None] +
               P[None] * e2_arr[:, 2, None, None, None])   # (E, O, O, O)

    I_val = _fourier_simplex_I_vec(alpha, mu, eps=eps)   # (E, O, O, O)

    # kdot_n[e, m, n, p] = m*nx + n*ny + p*nz
    kdot_n = (M[None] * nx[:, None, None, None] +
              N[None] * ny[:, None, None, None] +
              P[None] * nz[:, None, None, None])   # (E, O, O, O)

    # face_contrib[e, m, n, p] = A2[e] * kdot_n * exp(-j*beta) * I_val
    # (A2 = ||cross|| = 2*area, so this equals 2*A_f*kdot_n*... per Eq 31)
    face_contrib = (A2[:, None, None, None] * kdot_n *
                    np.exp(-1j * beta) * I_val)   # (E, O, O, O)

    # Accumulate into output (we'll divide by denominator after scatter-add)
    np.add.at(out, idx, face_contrib)

    # Divide by -j*pi*(m^2+n^2+p^2) for all (m,n,p) != (0,0,0)
    mnp_sq = M ** 2 + N ** 2 + P ** 2   # (O, O, O)
    mask_nonzero = mnp_sq != 0
    with np.errstate(divide='ignore', invalid='ignore'):
        denom = np.where(mask_nonzero, -1j * pi * mnp_sq, 1.0)   # avoid division by 0
    out = np.where(mask_nonzero[None], out / denom[None], out)

    # For m=n=p=0: return volume via monomial (0,0,0) moment
    if O > 0:
        vol = batch_moments_polyhedra_monomial_3d(faces, face_counts, tables)[:, 0, 0, 0]
        out[:, 0, 0, 0] = vol.astype(np.complex128)

    return out


def batch_moments_polyhedra_legendre_3d(
    faces: np.ndarray,        # (B, maxF, 3, 3)  padded
    face_counts: np.ndarray,  # (B,)  int
    tables: MomentTables3D,
) -> np.ndarray:              # (B, O, O, O)  float64
    """
    C_leg[b, m, n, p] = sum_{u,v,w} A[m,u]*A[n,v]*A[p,w] * C_mono[b,u,v,w]
    """
    C_mono = batch_moments_polyhedra_monomial_3d(faces, face_counts, tables)   # (B,O,O,O)
    A = tables.leg_coef   # (O, O)

    tmp1 = np.einsum('mu,buvw->bmvw', A, C_mono)
    tmp2 = np.einsum('nv,bmvw->bmnw', A, tmp1)
    C_leg = np.einsum('pw,bmnw->bmnp', A, tmp2)
    return C_leg


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

def batch_encode_3d(
    geom: np.ndarray,       # (B, maxPts, 3) or (B, 3) for points
    lengths: np.ndarray,    # (B,)  ignored for points
    dataset_type: str,      # "points" | "polylines" | "polygons" | "polyhedra"
    tables: MomentTables3D,
    basis: str = 'legendre',
    face_counts: np.ndarray = None,  # required when dataset_type == "polyhedra"
) -> np.ndarray:            # (B, O, O, O)
    """
    Single entry point that dispatches to the correct vectorized 3D encoder.
    """
    if dataset_type == 'points':
        if basis == 'monomial':
            return batch_moments_points_monomial_3d(geom, tables)
        elif basis == 'fourier':
            return batch_moments_points_fourier_3d(geom, tables)
        elif basis == 'legendre':
            return batch_moments_points_legendre_3d(geom, tables)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    elif dataset_type == 'polylines':
        return batch_moments_polylines_3d(geom, lengths, tables, basis=basis)

    elif dataset_type == 'polygons':
        if basis == 'monomial':
            return batch_moments_polygons_monomial_3d(geom, lengths, tables)
        elif basis == 'fourier':
            return batch_moments_polygons_fourier_3d(geom, lengths, tables)
        elif basis == 'legendre':
            return batch_moments_polygons_legendre_3d(geom, lengths, tables)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    elif dataset_type == 'polyhedra':
        if face_counts is None:
            raise ValueError("face_counts is required for polyhedra")
        if basis == 'monomial':
            return batch_moments_polyhedra_monomial_3d(geom, face_counts, tables)
        elif basis == 'fourier':
            return batch_moments_polyhedra_fourier_3d(geom, face_counts, tables)
        elif basis == 'legendre':
            return batch_moments_polyhedra_legendre_3d(geom, face_counts, tables)
        else:
            raise ValueError(f"Unknown basis: {basis}")

    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
