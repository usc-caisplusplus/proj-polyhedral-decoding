"""
Scalar 3D moment computations for points, line segments, flat polygons (3D vertices),
and polyhedra (triangular faces).

Implements the mathematical derivations from:
  "Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications"
  (ICML 2025), Equations (12)-(23), (28), (30)-(32).

Bases are three-index (m, n, p):
  Monomial : phi(x,y,z) = x^m * y^n * z^p
  Fourier  : phi(x,y,z) = exp(-j pi (m x + n y + p z))
  Legendre : phi(x,y,z) = P_m(x) * P_n(y) * P_p(z)
"""

import math
import numpy as np
from numpy.polynomial.polynomial import Polynomial


# ---------------------------------------------------------------------------
# Basis functions — Eq (12)(13)(14)
# ---------------------------------------------------------------------------

def basis_monomial_3d(x: float, y: float, z: float,
                      m: int, n: int, p: int) -> float:
    """phi(x,y,z) = x^m * y^n * z^p"""
    return (x ** m) * (y ** n) * (z ** p)


def basis_fourier_3d(x: float, y: float, z: float,
                     m: int, n: int, p: int) -> complex:
    """phi(x,y,z) = exp(-j pi (m x + n y + p z))"""
    return np.exp(-1j * math.pi * (m * x + n * y + p * z))


def basis_legendre_3d(x: float, y: float, z: float,
                      m: int, n: int, p: int) -> float:
    """phi(x,y,z) = P_m(x) * P_n(y) * P_p(z)"""
    Pm = np.polynomial.legendre.Legendre.basis(m)
    Pn = np.polynomial.legendre.Legendre.basis(n)
    Pp = np.polynomial.legendre.Legendre.basis(p)
    return float(Pm(x)) * float(Pn(y)) * float(Pp(z))


# ---------------------------------------------------------------------------
# Point moments — Eq (12)(13)(14): C_mnp = phi(x0, y0, z0)
# ---------------------------------------------------------------------------

def moment_of_point_monomial_3d(x: float, y: float, z: float,
                                 m: int, n: int, p: int) -> float:
    """Monomial moment of a point: x0^m * y0^n * z0^p"""
    return basis_monomial_3d(x, y, z, m, n, p)


def moment_of_point_fourier_3d(x: float, y: float, z: float,
                                m: int, n: int, p: int) -> complex:
    """Fourier moment of a point: exp(-j pi (m x + n y + p z))"""
    return basis_fourier_3d(x, y, z, m, n, p)


def moment_of_point_legendre_3d(x: float, y: float, z: float,
                                 m: int, n: int, p: int) -> float:
    """Legendre moment of a point: P_m(x) * P_n(y) * P_p(z)"""
    return basis_legendre_3d(x, y, z, m, n, p)


# ---------------------------------------------------------------------------
# Line segment moments — Eq (16)(17)(18)
# Segment from a=(ax,ay,az) to b=(bx,by,bz)
# d = b - a,  L = ||d||
# ---------------------------------------------------------------------------

def moment_of_line_segment_monomial_3d(
        ax: float, ay: float, az: float,
        bx: float, by: float, bz: float,
        m: int, n: int, p: int) -> float:
    """
    Monomial moment of a 3D line segment (Eq 16):

      C_mnp = L * sum_{i=0}^{m} sum_{j=0}^{n} sum_{k=0}^{p}
                C(m,i)*C(n,j)*C(p,k) * ax^(m-i)*ay^(n-j)*az^(p-k)
                                      * dx^i*dy^j*dz^k / (i+j+k+1)
    """
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    L = math.sqrt(dx*dx + dy*dy + dz*dz)

    total = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            for k in range(p + 1):
                total += (math.comb(m, i) * math.comb(n, j) * math.comb(p, k) *
                          (ax ** (m - i)) * (ay ** (n - j)) * (az ** (p - k)) *
                          (dx ** i) * (dy ** j) * (dz ** k) /
                          (i + j + k + 1))
    return L * total


def moment_of_line_segment_fourier_3d(
        ax: float, ay: float, az: float,
        bx: float, by: float, bz: float,
        m: int, n: int, p: int) -> complex:
    """
    Fourier moment of a 3D line segment (Eq 17):

      alpha = pi*(m*dx + n*dy + p*dz)
      beta0 = pi*(m*ax + n*ay + p*az)
      C_mnp = L * exp(-j*beta0) * { (1-exp(-j*alpha))/(j*alpha)  if alpha!=0, else 1 }
    """
    dx = bx - ax
    dy = by - ay
    dz = bz - az
    L = math.sqrt(dx*dx + dy*dy + dz*dz)

    kdot_a = m * ax + n * ay + p * az
    kdot_d = m * dx + n * dy + p * dz
    phase0 = np.exp(-1j * math.pi * kdot_a)

    if abs(kdot_d) < 1e-14:
        return L * phase0
    num = 1.0 - np.exp(-1j * math.pi * kdot_d)
    den = 1j * math.pi * kdot_d
    return L * phase0 * (num / den)


def moment_of_line_segment_legendre_3d(
        ax: float, ay: float, az: float,
        bx: float, by: float, bz: float,
        m: int, n: int, p: int) -> float:
    """
    Legendre moment of a 3D line segment (Eq 18):
    Expands P_m, P_n, P_p into monomials and calls the monomial formula.

      C_mnp^seg,L = sum_{u,v,w} cm[u]*cn[v]*cp[w] * C_uvw^seg,geo
    """
    Pm_poly = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
    Pn_poly = np.polynomial.legendre.Legendre.basis(n).convert(kind=Polynomial)
    Pp_poly = np.polynomial.legendre.Legendre.basis(p).convert(kind=Polynomial)

    cm = Pm_poly.coef
    cn = Pn_poly.coef
    cp = Pp_poly.coef

    total = 0.0
    for u, cmu in enumerate(cm):
        if cmu == 0.0:
            continue
        for v, cnv in enumerate(cn):
            if cnv == 0.0:
                continue
            for w, cpw in enumerate(cp):
                if cpw == 0.0:
                    continue
                total += (cmu * cnv * cpw *
                          moment_of_line_segment_monomial_3d(ax, ay, az, bx, by, bz, u, v, w))
    return total


# ---------------------------------------------------------------------------
# Flat polygon moments (surface integral, vertices in 3D)
# Polygon vertices: np.ndarray shape (K, 3)
# Fan triangulation from vertex 0: triangles (V[0], V[k], V[k+1]) for k=1..K-2
# ---------------------------------------------------------------------------

def _simplex_integral_2d(i: int, j: int) -> float:
    """
    Integral of s^i * t^j over the standard 2-simplex T = {s>=0, t>=0, s+t<=1}:
      result = i! * j! / (i+j+2)!
    """
    return math.factorial(i) * math.factorial(j) / math.factorial(i + j + 2)


def _fourier_simplex_I(alpha: float, mu: float, eps: float = 1e-14) -> complex:
    """
    Integral of exp(-j*(alpha*s + mu*t)) over the standard 2-simplex T
    (Eq 30 from the paper, same as Eq 21 for polygons).

    Cases:
      alpha=0, mu=0        : 1/2
      alpha=0, mu!=0       : (1 - e^{-jmu} - jmu) / mu^2
      mu=0, alpha!=0       : (1 - e^{-jalpha} - jalpha) / alpha^2
      alpha=mu!=0          : (1/(jalpha)) * [ (1-e^{-jalpha})/(jalpha)
                               + (e^{-jalpha}-1+jalpha*e^{-jalpha}) / (-alpha^2) ]
      general (all diff/0) : (1/(jalpha)) * [ (1-e^{-jmu})/(jmu)
                               - e^{-jalpha}*(1-e^{-j(mu-alpha)})/(j*(mu-alpha)) ]
    """
    j_unit = 1j
    if abs(alpha) < eps and abs(mu) < eps:
        return 0.5 + 0.0j

    if abs(alpha) < eps:
        # alpha=0, mu!=0: (1 - e^{-jmu} - jmu) / mu^2
        emu = np.exp(-j_unit * mu)
        return (1.0 - emu - j_unit * mu) / (mu ** 2)

    if abs(mu) < eps:
        # mu=0, alpha!=0: (1 - e^{-jalpha} - jalpha) / alpha^2
        ea = np.exp(-j_unit * alpha)
        return (1.0 - ea - j_unit * alpha) / (alpha ** 2)

    if abs(alpha - mu) < eps:
        # alpha=mu!=0
        # (1/jalpha) * [ (1-e^{-jalpha})/(jalpha) + (e^{-jalpha}-1+jalpha*e^{-jalpha}) / (-alpha^2) ]
        ea = np.exp(-j_unit * alpha)
        term1 = (1.0 - ea) / (j_unit * alpha)
        term2 = (ea - 1.0 + j_unit * alpha * ea) / (-alpha ** 2)
        return (1.0 / (j_unit * alpha)) * (term1 + term2)

    # General: alpha!=0, mu!=0, alpha!=mu
    # (1/jalpha) * [ (1-e^{-jmu})/(jmu) - e^{-jalpha}*(1-e^{-j(mu-alpha)})/(j*(mu-alpha)) ]
    ea = np.exp(-j_unit * alpha)
    emu = np.exp(-j_unit * mu)
    ema = np.exp(-j_unit * (mu - alpha))
    term1 = (1.0 - emu) / (j_unit * mu)
    term2 = ea * (1.0 - ema) / (j_unit * (mu - alpha))
    return (1.0 / (j_unit * alpha)) * (term1 - term2)


def _triangle_monomial_surface_3d(
        x0: float, y0: float, z0: float,
        dx1: float, dy1: float, dz1: float,
        dx2: float, dy2: float, dz2: float,
        m: int, n: int, p: int,
        A_2: float) -> float:
    """
    Monomial surface integral over a triangle (Eq 20).

    Triangle vertices: a=(x0,y0,z0), b=a+e1, c=a+e2
    where e1=(dx1,dy1,dz1), e2=(dx2,dy2,dz2).
    A_2 = 2 * triangle_area = ||e1 x e2||

    C_mnp^triangle,geo = A_2 * sum_{ix,kx,iy,ky,iz,kz} B^m_{ix,kx} * B^n_{iy,ky} * B^p_{iz,kz}
                           * x0^(m-ix-kx) * y0^(n-iy-ky) * z0^(p-iz-kz)
                           * dx1^ix * dx2^kx * dy1^iy * dy2^ky * dz1^iz * dz2^kz
                           * simplex(ix+iy+iz, kx+ky+kz)

    where B^q_{i,k} = C(q,i)*C(q-i,k)  and  simplex(I,K) = I!*K!/(I+K+2)!
    """
    total = 0.0
    for ix in range(m + 1):
        for kx in range(m - ix + 1):
            Bmixkx = math.comb(m, ix) * math.comb(m - ix, kx)
            x0_pow = x0 ** (m - ix - kx)
            dx1_pow = dx1 ** ix
            dx2_pow = dx2 ** kx
            for iy in range(n + 1):
                for ky in range(n - iy + 1):
                    Bniyka = math.comb(n, iy) * math.comb(n - iy, ky)
                    y0_pow = y0 ** (n - iy - ky)
                    dy1_pow = dy1 ** iy
                    dy2_pow = dy2 ** ky
                    for iz in range(p + 1):
                        for kz in range(p - iz + 1):
                            Bpizka = math.comb(p, iz) * math.comb(p - iz, kz)
                            z0_pow = z0 ** (p - iz - kz)
                            dz1_pow = dz1 ** iz
                            dz2_pow = dz2 ** kz
                            I = ix + iy + iz
                            K = kx + ky + kz
                            sf = _simplex_integral_2d(I, K)
                            total += (Bmixkx * Bniyka * Bpizka *
                                      x0_pow * y0_pow * z0_pow *
                                      dx1_pow * dx2_pow *
                                      dy1_pow * dy2_pow *
                                      dz1_pow * dz2_pow *
                                      sf)
    return A_2 * total


def moment_of_polygon_monomial_3d(vertices: np.ndarray,
                                   m: int, n: int, p: int) -> float:
    """
    Monomial surface moment of a flat 3D polygon (Eq 20).
    vertices: np.ndarray shape (K, 3) — ordered vertices, K >= 3.
    Fan triangulation from vertex 0.
    """
    V = np.asarray(vertices, dtype=float)
    K = V.shape[0]
    total = 0.0
    for k in range(1, K - 1):
        a = V[0]
        b = V[k]
        c = V[k + 1]
        e1 = b - a
        e2 = c - a
        cross = np.cross(e1, e2)
        A_2 = np.linalg.norm(cross)  # = 2 * triangle area
        total += _triangle_monomial_surface_3d(
            a[0], a[1], a[2],
            e1[0], e1[1], e1[2],
            e2[0], e2[1], e2[2],
            m, n, p, A_2)
    return total


def moment_of_polygon_fourier_3d(vertices: np.ndarray,
                                  m: int, n: int, p: int) -> complex:
    """
    Fourier surface moment of a flat 3D polygon (Eq 22).
    vertices: np.ndarray shape (K, 3).
    Fan triangulation from vertex 0.

    C_mnp = sum_tau 2*A_tau * exp(-j*beta_tau) * I(alpha_tau, mu_tau)
    where:
      beta_tau = pi*(m*x0 + n*y0 + p*z0)
      alpha_tau = pi*(m*dx1 + n*dy1 + p*dz1)
      mu_tau    = pi*(m*dx2 + n*dy2 + p*dz2)
    """
    V = np.asarray(vertices, dtype=float)
    K = V.shape[0]
    total = 0.0 + 0.0j
    pi = math.pi

    for k in range(1, K - 1):
        a = V[0]
        b = V[k]
        c = V[k + 1]
        e1 = b - a
        e2 = c - a
        cross = np.cross(e1, e2)
        A_2 = np.linalg.norm(cross)  # = 2 * triangle area

        beta = pi * (m * a[0] + n * a[1] + p * a[2])
        alpha = pi * (m * e1[0] + n * e1[1] + p * e1[2])
        mu = pi * (m * e2[0] + n * e2[1] + p * e2[2])

        total += A_2 * np.exp(-1j * beta) * _fourier_simplex_I(alpha, mu)

    return total


def moment_of_polygon_legendre_3d(vertices: np.ndarray,
                                   m: int, n: int, p: int) -> float:
    """
    Legendre surface moment of a flat 3D polygon (Eq 23).
    Expands P_m, P_n, P_p into monomials, then calls the monomial version.

      C_mnp^pgon,L = sum_{u,v,w} cm[u]*cn[v]*cp[w] * C_uvw^pgon,geo
    """
    Pm_poly = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
    Pn_poly = np.polynomial.legendre.Legendre.basis(n).convert(kind=Polynomial)
    Pp_poly = np.polynomial.legendre.Legendre.basis(p).convert(kind=Polynomial)

    cm = Pm_poly.coef
    cn = Pn_poly.coef
    cp = Pp_poly.coef

    total = 0.0
    for u, cmu in enumerate(cm):
        if cmu == 0.0:
            continue
        for v, cnv in enumerate(cn):
            if cnv == 0.0:
                continue
            for w, cpw in enumerate(cp):
                if cpw == 0.0:
                    continue
                total += cmu * cnv * cpw * moment_of_polygon_monomial_3d(vertices, u, v, w)
    return total


# ---------------------------------------------------------------------------
# Polyhedron moments (volume integral via Divergence Theorem)
# faces: np.ndarray shape (F, 3, 3)
#   faces[f] = [[x0,y0,z0], [x1,y1,z1], [x2,y2,z2]] — outward-facing CCW normals
# ---------------------------------------------------------------------------

def moment_of_polyhedron_monomial_3d(faces: np.ndarray,
                                      m: int, n: int, p: int) -> float:
    """
    Monomial volume moment of a polyhedron (Eq 28).
    faces: np.ndarray shape (F, 3, 3).

    Uses the divergence theorem with F_x = x^{m+1}*y^n*z^p/(m+1), so:

    C_mnp = sum_f (2*A_f*nx_f/(m+1)) *
            sum_{i1,i2,j1,j2,k1,k2} B^{m+1}_{i1,i2} * B^n_{j1,j2} * B^p_{k1,k2}
            * x0^(m+1-i1-i2) * y0^(n-j1-j2) * z0^(p-k1-k2)
            * dx1^i1 * dx2^i2 * dy1^j1 * dy2^j2 * dz1^k1 * dz2^k2
            * (i1+j1+k1)! * (i2+j2+k2)! / (i1+j1+k1+i2+j2+k2+2)!
    """
    F_arr = np.asarray(faces, dtype=float)
    total = 0.0
    mp1 = m + 1  # m+1 for the divergence theorem expansion

    for f in range(F_arr.shape[0]):
        a = F_arr[f, 0]
        b = F_arr[f, 1]
        c = F_arr[f, 2]
        e1 = b - a
        e2 = c - a
        cross = np.cross(e1, e2)
        A_2 = np.linalg.norm(cross)  # = 2 * triangle area
        if A_2 == 0.0:
            continue
        n_hat = cross / A_2  # unit normal (already normalized by ||cross||)
        nx_f = n_hat[0]

        if nx_f == 0.0:
            continue  # this face contributes nothing to the x-divergence branch

        x0, y0, z0 = a[0], a[1], a[2]
        dx1, dy1, dz1 = e1[0], e1[1], e1[2]
        dx2, dy2, dz2 = e2[0], e2[1], e2[2]

        face_sum = 0.0
        for i1 in range(mp1 + 1):
            for i2 in range(mp1 - i1 + 1):
                Bm1_i1i2 = math.comb(mp1, i1) * math.comb(mp1 - i1, i2)
                xpow = x0 ** (mp1 - i1 - i2)
                dx1_pow = dx1 ** i1
                dx2_pow = dx2 ** i2
                for j1 in range(n + 1):
                    for j2 in range(n - j1 + 1):
                        Bn_j1j2 = math.comb(n, j1) * math.comb(n - j1, j2)
                        ypow = y0 ** (n - j1 - j2)
                        dy1_pow = dy1 ** j1
                        dy2_pow = dy2 ** j2
                        for k1 in range(p + 1):
                            for k2 in range(p - k1 + 1):
                                Bp_k1k2 = math.comb(p, k1) * math.comb(p - k1, k2)
                                zpow = z0 ** (p - k1 - k2)
                                dz1_pow = dz1 ** k1
                                dz2_pow = dz2 ** k2
                                I = i1 + j1 + k1
                                K = i2 + j2 + k2
                                sf = _simplex_integral_2d(I, K)
                                face_sum += (Bm1_i1i2 * Bn_j1j2 * Bp_k1k2 *
                                             xpow * ypow * zpow *
                                             dx1_pow * dx2_pow *
                                             dy1_pow * dy2_pow *
                                             dz1_pow * dz2_pow *
                                             sf)

        total += (A_2 * nx_f / mp1) * face_sum

    return total


def moment_of_polyhedron_fourier_3d(faces: np.ndarray,
                                     m: int, n: int, p: int) -> complex:
    """
    Fourier volume moment of a polyhedron (Eq 31).
    faces: np.ndarray shape (F, 3, 3).

    Special case m=n=p=0: return the volume (monomial moment with m=n=p=0).

    Otherwise:
    C_mnp = 1/(-j pi (m^2+n^2+p^2)) *
            sum_f 2*A_f*(m*nx+n*ny+p*nz)*exp(-j*beta_f)*I(alpha_f, mu_f)
    """
    if m == 0 and n == 0 and p == 0:
        return complex(moment_of_polyhedron_monomial_3d(faces, 0, 0, 0))

    F_arr = np.asarray(faces, dtype=float)
    pi = math.pi
    denom = -1j * pi * (m * m + n * n + p * p)
    total = 0.0 + 0.0j

    for f in range(F_arr.shape[0]):
        a = F_arr[f, 0]
        b = F_arr[f, 1]
        c = F_arr[f, 2]
        e1 = b - a
        e2 = c - a
        cross = np.cross(e1, e2)
        A_2 = np.linalg.norm(cross)  # = 2 * triangle area
        if A_2 == 0.0:
            continue
        n_hat = cross / A_2
        nx_f, ny_f, nz_f = n_hat[0], n_hat[1], n_hat[2]
        kdot_n = m * nx_f + n * ny_f + p * nz_f

        beta_f = pi * (m * a[0] + n * a[1] + p * a[2])
        alpha_f = pi * (m * e1[0] + n * e1[1] + p * e1[2])
        mu_f = pi * (m * e2[0] + n * e2[1] + p * e2[2])

        total += A_2 * kdot_n * np.exp(-1j * beta_f) * _fourier_simplex_I(alpha_f, mu_f)

    return total / denom


def moment_of_polyhedron_legendre_3d(faces: np.ndarray,
                                      m: int, n: int, p: int) -> float:
    """
    Legendre volume moment of a polyhedron (Eq 32).
    Expands P_m, P_n, P_p into monomials, then calls the monomial version.

      C_mnp^poly,L = sum_{u,v,w} cm[u]*cn[v]*cp[w] * C_uvw^poly,geo
    """
    Pm_poly = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
    Pn_poly = np.polynomial.legendre.Legendre.basis(n).convert(kind=Polynomial)
    Pp_poly = np.polynomial.legendre.Legendre.basis(p).convert(kind=Polynomial)

    cm = Pm_poly.coef
    cn = Pn_poly.coef
    cp = Pp_poly.coef

    total = 0.0
    for u, cmu in enumerate(cm):
        if cmu == 0.0:
            continue
        for v, cnv in enumerate(cn):
            if cnv == 0.0:
                continue
            for w, cpw in enumerate(cp):
                if cpw == 0.0:
                    continue
                total += cmu * cnv * cpw * moment_of_polyhedron_monomial_3d(faces, u, v, w)
    return total
