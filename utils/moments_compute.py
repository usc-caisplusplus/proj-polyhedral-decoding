import numpy as np
import math
from numpy.polynomial.polynomial import Polynomial

####################################################################################################
# Moment bases

# Monomial basis: phi_mn(x,y) = x^m y^n
def basis_monomial(x: float, y: float, m: int, n: int) -> float:
    return (x ** m) * (y ** n)

# Fourier basis: phi_mn(x,y) = exp(-i pi (m x + n y))
def basis_fourier(x: float, y: float, m: int, n: int) -> complex:
    return np.exp(-1j * math.pi * (m * x + n * y))

# Legendre basis: phi_mn(x,y) = P_m(x) P_n(y)
def basis_legendre(x: float, y: float, m: int, n: int) -> float:
    # Speed up by precomputing polynomials in a permanent dictionary?
    Pm = np.polynomial.legendre.Legendre.basis(m)
    Pn = np.polynomial.legendre.Legendre.basis(n)
    return Pm(x) * Pn(y)

####################################################################################################
# Moments of point
# Point modeled as Dirac delta function

# Monomial moment of point
def moment_of_point_monomial(x: float, y: float, m: int, n: int) -> float:
    return basis_monomial(x, y, m, n)

# Fourier moment of point
def moment_of_point_fourier(x: float, y: float, m: int, n: int) -> complex:
    return basis_fourier(x, y, m, n)

# Legendre moment of point
def moment_of_point_legendre(x: float, y: float, m: int, n: int) -> float:
    return basis_legendre(x, y, m, n)

####################################################################################################
# Moments of line segment
# Line segment from point a to point b
# (ax, ay) and (bx, by) are the endpoints of the line segment
# Modeled as integral over parameterized Dirac delta functions along line

# Monomial moment of line segment
def moment_of_line_segment_monomial(ax:float, ay: float, bx:float, by:float, m: int, n: int) -> float:
    dx = bx - ax
    dy = by - ay
    L = math.hypot(dx, dy)

    # Speed up by precomputing comb() and powers?
    total = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            total += (math.comb(m, i) * math.comb(n, j) *
                      (ax ** (m - i)) * (ay ** (n - j)) *
                      (dx ** i) * (dy ** j) /
                      (i + j + 1))
    return L * total

# Fourier moment of line segment
def moment_of_line_segment_fourier(ax:float, ay: float, bx:float, by:float, m: int, n: int) -> complex:
    dx = bx - ax
    dy = by - ay
    L = math.hypot(dx, dy)

    kdot_a = m * ax + n * ay
    kdot_d = m * dx + n * dy
    phase0 = np.exp(-1j * math.pi * kdot_a)
    if abs(kdot_d) < 1e-14:
        return L * phase0
    num = 1.0 - np.exp(-1j * math.pi * kdot_d)
    den = 1j * math.pi * kdot_d
    return L * phase0 * (num / den)

# Legendre moment of line segment



def moment_of_line_segment_legendre(ax:float, ay: float, bx:float, by:float, m: int, n: int) -> float:
    """
    Closed-form Legendre moment C_{mn}^{seg,L} of a line segment from a to b.

    The segment is parameterized as:
        x(t) = ax + dx t
        y(t) = ay + dy t,   t in [0,1]
    and the moment is taken with respect to arc length:
        ds = L dt.

    Implements exactly:
        C_{mn}^{seg,L}
        = L * sum_{k,â„“} A_{m,k} A_{n,â„“}
              sum_{i,j} binom(m-2k,i) binom(n-2â„“,j)
                       a_x^{m-2k-i} a_y^{n-2â„“-j}
                       d_x^i d_y^j / (i+j+1)
    """

    dx = bx - ax
    dy = by - ay
    L = math.hypot(dx, dy)

    # Get Legendre polynomials in power (monomial) basis
    # P_m(x) = sum_u c_m[u] x^u
    Pm_poly: np.polynomial.polynomial = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
    Pn_poly: np.polynomial.polynomial = np.polynomial.legendre.Legendre.basis(n).convert(kind=Polynomial)

    cm = Pm_poly.coef  # coefficients for x^u
    cn = Pn_poly.coef  # coefficients for y^v

    total = 0.0

    # Loop over monomial powers in the Legendre expansions
    for u, cmu in enumerate(cm):
        if cmu == 0.0:
            continue

        for v, cnv in enumerate(cn):
            if cnv == 0.0:
                continue

            total += cmu * cnv * moment_of_line_segment_monomial(ax, ay, bx, by, u, v)

    return total

import numpy as np
import math
from typing import Union
from numpy.polynomial.polynomial import Polynomial

ArrayLike = Union[np.ndarray]

#####################################################################################
# Polygon moments (area moments) for three bases:
#   - Monomial:  âˆ¬_Î© x^m y^n dA
#   - Fourier:   âˆ¬_Î© exp(-i pi (m x + n y)) dA
#   - Legendre:  âˆ¬_Î© P_m(x) P_n(y) dA
#
# Polygon is given as a numpy array of vertices of shape (K, 2):
#   V[k] = (x_k, y_k), ordered CW or CCW, implicitly closed.

def _check_vertices(vertices: np.ndarray) -> np.ndarray:
    V = np.asarray(vertices, dtype=float)
    if V.ndim != 2 or V.shape[1] != 2:
        raise ValueError("vertices must be a numpy array of shape (K, 2).")
    if V.shape[0] < 3:
        raise ValueError("Polygon must have at least 3 vertices.")
    return V


def polygon_area(vertices: np.ndarray) -> float:
    """
    Signed area (positive for CCW ordering, negative for CW ordering).
    """
    V = _check_vertices(vertices)
    x = V[:, 0]
    y = V[:, 1]
    x2 = np.roll(x, -1)
    y2 = np.roll(y, -1)
    return 0.5 * float(np.sum(x * y2 - x2 * y))


#####################################################################################
# 1) Monomial polygon moment via Green's theorem (edge sum)

def moment_of_polygon_monomial(vertices: np.ndarray, m: int, n: int) -> float:
    """
    Closed-form area moment:
        M_{mn} = âˆ¬_Î© x^m y^n dA

    Uses Green's theorem with F = (x^{m+1} y^n / (m+1), 0),
    giving a sum over edges.

    Returns a signed quantity that flips sign if polygon orientation flips.
    """
    if m < 0 or n < 0:
        raise ValueError("m and n must be nonnegative integers.")
    V = _check_vertices(vertices)

    p = m + 1
    total = 0.0
    K = V.shape[0]

    for k in range(K):
        ax, ay = V[k]
        bx, by = V[(k + 1) % K]
        dx = bx - ax
        dy = by - ay

        inner = 0.0
        for i in range(p + 1):
            bin_p = math.comb(p, i)
            x_part = (ax ** (p - i)) * (dx ** i)

            for j in range(n + 1):
                bin_n = math.comb(n, j)
                y_part = (ay ** (n - j)) * (dy ** j)

                inner += (bin_p * bin_n * x_part * y_part) / (i + j + 1)

        total += (dy / (m + 1)) * inner

    return total


#####################################################################################
# 2) Fourier polygon moment via Green's theorem (edge sum)

def moment_of_polygon_fourier(vertices: np.ndarray, m: int, n: int, eps: float = 1e-14) -> complex:
    """
    Closed-form area Fourier moment:
        F_{mn} = âˆ¬_Î© exp(-i pi (m x + n y)) dA

    Uses Green's theorem with a vector field whose divergence equals the integrand.
    Two branches:
      - if m != 0: use F = (exp(...) / (-i pi m), 0)
      - else if n != 0: use F = (0, exp(...) / (-i pi n))
      - else: integrand is 1, so return polygon area

    Returns a signed quantity that flips sign if polygon orientation flips.
    """
    V = _check_vertices(vertices)
    K = V.shape[0]

    if m == 0 and n == 0:
        return complex(polygon_area(V), 0.0)

    total = 0.0 + 0.0j
    pi = math.pi

    # Choose divergence-theorem branch
    if m != 0:
        inv = 1.0 / (-1j * pi * m)  # 1 / (-i pi m)

        for k in range(K):
            ax, ay = V[k]
            bx, by = V[(k + 1) % K]
            dx = bx - ax
            dy = by - ay

            kdot_a = m * ax + n * ay
            kdot_d = m * dx + n * dy

            phase0 = np.exp(-1j * pi * kdot_a)

            # Integral along the edge: âˆ«_0^1 exp(-i pi (kÂ·a + kÂ·d t)) dt
            if abs(kdot_d) < eps:
                edge_int = phase0
            else:
                edge_int = phase0 * (1.0 - np.exp(-1j * pi * kdot_d)) / (1j * pi * kdot_d)

            # Boundary term uses n_x ds = dy dt, so contributes dy * edge_int
            total += inv * dy * edge_int

        return total

    else:
        # m == 0 and n != 0
        inv = 1.0 / (-1j * pi * n)  # 1 / (-i pi n)

        for k in range(K):
            ax, ay = V[k]
            bx, by = V[(k + 1) % K]
            dx = bx - ax
            dy = by - ay

            kdot_a = m * ax + n * ay  # here m=0
            kdot_d = m * dx + n * dy  # here m=0

            phase0 = np.exp(-1j * pi * kdot_a)

            if abs(kdot_d) < eps:
                edge_int = phase0
            else:
                edge_int = phase0 * (1.0 - np.exp(-1j * pi * kdot_d)) / (1j * pi * kdot_d)

            # Boundary term uses n_y ds = -dx dt, so contributes (-dx) * edge_int
            total += inv * (-dx) * edge_int

        return total


#####################################################################################
# 3) Legendre polygon moment via monomial expansion + monomial polygon moments

def moment_of_polygon_legendre(vertices: np.ndarray, m: int, n: int) -> float:
    """
    Closed-form area Legendre moment:
        C_{mn} = âˆ¬_Î© P_m(x) P_n(y) dA

    Expands P_m and P_n into monomials, then reuses monomial polygon moments.
    Returns a signed quantity that flips sign if polygon orientation flips.
    """
    if m < 0 or n < 0:
        raise ValueError("m and n must be nonnegative integers.")
    V = _check_vertices(vertices)

    Pm_poly = np.polynomial.legendre.Legendre.basis(m).convert(kind=Polynomial)
    Pn_poly = np.polynomial.legendre.Legendre.basis(n).convert(kind=Polynomial)
    cm = Pm_poly.coef  # Pm(x) = sum_u cm[u] x^u
    cn = Pn_poly.coef  # Pn(y) = sum_v cn[v] y^v

    total = 0.0
    for u, cmu in enumerate(cm):
        if cmu == 0.0:
            continue
        for v, cnv in enumerate(cn):
            if cnv == 0.0:
                continue
            total += cmu * cnv * moment_of_polygon_monomial(V, u, v)

    return total