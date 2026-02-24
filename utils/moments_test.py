import numpy as np

from moments_compute import moment_of_point_monomial, moment_of_point_fourier, moment_of_point_legendre
from moments_compute import moment_of_line_segment_monomial, moment_of_line_segment_fourier, moment_of_line_segment_legendre
from moments_compute import moment_of_polygon_monomial, moment_of_polygon_fourier, moment_of_polygon_legendre

####################################################################################################
# Moments of point
####################################################################################################
x0 = 0.7
y0 = -0.2
m = 0
n = 0

print(f"\nMoments for point: (x0,y0) = ({x0},{y0}), (m,n) = ({m},{n})")

point_monomial_should_be = 1.0
point_monomial_computed = moment_of_point_monomial(x0, y0, m, n)
print(f"\tExpected Monomial Moment: {point_monomial_should_be}, Computed: {point_monomial_computed}, Error: {point_monomial_computed - point_monomial_should_be}")

point_fourier_should_be = 1.0 + 0.0j
point_fourier_computed = moment_of_point_fourier(x0, y0, m, n)
print(f"\tExpected Fourier Moment: {point_fourier_should_be}, Computed: {point_fourier_computed}, Error: {point_fourier_computed - point_fourier_should_be}")

point_legendre_should_be = 1.0
point_legendre_computed = moment_of_point_legendre(x0, y0, m, n)
print(f"\tExpected Legendre Moment: {point_legendre_should_be}, Computed: {point_legendre_computed}, Error: {point_legendre_computed - point_legendre_should_be}")

m = 2
n = 1
print(f"\nMoments for point: (x0,y0) = ({x0},{y0}), (m,n) = ({m},{n})")

point_monomial_should_be = -0.098000000
point_monomial_computed = moment_of_point_monomial(x0, y0, m, n)
print(f"\tExpected Monomial Moment: {point_monomial_should_be}, Computed: {point_monomial_computed}, Error: {point_monomial_computed - point_monomial_should_be}")

point_fourier_should_be = -0.809016994 + 0.587785252j
point_fourier_computed = moment_of_point_fourier(x0, y0, m, n)
print(f"\tExpected Fourier Moment: {point_fourier_should_be}, Computed: {point_fourier_computed}, Error: {point_fourier_computed - point_fourier_should_be}")

point_legendre_should_be = -0.047000000
point_legendre_computed = moment_of_point_legendre(x0, y0, m, n)
print(f"\tExpected Legendre Moment: {point_legendre_should_be}, Computed: {point_legendre_computed}, Error: {point_legendre_computed - point_legendre_should_be}")

m = 2
n = 2
print(f"\nMoments for point: (x0,y0) = ({x0},{y0}), (m,n) = ({m},{n})")

point_monomial_should_be = 0.019600000
point_monomial_computed = moment_of_point_monomial(x0, y0, m, n)
print(f"\tExpected Monomial Moment: {point_monomial_should_be}, Computed: {point_monomial_computed}, Error: {point_monomial_computed - point_monomial_should_be}")

point_fourier_should_be = -1.0 + 0.0j
point_fourier_computed = moment_of_point_fourier(x0, y0, m, n)
print(f"\tExpected Fourier Moment: {point_fourier_should_be}, Computed: {point_fourier_computed}, Error: {point_fourier_computed - point_fourier_should_be}")

point_legendre_should_be = -0.103400000
point_legendre_computed = moment_of_point_legendre(x0, y0, m, n)
print(f"\tExpected Legendre Moment: {point_legendre_should_be}, Computed: {point_legendre_computed}, Error: {point_legendre_computed - point_legendre_should_be}")

####################################################################################################
# Moments of line segment
####################################################################################################
a = (-0.6, -0.1)
b = (0.2, 0.7)
ax = a[0]
ay = a[1]
bx = b[0]
by = b[1]
m = 0
n = 0

length = np.sqrt((bx - ax)**2 + (by - ay)**2)

print(f"\nMoments for line segment: a = {a}, b = {b}, (m,n) = ({m},{n})")

line_monomial_should_be = length
line_monomial_computed = moment_of_line_segment_monomial(ax,ay,bx,by, m, n)
print(f"\tExpected Monomial Moment: {line_monomial_should_be}, Computed: {line_monomial_computed}, Error: {line_monomial_computed - line_monomial_should_be}")

line_fourier_should_be = length + 0.0j,
line_fourier_computed = moment_of_line_segment_fourier(ax,ay,bx,by, m, n)
print(f"\tExpected Fourier Moment: {line_fourier_should_be}, Computed: {line_fourier_computed}, Error: {line_fourier_computed - line_fourier_should_be}")

line_legendre_should_be = length
line_legendre_computed = moment_of_line_segment_legendre(ax,ay,bx,by, m, n)
print(f"\tExpected Legendre Moment: {line_legendre_should_be}, Computed: {line_legendre_computed}, Error: {line_legendre_computed - line_legendre_should_be}")

m = 2
n = 1

print(f"\nMoments for line segment: a = {a}, b = {b}, (m,n) = ({m},{n})")

line_monomial_should_be = 0.007542472
line_monomial_computed = moment_of_line_segment_monomial(ax,ay,bx,by, m, n)
print(f"\tExpected Monomial Moment: {line_monomial_should_be}, Computed: {line_monomial_computed}, Error: {line_monomial_computed - line_monomial_should_be}")

line_fourier_should_be = -0.167764040 - 0.054509841j,
line_fourier_computed = moment_of_line_segment_fourier(ax,ay,bx,by, m, n)
print(f"\tExpected Fourier Moment: {line_fourier_should_be}, Computed: {line_fourier_computed}, Error: {line_fourier_computed - line_fourier_should_be}")

line_legendre_should_be = -0.158391919
line_legendre_computed = moment_of_line_segment_legendre(ax,ay,bx,by, m, n)
print(f"\tExpected Legendre Moment: {line_legendre_should_be}, Computed: {line_legendre_computed}, Error: {line_legendre_computed - line_legendre_should_be}")

m = 2
n = 2

print(f"\nMoments for line segment: a = {a}, b = {b}, (m,n) = ({m},{n})")

line_monomial_should_be = 0.003228178
line_monomial_computed = moment_of_line_segment_monomial(ax,ay,bx,by, m, n)
print(f"\tExpected Monomial Moment: {line_monomial_should_be}, Computed: {line_monomial_computed}, Error: {line_monomial_computed - line_monomial_should_be}")

line_fourier_should_be = -0.173180544 + 0.125823030j,
line_fourier_computed = moment_of_line_segment_fourier(ax,ay,bx,by, m, n)
print(f"\tExpected Fourier Moment: {line_fourier_should_be}, Computed: {line_fourier_computed}, Error: {line_fourier_computed - line_fourier_should_be}")

line_legendre_should_be = 0.089287787
line_legendre_computed = moment_of_line_segment_legendre(ax,ay,bx,by, m, n)
print(f"\tExpected Legendre Moment: {line_legendre_should_be}, Computed: {line_legendre_computed}, Error: {line_legendre_computed - line_legendre_should_be}")

######################################################################################################
# Moments of polygon
######################################################################################################
import numpy as np
import math

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def assert_close_real(name: str, got: float, exp: float, atol: float = 1e-9, rtol: float = 1e-9):
    if not math.isfinite(got) or not math.isfinite(exp):
        raise AssertionError(f"{name}: non-finite value got={got}, exp={exp}")
    if abs(got - exp) > (atol + rtol * abs(exp)):
        raise AssertionError(f"{name}: got={got:.12g}, exp={exp:.12g}, diff={got-exp:.12g}")

def assert_close_complex(name: str, got: complex, exp: complex, atol: float = 1e-9, rtol: float = 1e-9):
    if not (math.isfinite(got.real) and math.isfinite(got.imag)):
        raise AssertionError(f"{name}: non-finite value got={got}")
    if not (math.isfinite(exp.real) and math.isfinite(exp.imag)):
        raise AssertionError(f"{name}: non-finite value exp={exp}")
    diff = got - exp
    if abs(diff) > (atol + rtol * abs(exp)):
        raise AssertionError(f"{name}: got={got}, exp={exp}, |diff|={abs(diff):.12g}")

def to_np_verts(verts_list):
    """Return float64 Nx2 numpy array."""
    v = np.array(verts_list, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 2:
        raise ValueError(f"verts must be Nx2, got shape {v.shape}")
    return v

# ---------------------------------------------------------------------
# Test cases from the PDF
#   Square: half side a=0.15, CCW
#   Quad: CCW vertices
# Expected totals are from the PDF tables. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
# ---------------------------------------------------------------------

def run_polygon_tests():
    # Polygons (CCW)
    half = 0.15
    square = to_np_verts([
        (-half, -half),
        ( half, -half),
        ( half,  half),
        (-half,  half),
    ])

    quad = to_np_verts([
        (-0.30, -0.20),
        ( 0.35, -0.10),
        ( 0.25,  0.40),
        (-0.20,  0.30),
    ])

    # Expected totals from the PDF (Section 14.3 and 15.1). :contentReference[oaicite:4]{index=4}
    expected = {
        # (m,n) = (0,0): equals signed area for all three bases
        ("square", 0, 0): {
            "geo": 0.090000000,
            "fourier": complex(0.090000000, 0.0),
            "legendre": 0.090000000,
        },
        ("quad", 0, 0): {
            "geo": 0.275000000,
            "fourier": complex(0.275000000, 0.0),
            "legendre": 0.275000000,
        },

        ("square", 2, 1): {
            "geo": 0.000000000,
            "fourier": complex(0.074427711, 0.0),
            "legendre": 0.000000000,
        },
        ("square", 2, 2): {
            "geo": 0.0000050625,
            "fourier": complex(0.066315576, 0.0),
            "legendre": 0.021498890625,
        },
        ("quad", 2, 1): {
            "geo": 0.000488896,
            "fourier": complex(0.106454907, -0.061860239),
            "legendre": -0.011016656,
        },
        ("quad", 2, 2): {
            "geo": 0.000188216,
            "fourier": complex(0.046986391, -0.059875625),
            "legendre": 0.057707861,
        },
    }

    tests = [
        ("square", square),
        ("quad", quad),
    ]
    indices = [(0, 0), (2, 1), (2, 2)]

    # Tolerances: these formulas are closed form but floating error accumulates over edges.
    atol_real = 1e-9
    rtol_real = 1e-9
    atol_cplx = 1e-9
    rtol_cplx = 1e-9

    for poly_name, verts in tests:
        for (m, n) in indices:
            exp = expected[(poly_name, m, n)]

            # --- Call your routines ---
            C_geo = moment_of_polygon_monomial(verts, m, n)
            C_F   = moment_of_polygon_fourier(verts, m, n)
            C_L   = moment_of_polygon_legendre(verts, m, n)

            # --- Check ---
            assert_close_real(f"{poly_name} geo ({m},{n})", C_geo, exp["geo"], atol=atol_real, rtol=rtol_real)
            assert_close_complex(f"{poly_name} Fourier ({m},{n})", C_F, exp["fourier"], atol=atol_cplx, rtol=rtol_cplx)
            assert_close_real(f"{poly_name} Legendre ({m},{n})", C_L, exp["legendre"], atol=atol_real, rtol=rtol_real)

            print(f"PASS: {poly_name} (m,n)=({m},{n}) "
                  f"geo={C_geo:.12g}  F={C_F.real:.12g}{C_F.imag:+.12g}j  L={C_L:.12g}")

    print("All polygon tests passed.")

# ---------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------
run_polygon_tests()