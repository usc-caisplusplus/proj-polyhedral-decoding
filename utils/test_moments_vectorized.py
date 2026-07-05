"""
Tests for moments_compute_vectorized.py

Strategy: for every vectorized function, call the original scalar function from
moments_compute.py on the same inputs and assert the results match to within
floating-point tolerance.  We test:

  1. Points       — monomial, fourier, legendre
  2. Line segs    — monomial, fourier, legendre
  3. Polylines    — monomial, fourier, legendre  (via batch_moments_polylines)
  4. Polygons     — monomial, fourier, legendre

Run with:
    cd utils && python test_moments_vectorized.py
or with pytest:
    pytest utils/test_moments_vectorized.py -v
"""

import sys, os

# Allow both `python utils/test_moments_vectorized.py` (from project root)
# and `python test_moments_vectorized.py` (from inside utils/).
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _here)   # so bare `import moments_compute` works
sys.path.insert(0, _root)   # so `from utils.moments_compute` also works

import numpy as np
try:
    import pytest
except ImportError:
    # Stub so the file is importable without pytest
    class _FakeMark:
        def parametrize(self, *a, **kw):
            return lambda f: f
    class _FakePytest:
        mark = _FakeMark()
        @staticmethod
        def fixture(*a, **kw):
            return lambda f: f
    pytest = _FakePytest()

from moments_compute import (
    moment_of_point_monomial, moment_of_point_fourier, moment_of_point_legendre,
    moment_of_line_segment_monomial, moment_of_line_segment_fourier, moment_of_line_segment_legendre,
    moment_of_polygon_monomial, moment_of_polygon_fourier, moment_of_polygon_legendre,
)
from moments_compute_vectorized import (
    MomentTables,
    batch_moments_points_monomial, batch_moments_points_fourier, batch_moments_points_legendre,
    batch_moments_line_monomial, batch_moments_line_fourier, batch_moments_line_legendre,
    batch_moments_polylines,
    batch_moments_polygons_monomial, batch_moments_polygons_fourier, batch_moments_polygons_legendre,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MAX_ORDER = 4       # keep small for speed; tests cover correctness not scale
ATOL_REAL = 1e-9
ATOL_CPLX = 1e-9

@pytest.fixture(scope='module')
def tables():
    return MomentTables(MAX_ORDER)


def _scalar_point_grid(x, y, basis):
    """Build expected (O,O) grid for a single point using scalar functions."""
    O = MAX_ORDER + 1
    fn = {'monomial': moment_of_point_monomial,
          'fourier':  moment_of_point_fourier,
          'legendre': moment_of_point_legendre}[basis]
    out = np.zeros((O, O), dtype=np.complex128 if basis == 'fourier' else np.float64)
    for m in range(O):
        for n in range(O):
            out[m, n] = fn(x, y, m, n)
    return out


def _scalar_line_grid(ax, ay, bx, by, basis):
    """Build expected (O,O) grid for a single line segment."""
    O = MAX_ORDER + 1
    fn = {'monomial': moment_of_line_segment_monomial,
          'fourier':  moment_of_line_segment_fourier,
          'legendre': moment_of_line_segment_legendre}[basis]
    out = np.zeros((O, O), dtype=np.complex128 if basis == 'fourier' else np.float64)
    for m in range(O):
        for n in range(O):
            out[m, n] = fn(ax, ay, bx, by, m, n)
    return out


def _scalar_polygon_grid(verts, basis):
    """Build expected (O,O) grid for a single polygon."""
    O = MAX_ORDER + 1
    fn = {'monomial': moment_of_polygon_monomial,
          'fourier':  moment_of_polygon_fourier,
          'legendre': moment_of_polygon_legendre}[basis]
    out = np.zeros((O, O), dtype=np.complex128 if basis == 'fourier' else np.float64)
    for m in range(O):
        for n in range(O):
            out[m, n] = fn(verts, m, n)
    return out


# ---------------------------------------------------------------------------
# 1. Points
# ---------------------------------------------------------------------------

POINT_CASES = [
    (0.7,  -0.2),
    (0.0,   0.0),
    (-0.5,  0.9),
    (1.0,   1.0),
    (-1.0, -1.0),
]

@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_points_single(tables, basis):
    """Vectorized result for a single point matches scalar on every (m,n)."""
    x, y = 0.7, -0.2
    pts = np.array([[x, y]], dtype=np.float64)

    fn = {'monomial': batch_moments_points_monomial,
          'fourier':  batch_moments_points_fourier,
          'legendre': batch_moments_points_legendre}[basis]

    got = fn(pts, tables)[0]          # (O, O)
    exp = _scalar_point_grid(x, y, basis)

    np.testing.assert_allclose(got, exp, atol=ATOL_CPLX, rtol=0,
                                err_msg=f"points {basis} single mismatch")


@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_points_batch(tables, basis):
    """Batch of B points: each row matches the scalar."""
    pts = np.array(POINT_CASES, dtype=np.float64)   # (B, 2)
    fn  = {'monomial': batch_moments_points_monomial,
           'fourier':  batch_moments_points_fourier,
           'legendre': batch_moments_points_legendre}[basis]

    got = fn(pts, tables)   # (B, O, O)

    for b, (x, y) in enumerate(POINT_CASES):
        exp = _scalar_point_grid(x, y, basis)
        np.testing.assert_allclose(got[b], exp, atol=ATOL_CPLX, rtol=0,
                                    err_msg=f"points {basis} batch idx={b} mismatch")


# ---------------------------------------------------------------------------
# 2. Line segments
# ---------------------------------------------------------------------------

LINE_CASES = [
    (-0.6, -0.1,  0.2,  0.7),   # general
    ( 0.0,  0.0,  1.0,  0.0),   # horizontal
    ( 0.0,  0.0,  0.0,  1.0),   # vertical
    (-0.5, -0.5,  0.5,  0.5),   # diagonal
    ( 0.3,  0.1,  0.3,  0.9),   # vertical again, different x
]

@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_line_single(tables, basis):
    ax, ay, bx, by = -0.6, -0.1, 0.2, 0.7
    a  = np.array([ax], dtype=np.float64)
    ay_ = np.array([ay], dtype=np.float64)
    b  = np.array([bx], dtype=np.float64)
    by_ = np.array([by], dtype=np.float64)

    fn = {'monomial': batch_moments_line_monomial,
          'fourier':  batch_moments_line_fourier,
          'legendre': batch_moments_line_legendre}[basis]

    got = fn(a, ay_, b, by_, tables)[0]
    exp = _scalar_line_grid(ax, ay, bx, by, basis)

    np.testing.assert_allclose(got, exp, atol=ATOL_CPLX, rtol=0,
                                err_msg=f"line {basis} single mismatch")


@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_line_batch(tables, basis):
    cases = np.array(LINE_CASES, dtype=np.float64)   # (B, 4)
    ax_, ay_, bx_, by_ = cases[:,0], cases[:,1], cases[:,2], cases[:,3]

    fn = {'monomial': batch_moments_line_monomial,
          'fourier':  batch_moments_line_fourier,
          'legendre': batch_moments_line_legendre}[basis]

    got = fn(ax_, ay_, bx_, by_, tables)   # (B, O, O)

    for b, (ax, ay, bx, by) in enumerate(LINE_CASES):
        exp = _scalar_line_grid(ax, ay, bx, by, basis)
        np.testing.assert_allclose(got[b], exp, atol=ATOL_CPLX, rtol=0,
                                    err_msg=f"line {basis} batch idx={b} mismatch")


# ---------------------------------------------------------------------------
# 3. Polylines
# ---------------------------------------------------------------------------

def _make_polyline_batch(polyline_list):
    """Pad a list of variable-length polylines into (B, maxS, 2) + lengths."""
    B = len(polyline_list)
    lengths = np.array([len(p) for p in polyline_list], dtype=int)
    maxS = int(lengths.max())
    padded = np.zeros((B, maxS, 2), dtype=np.float64)
    for b, pts in enumerate(polyline_list):
        padded[b, :len(pts)] = pts
    return padded, lengths


def _scalar_polyline_grid(pts, basis):
    """Sum scalar line-segment moments over all segments of a polyline."""
    O = MAX_ORDER + 1
    fn = {'monomial': moment_of_line_segment_monomial,
          'fourier':  moment_of_line_segment_fourier,
          'legendre': moment_of_line_segment_legendre}[basis]
    out = np.zeros((O, O), dtype=np.complex128 if basis == 'fourier' else np.float64)
    for i in range(len(pts) - 1):
        ax, ay = pts[i]
        bx, by = pts[i+1]
        for m in range(O):
            for n in range(O):
                out[m, n] += fn(ax, ay, bx, by, m, n)
    return out


POLYLINE_CASES = [
    np.array([[-0.5, -0.3], [0.2, 0.1], [0.6, 0.5]]),          # 2 segments
    np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]), # L-shape, 3 segments
    np.array([[-0.8, 0.2], [0.0, -0.4], [0.7, 0.3]]),           # general
]

@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_polylines_batch(tables, basis):
    padded, lengths = _make_polyline_batch(POLYLINE_CASES)
    got = batch_moments_polylines(padded, lengths, tables, basis=basis)   # (B, O, O)

    for b, pts in enumerate(POLYLINE_CASES):
        exp = _scalar_polyline_grid(pts, basis)
        np.testing.assert_allclose(got[b], exp, atol=ATOL_CPLX, rtol=0,
                                    err_msg=f"polyline {basis} batch idx={b} mismatch")


@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_polylines_short(tables, basis):
    """Polylines with length < 2 should produce all-zero moments."""
    pts = np.array([[0.1, 0.2]])   # single point, 0 segments
    padded, lengths = _make_polyline_batch([pts])
    got = batch_moments_polylines(padded, lengths, tables, basis=basis)
    assert np.all(got == 0), f"polyline {basis} short should be zero"


# ---------------------------------------------------------------------------
# 4. Polygons
# ---------------------------------------------------------------------------

def _make_polygon_batch(polygon_list):
    """Pad a list of variable-vertex polygons into (B, maxK, 2) + lengths."""
    B = len(polygon_list)
    lengths = np.array([len(p) for p in polygon_list], dtype=int)
    maxK = int(lengths.max())
    padded = np.zeros((B, maxK, 2), dtype=np.float64)
    for b, verts in enumerate(polygon_list):
        padded[b, :len(verts)] = verts
    return padded, lengths


POLYGON_CASES = [
    # unit square (CCW)
    np.array([[-0.15, -0.15], [0.15, -0.15], [0.15, 0.15], [-0.15, 0.15]]),
    # general quad (CCW)
    np.array([[-0.30, -0.20], [0.35, -0.10], [0.25, 0.40], [-0.20, 0.30]]),
    # triangle (CCW)
    np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.8]]),
]

@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_polygons_batch(tables, basis):
    padded, lengths = _make_polygon_batch(POLYGON_CASES)
    fn = {'monomial': batch_moments_polygons_monomial,
          'fourier':  batch_moments_polygons_fourier,
          'legendre': batch_moments_polygons_legendre}[basis]

    got = fn(padded, lengths, tables)   # (B, O, O)

    for b, verts in enumerate(POLYGON_CASES):
        exp = _scalar_polygon_grid(verts, basis)
        np.testing.assert_allclose(got[b], exp, atol=1e-8, rtol=0,
                                    err_msg=f"polygon {basis} batch idx={b} mismatch")


@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_polygon_area_m0n0(tables, basis):
    """(m,n)=(0,0) moment should equal the signed polygon area."""
    padded, lengths = _make_polygon_batch(POLYGON_CASES)
    fn = {'monomial': batch_moments_polygons_monomial,
          'fourier':  batch_moments_polygons_fourier,
          'legendre': batch_moments_polygons_legendre}[basis]

    got = fn(padded, lengths, tables)   # (B, O, O)

    for b, verts in enumerate(POLYGON_CASES):
        x, y = verts[:, 0], verts[:, 1]
        area = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
        val = got[b, 0, 0].real if np.iscomplexobj(got) else got[b, 0, 0]
        np.testing.assert_allclose(abs(val), area, atol=1e-9, rtol=0,
                                    err_msg=f"polygon {basis} area idx={b} mismatch")


@pytest.mark.parametrize('basis', ['monomial', 'fourier', 'legendre'])
def test_polygon_with_hole(tables, basis):
    """NaN-separated rings close independently and holes are subtracted."""
    exterior = np.array([
        [-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0],
    ])
    hole = np.array([
        [-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5],
    ])
    vertices = np.concatenate(
        [exterior, np.array([[np.nan, np.nan]]), hole],
        axis=0,
    )
    polygons = vertices[None, :, :]
    lengths = np.array([len(vertices)])
    signs = np.concatenate([
        np.ones(len(exterior)), np.zeros(1), -np.ones(len(hole))
    ])[None, :]
    fn = {'monomial': batch_moments_polygons_monomial,
          'fourier':  batch_moments_polygons_fourier,
          'legendre': batch_moments_polygons_legendre}[basis]

    got = fn(polygons, lengths, tables, signs=signs)[0]
    expected = (
        _scalar_polygon_grid(exterior, basis)
        - _scalar_polygon_grid(hole, basis)
    )
    np.testing.assert_allclose(got, expected, atol=1e-8, rtol=0)


# ---------------------------------------------------------------------------
# 5. Numerical sanity checks (independent of scalar reference)
# ---------------------------------------------------------------------------

def test_point_monomial_known(tables):
    """x^m * y^n = 0.7^2 * (-0.2)^1 = -0.098."""
    pts = np.array([[0.7, -0.2]])
    got = batch_moments_points_monomial(pts, tables)
    np.testing.assert_allclose(got[0, 2, 1], -0.098, atol=1e-12)


def test_point_fourier_m0n0(tables):
    """exp(-i pi * 0) = 1 for all points."""
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (10, 2))
    got = batch_moments_points_fourier(pts, tables)
    np.testing.assert_allclose(got[:, 0, 0], np.ones(10, dtype=complex), atol=1e-12)


def test_line_monomial_m0n0_equals_length(tables):
    """Monomial (0,0) moment of a line segment = arc length."""
    cases = np.array(LINE_CASES, dtype=np.float64)
    ax, ay, bx, by = cases[:,0], cases[:,1], cases[:,2], cases[:,3]
    got = batch_moments_line_monomial(ax, ay, bx, by, tables)
    expected_lengths = np.hypot(bx - ax, by - ay)
    np.testing.assert_allclose(got[:, 0, 0], expected_lengths, atol=1e-12)


def test_line_fourier_m0n0_equals_length(tables):
    """Fourier (0,0) moment of a line segment = arc length (real)."""
    cases = np.array(LINE_CASES, dtype=np.float64)
    ax, ay, bx, by = cases[:,0], cases[:,1], cases[:,2], cases[:,3]
    got = batch_moments_line_fourier(ax, ay, bx, by, tables)
    expected_lengths = np.hypot(bx - ax, by - ay)
    np.testing.assert_allclose(got[:, 0, 0].real, expected_lengths, atol=1e-12)
    np.testing.assert_allclose(got[:, 0, 0].imag, np.zeros(len(LINE_CASES)), atol=1e-12)


def test_polygon_monomial_square_area(tables):
    """Unit square with half-side 0.15 has area 0.09."""
    half = 0.15
    square = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    padded, lengths = _make_polygon_batch([square])
    got = batch_moments_polygons_monomial(padded, lengths, tables)
    np.testing.assert_allclose(abs(got[0, 0, 0]), 0.09, atol=1e-12)


# ---------------------------------------------------------------------------
# Runner for direct execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import traceback

    tables_ = MomentTables(MAX_ORDER)
    tests = [
        # points
        lambda: test_points_single(tables_, 'monomial'),
        lambda: test_points_single(tables_, 'fourier'),
        lambda: test_points_single(tables_, 'legendre'),
        lambda: test_points_batch(tables_, 'monomial'),
        lambda: test_points_batch(tables_, 'fourier'),
        lambda: test_points_batch(tables_, 'legendre'),
        # lines
        lambda: test_line_single(tables_, 'monomial'),
        lambda: test_line_single(tables_, 'fourier'),
        lambda: test_line_single(tables_, 'legendre'),
        lambda: test_line_batch(tables_, 'monomial'),
        lambda: test_line_batch(tables_, 'fourier'),
        lambda: test_line_batch(tables_, 'legendre'),
        # polylines
        lambda: test_polylines_batch(tables_, 'monomial'),
        lambda: test_polylines_batch(tables_, 'fourier'),
        lambda: test_polylines_batch(tables_, 'legendre'),
        lambda: test_polylines_short(tables_, 'monomial'),
        lambda: test_polylines_short(tables_, 'fourier'),
        lambda: test_polylines_short(tables_, 'legendre'),
        # polygons
        lambda: test_polygons_batch(tables_, 'monomial'),
        lambda: test_polygons_batch(tables_, 'fourier'),
        lambda: test_polygons_batch(tables_, 'legendre'),
        lambda: test_polygon_area_m0n0(tables_, 'monomial'),
        lambda: test_polygon_area_m0n0(tables_, 'fourier'),
        lambda: test_polygon_area_m0n0(tables_, 'legendre'),
        lambda: test_polygon_with_hole(tables_, 'monomial'),
        lambda: test_polygon_with_hole(tables_, 'fourier'),
        lambda: test_polygon_with_hole(tables_, 'legendre'),
        # sanity
        lambda: test_point_monomial_known(tables_),
        lambda: test_point_fourier_m0n0(tables_),
        lambda: test_line_monomial_m0n0_equals_length(tables_),
        lambda: test_line_fourier_m0n0_equals_length(tables_),
        lambda: test_polygon_monomial_square_area(tables_),
    ]

    passed, failed = 0, 0
    for t in tests:
        name = t.__code__.co_consts[0] if t.__code__.co_consts else t.__name__
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
