# Moments of 3D Shapes — Chunk 2: Worked Examples & Numerical Verification

Companion to [Moments_3D_Shapes.md](Moments_3D_Shapes.md). Every boxed
formula in Chunk 1 is here cross-checked against **hand-derived reference
values** by running the scalar implementation in
[utils/moments_compute_3d.py](../../utils/moments_compute_3d.py).

The verification script lives at
[scripts/verify_3d_moments.py](../../scripts/verify_3d_moments.py) and
runs all 18 checks below. Re-run it after any change to the moment code:

```
py scripts/verify_3d_moments.py
```

---

## 1. What is and isn't verified

**Verified in this chunk** (scalar path in `utils/moments_compute_3d.py`):

| Shape | Basis | Cases |
|---|---|---|
| Point | geo, Fourier | 4 |
| Segment | geo, Fourier | 4 |
| Triangle (3D surface) | geo | 5 |
| Polyhedron volume — cube | geo, Fourier (incl. DC branch) | 8 |
| Polyhedron volume — tetrahedron | geo | 1 |

**Vectorized batched path** is also cross-checked against the scalar
reference for both the cube and the tetrahedron across the full
$4\times 4\times 4$ moment grid — see §6 below.

**Explicitly not yet verified** (open for Chunk 3):

- **Legendre branches.** The implementation is `Legendre → monomial` expansion
  which is a linear combination of already-verified cases; it has to be
  correct *if* the monomial cases are, but it hasn't been numerically
  exercised head-to-head against an independent orthogonal-polynomial quadrature.
- **Fourier *triangle* (surface)** and **Fourier segment** at their removable-singularity
  branches (e.g. $\alpha=\beta$ for triangles). Only the generic branch
  is hit by the cube's axis-aligned faces.
- **Polyhedron *surface* moments** (Chunk 1 §8). Trivially a sum of
  verified triangle moments, but worth an explicit cube test (side area
  = 6 for a unit cube; $\iint x^2 dS = 6 \cdot \tfrac{1}{12} + \text{cross-face terms}$).

---

## 2. Convention reminder

Repeated from [Chunk 1](Moments_3D_Shapes.md) for self-containment:

- Domain: $[-1, 1]^3$
- Fourier scale: $\phi^{\mathrm F}_{mnp} = e^{-j\pi(mx+ny+pz)}$
- Polyhedron faces: **outward-consistent** CCW normals — the
  divergence-theorem formulas silently give $-C_{mnp}$ if the ordering is
  reversed, as demonstrated in §5.1 below.

---

## 3. Shared example inputs

All examples below are used by the verification script.

### 3.1 Point
- $\mathbf{x}_0 = (0, 0, 0)$
- $\mathbf{x}_1 = (0.5, 0.2, -0.3)$
- $\mathbf{x}_2 = (0.5, 0, 0)$ (for a non-trivial Fourier check)

### 3.2 Line segment
- $\mathbf{a} = (-0.5, 0, 0),\ \mathbf{b} = (0.5, 0, 0)$ — axis-aligned, length 1.

### 3.3 Right triangle in $z = 0$ plane
- Vertices $(0,0,0) \to (1,0,0) \to (0,1,0)$. Area $A = \tfrac{1}{2}$, so $2A = 1$.

### 3.4 Unit cube centered at origin
- Side 1, vertices at $\pm 0.5$ on each axis.
- Triangulated into 12 outward-CCW triangles (2 per face, 6 faces).

### 3.5 Regular tetrahedron from alternating cube corners
- Vertices: $(\tfrac12,\tfrac12,\tfrac12),\ (\tfrac12,-\tfrac12,-\tfrac12),\ (-\tfrac12,\tfrac12,-\tfrac12),\ (-\tfrac12,-\tfrac12,\tfrac12)$.
- Edge length $\sqrt{2}$, volume $= \tfrac{|\det(v_1-v_0,\ v_2-v_0,\ v_3-v_0)|}{6} = \tfrac{2}{6} = \tfrac{1}{3}$.

---

## 4. Reference values (derived by hand)

### 4.1 Point — §3 of Chunk 1
- $C^{\mathrm{pt,geo}}_{000}(\mathbf{0}) = 0^0\cdot 0^0\cdot 0^0 = 1$
- $C^{\mathrm{pt,geo}}_{200}(\mathbf{0}) = 0$
- $C^{\mathrm{pt,geo}}_{111}((0.5, 0.2, -0.3)) = 0.5\cdot 0.2\cdot(-0.3) = -0.03$
- $C^{\mathrm{pt,F}}_{100}((0.5, 0, 0)) = e^{-j\pi\cdot 0.5} = \cos(\pi/2) - j\sin(\pi/2) = -j$

### 4.2 Segment — §4 of Chunk 1
For $\mathbf{a}=(-0.5,0,0)\to\mathbf{b}=(0.5,0,0)$, $L = 1$:
- $C^{\mathrm{seg,geo}}_{000} = L = 1$
- $C^{\mathrm{seg,geo}}_{200} = \int_{-0.5}^{0.5} x^2\, dx = \tfrac{1}{12} \approx 0.0833$
- $C^{\mathrm{seg,geo}}_{300} = 0$ (odd power, symmetric interval)
- Fourier: $C^{\mathrm{seg,F}}_{100} = \int_{-0.5}^{0.5} e^{-j\pi x}\,dx = \frac{2\sin(\pi/2)}{\pi} = \tfrac{2}{\pi} \approx 0.6366$

### 4.3 Triangle surface — §5 of Chunk 1
Right triangle $T$, area $\tfrac{1}{2}$:
- $C^{\mathrm{tri,geo}}_{000} = \iint_T 1\,dS = \tfrac{1}{2}$
- $C^{\mathrm{tri,geo}}_{100} = \iint_T x\,dS = \int_0^1\!\int_0^{1-y} x\,dx\,dy = \int_0^1 \tfrac{(1-y)^2}{2}\,dy = \tfrac{1}{6}$
- $C^{\mathrm{tri,geo}}_{010} = \tfrac{1}{6}$ by symmetry.
- $C^{\mathrm{tri,geo}}_{001} = 0$ since $z \equiv 0$.
- $C^{\mathrm{tri,geo}}_{200} = \int_0^1\!\int_0^{1-y} x^2\,dx\,dy = \int_0^1 \tfrac{(1-y)^3}{3}\,dy = \tfrac{1}{12}$

### 4.4 Unit cube volume — §7 of Chunk 1
Separable in $x,y,z$: $C^{\mathrm{vol,geo}}_{mnp} = I(m)\,I(n)\,I(p)$ with
$I(k) = \int_{-0.5}^{0.5} x^k\,dx = \tfrac{1 - (-1)^{k+1}}{2^{k+1}(k+1)}$
(= $\tfrac{2}{2^{k+1}(k+1)}$ for even $k$, else 0).

- Volume: $C^{\mathrm{vol,geo}}_{000} = 1$
- $C^{\mathrm{vol,geo}}_{100} = 0$
- $C^{\mathrm{vol,geo}}_{200} = I(2) = \tfrac{1}{12}$
- $C^{\mathrm{vol,geo}}_{220} = \tfrac{1}{12}\cdot\tfrac{1}{12} = \tfrac{1}{144}$
- $C^{\mathrm{vol,geo}}_{400} = I(4) = \tfrac{1}{80}$

Fourier (cube centered at origin, side $s$):
$C^{\mathrm{vol,F}}_{mnp} = \prod_{k\in\{m,n,p\}} \begin{cases} s & k=0 \\ \tfrac{2\sin(\pi k s/2)}{\pi k} & k\ne 0 \end{cases}$

- **DC**: $C^{\mathrm{vol,F}}_{000} = 1$ (exercises the zero-index branch)
- $C^{\mathrm{vol,F}}_{100} = \tfrac{2\sin(\pi/2)}{\pi} = \tfrac{2}{\pi}$ (real — cube is symmetric about 0)
- $C^{\mathrm{vol,F}}_{200} = \tfrac{2\sin(\pi)}{2\pi} = 0$

### 4.5 Regular tetrahedron volume
$\tfrac{1}{3}$ (computed in §3.5).

---

## 5. Numerical verification (output from `verify_3d_moments.py`)

The raw output below is from running the script; `err` is absolute error
against the reference value. All rows must report `[OK ]`.

```text
================================================================================
§1. Point at origin
================================================================================
  [OK ] point geo (0,0,0)                          computed=+1.000000        ref=+1.000000        err=0.00e+00
  [OK ] point geo (2,0,0) at origin                computed=+0.000000        ref=+0.000000        err=0.00e+00
  [OK ] point geo (1,1,1) at (0.5,0.2,-0.3)        computed=-0.030000        ref=-0.030000        err=0.00e+00
  [OK ] point Fourier (1,0,0) at (0.5,0,0)         computed=+0.000000 -1.000000j   ref=+0.000000 -1.000000j   err=0.00e+00

================================================================================
§2. Axis-aligned unit segment [-0.5,0,0] -> [0.5,0,0]
================================================================================
  [OK ] segment geo (0,0,0)                        computed=+1.000000        ref=+1.000000        err=0.00e+00
  [OK ] segment geo (2,0,0)                        computed=+0.083333        ref=+0.083333        err=1.39e-17
  [OK ] segment geo (3,0,0) [numeric ref]          computed=+0.000000        ref=+0.000000        err=5.68e-18
  [OK ] segment Fourier (1,0,0)                    computed=+0.636620 +0.000000j  ref=+0.636620 +0.000000j   err=0.00e+00

================================================================================
§3. Right triangle (0,0,0)-(1,0,0)-(0,1,0) in z=0 plane
================================================================================
  [OK ] triangle geo (0,0,0)                       computed=+0.500000        ref=+0.500000        err=0.00e+00
  [OK ] triangle geo (1,0,0)                       computed=+0.166667        ref=+0.166667        err=0.00e+00
  [OK ] triangle geo (0,1,0)                       computed=+0.166667        ref=+0.166667        err=0.00e+00
  [OK ] triangle geo (0,0,1) [z=0 everywhere]      computed=+0.000000        ref=+0.000000        err=0.00e+00
  [OK ] triangle geo (2,0,0)                       computed=+0.083333        ref=+0.083333        err=0.00e+00

================================================================================
§4. Unit cube centered at origin (side = 1)
================================================================================
  [OK ] cube vol, geo (0,0,0) [volume]             computed=+1.000000        ref=+1.000000        err=0.00e+00
  [OK ] cube vol, geo (1,0,0)                      computed=+0.000000        ref=+0.000000        err=0.00e+00
  [OK ] cube vol, geo (2,0,0)                      computed=+0.083333        ref=+0.083333        err=0.00e+00
  [OK ] cube vol, geo (2,2,0)                      computed=+0.006944        ref=+0.006944        err=8.67e-19
  [OK ] cube vol, geo (4,0,0)                      computed=+0.012500        ref=+0.012500        err=0.00e+00
  [OK ] cube vol, Fourier (0,0,0) [DC branch]      computed=+1.000000 +0.000000j  ref=+1.000000 +0.000000j   err=0.00e+00
  [OK ] cube vol, Fourier (1,0,0)                  computed=+0.636620 -0.000000j  ref=+0.636620 +0.000000j   err=0.00e+00
  [OK ] cube vol, Fourier (2,0,0)                  computed=+0.000000 -0.000000j  ref=+0.000000 +0.000000j   err=0.00e+00

================================================================================
§5. Regular tetrahedron (vertices from alternating cube corners at scale 1)
================================================================================
  [OK ] tetra vol, geo (0,0,0) [volume]            computed=+0.333333        ref=+0.333333        err=0.00e+00

================================================================================
§6. Scalar vs. vectorized cross-check (cube + tetrahedron)
================================================================================
  [OK ] vectorized polyhedra geo — max error over 4^3 * 2 cases   err=4.07e-17
  [OK ] vectorized polyhedra Fourier — max error over 4^3 * 2 cases err=1.39e-17
  [OK ] vectorized points geo — max error over 4^3 * 2 cases      err=0.00e+00
  [OK ] vectorized points Fourier — max error over 4^3 * 2 cases   err=0.00e+00

================================================================================
Overall: ALL OK
================================================================================
```

The §6 cross-check builds a batched input of shape `(B=2, maxF=12, 3, 3)`
containing the cube and tetrahedron, runs both vectorized entry points
in `utils/moments_compute_3d_vectorized.py`, and compares every element
of the $4\times 4\times 4$ moment tensor against the scalar
implementation — 128 per-geometry comparisons each for geo and Fourier
polyhedra, plus 64 for points. The residual `~4e-17` is float64
round-off from differing summation orders, nothing more.

---

## 6. Notes on face orientation (a reminder, not a bug)

The tetrahedron case initially produced the *correct magnitude* but
**opposite sign** ($-\tfrac{1}{3}$ instead of $+\tfrac{1}{3}$). That was
expected — Chunk 1 §0 requires outward-consistent normals for the
divergence theorem. With CW (inward) ordering the signed volume flips
sign, and so do *all* volume moments.

The verification script now uses the CCW outward ordering:

- Face opposite $v_3$: $(v_0, v_1, v_2)$
- Face opposite $v_2$: $(v_0, v_3, v_1)$
- Face opposite $v_1$: $(v_0, v_2, v_3)$
- Face opposite $v_0$: $(v_1, v_3, v_2)$

**Downstream note**: when serializing polyhedra from `generator_3d/`,
the extruded prism is constructed with outward normals by construction
(bottom face CW in xy viewed from below, top face CCW from above, sides
oriented with right-hand rule outward). Any future custom mesh input
should pass through a normal-consistency check (see e.g. trimesh's
`fix_normals`).

---

## 7. Cube Fourier DC: one note on numerical zero

Row "cube vol, Fourier (2,0,0)" reports `+0.000000 -0.000000j` while the
analytic value is exactly $0$. The `-0.000000j` is **signed zero** from
floating-point; the absolute error is `0.00e+00`. If any downstream
consumer depends on strict `0+0j`, apply `np.where(np.abs(x) < eps, 0, x)`.

---

## 8. What to add to this chunk next

In rough priority order:

1. **Fourier triangle branch tests** that specifically hit each of the
   four simplex-kernel sub-cases from Chunk 1 §5.2 ($\alpha=\beta=0$,
   $\alpha=0$, $\beta=0$, $\alpha=\beta\ne 0$). A degenerate orientation of
   the right triangle relative to the index vector exercises each.
2. **Legendre sanity**: pick an orthonormal pair $(P_m, P_n)$ on a known
   shape (e.g. cube → moments of order $\ge 1$ along symmetry axes
   should vanish by parity), assert they vanish.
3. **Polyhedron surface-moment cube test** (Chunk 1 §8): $\iint_{\partial\text{cube}}1\,dS = 6$; $\iint x^2\,dS = \tfrac{1}{2}\cdot 2 + 4\cdot\tfrac{1}{12} = \tfrac{4}{3}$.
