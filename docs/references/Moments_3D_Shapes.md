# Moments of 3D Shapes — Chunk 1: Theory + Closed Forms

Companion to [Moments_2D_Shapes.pdf](Moments_2D_Shapes.pdf). Same spirit:
every boxed equation is a *true closed form* — finite sums of
elementary functions, no residual integrals.

This document **(a)** lifts the 2D derivation to 3D and **(b)** plugs the
six gaps identified during the 2D → 3D discussion:

1. Explicit Fourier-simplex kernel with its four removable singularities.
2. DC / zero-index branch for the Fourier volume formula.
3. Surface-moment vs. volume-moment distinction for a solid.
4. First-class treatment of point clouds, polyline-3D, flat polygon-in-3D.
5. Conventions (domain, Fourier scale, Legendre normalization, central vs. raw).
6. Orientation / outward-normal requirement for divergence theorem.

---

## 0. Conventions (pick once, use everywhere)

| Knob | Value | Notes |
|---|---|---|
| Coordinate domain | $[-1,1]^3$ | inputs affinely normalized beforehand, same as 2D |
| Fourier scale | $\phi^{\text{F}}_{mnp} = e^{-j\pi(mx+ny+pz)}$ | matches the 2D PDF's $\pi$ (not $2\pi$) convention |
| Index set | integer $(m,n,p) \in \{0,\dots,M\}^3$ | direct analog of `max_order` in 2D |
| Legendre | raw $P_m(x)P_n(y)P_p(z)$ | orthonormal factor $\sqrt{(2m+1)/2}$ is optional post-processing |
| Default moments | **raw** (no centroid subtraction, no $m_{000}$ rescaling) | central / scale-normalized are options, not defaults |
| Polygon / face orientation | outward-consistent normals on closed surfaces | required for divergence theorem; triangulation preserves it |
| Triangle parameterization | $\mathbf{x}(u,v) = v_1 + u(v_2-v_1) + v(v_3-v_1)$, $(u,v)\in\triangle_1$ | unit simplex $\triangle_1 = \{u,v\ge 0,\ u+v\le 1\}$; $dS = 2A\,du\,dv$ |

All nine basis / shape combinations below respect these conventions.

---

## 1. The three bases in 3D

Moment coefficient:
$$C_{mnp} = \langle f, \phi_{mnp}\rangle = \iiint_{[-1,1]^3} f(x,y,z)\,\phi_{mnp}(x,y,z)\,dx\,dy\,dz.$$

| Basis | Definition | Notes |
|---|---|---|
| Geometric (monomial) | $\phi^{\text{geo}}_{mnp} = x^m y^n z^p$ | not orthogonal; easiest to compute |
| Fourier | $\phi^{\text{F}}_{mnp} = e^{-j\pi(mx+ny+pz)}$ | complex-valued; integer indexing |
| Legendre | $\phi^{\text{L}}_{mnp} = P_m(x)P_n(y)P_p(z)$ | tensor product, orthogonal on $[-1,1]^3$ |

Legendre polynomial expansion (same as 2D):
$$P_m(x) = \sum_{k=0}^{\lfloor m/2\rfloor} A_{m,k}\, x^{m-2k}, \qquad A_{m,k} = (-1)^k \frac{(2m-2k)!}{2^m\,k!\,(m-k)!\,(m-2k)!}.$$

---

## 2. Shape models

Same indicator-function view as 2D, lifted to $\mathbb{R}^3$:

### 2.1 Point (0-manifold)
$$f_{\text{pt}}(\mathbf{x}) = \delta^3(\mathbf{x}-\mathbf{x}_0).$$

### 2.2 Point cloud (countable union of points)
$$f_{\text{pc}}(\mathbf{x}) = \sum_{i=1}^{K} w_i\,\delta^3(\mathbf{x}-\mathbf{x}_i).$$
Default weights $w_i=1$; weighted variants allow intensity / confidence.

### 2.3 Line segment (1-manifold in 3D)
Parameterize $\mathbf{x}(t) = \mathbf{a} + t\,\mathbf{d}$, $\mathbf{d}=\mathbf{b}-\mathbf{a}$, $L=\|\mathbf{d}\|$:
$$f_{\text{seg}}(\mathbf{x}) = \int_0^1 \delta^3(\mathbf{x}-\mathbf{x}(t))\,L\,dt.$$
Same "arclength-weighted ridge" as 2D; $L$ turns $dt$ into $ds$.

### 2.4 Polyline-3D (piecewise segments)
$$f_{\text{pl}} = \sum_{e=1}^{E} f_{\text{seg},e}.$$
Trivial sum; stated explicitly so it's first-class.

### 2.5 Planar polygon in 3D (2-manifold, flat)
A polygon lying in a single plane embedded in 3D (one building wall, one roof face). Let $\mathbf{R}\in\mathrm{SO}(3)$ map the polygon's plane to the $xy$-plane, $\mathbf{t}\in\mathbb{R}^3$ the offset. Then
$$f_{\text{poly3}}(\mathbf{x}) = \delta(\hat{n}\cdot(\mathbf{x}-\mathbf{t}))\cdot \mathbf{1}_\Omega(\mathbf{R}^{-1}(\mathbf{x}-\mathbf{t}))$$
where $\Omega$ is the in-plane 2D polygon and $\hat{n}$ its normal. See §6 for the resulting 3D moment (reduces to a 2D moment plus phase).

### 2.6 Triangulated surface (2-manifold, mesh)
$$f_{\text{surf}}(\mathbf{x}) = \sum_{T\in\mathcal{T}} \delta_T(\mathbf{x}),$$
where $\delta_T$ is the surface delta of triangle $T$. Moment = sum of per-triangle integrals (§5).

### 2.7 Polyhedron / solid (3-manifold)
$$f_{\text{vol}}(\mathbf{x}) = \mathbf{1}_\Omega(\mathbf{x}),$$
indicator of the closed region $\Omega$ with piecewise-planar boundary $\partial\Omega$. Moment = volume integral (§7).

> **Two descriptors for one solid.** A polyhedron carries *both* a
> **volume moment** $\iiint_\Omega \phi\,dV$ (weighted by indicator) *and*
> a **surface moment** $\oiint_{\partial\Omega} \phi\,dS$ (weighted by
> area measure on the boundary). They encode complementary information
> (interior vs. skin). We derive both; downstream can pick one or
> concatenate both.

---

## 3. Point moments (3 cases)

Apply the 3D delta sifting property:
$$\boxed{C^{\text{pt}}_{mnp} = \phi_{mnp}(x_0, y_0, z_0).}$$

Explicitly:

$$\boxed{\;C^{\text{pt,geo}}_{mnp} = x_0^m y_0^n z_0^p,\quad
C^{\text{pt,F}}_{mnp} = e^{-j\pi(m x_0 + n y_0 + p z_0)},\quad
C^{\text{pt,L}}_{mnp} = P_m(x_0) P_n(y_0) P_p(z_0).\;}$$

### Point cloud (trivial extension)
$$C^{\text{pc}}_{mnp} = \sum_{i=1}^{K} w_i\,\phi_{mnp}(\mathbf{x}_i).$$

---

## 4. Segment moments in 3D (3 cases)

**Master formula** (direct lift of 2D Eq. 11):
$$\boxed{\;C^{\text{seg}}_{mnp} = L \int_0^1 \phi_{mnp}\!\big(\mathbf{a}+t\mathbf{d}\big)\,dt.\;}$$

### 4.1 Geometric — finite sum via triple binomial

Expand $(a_x + t d_x)^m$, $(a_y + t d_y)^n$, $(a_z + t d_z)^p$ each by the
binomial theorem, multiply, collect powers of $t$, integrate
$\int_0^1 t^{i+j+k}dt = 1/(i+j+k+1)$:

$$\boxed{\;C^{\text{seg,geo}}_{mnp} = L \sum_{i=0}^{m}\sum_{j=0}^{n}\sum_{k=0}^{p} \binom{m}{i}\binom{n}{j}\binom{p}{k}\,a_x^{m-i}a_y^{n-j}a_z^{p-k}\,d_x^{i}d_y^{j}d_z^{k}\,\frac{1}{i+j+k+1}.\;}$$

### 4.2 Fourier — closed sinc-like kernel

Let $\alpha = \pi(m d_x + n d_y + p d_z)$ and
$\beta_0 = \pi(m a_x + n a_y + p a_z)$. Then
$\phi^{\text{F}}(\mathbf{a}+t\mathbf{d}) = e^{-j\beta_0}\,e^{-j\alpha t}$, and
$\int_0^1 e^{-j\alpha t}dt = f(\alpha)$ where

$$f(\alpha) := \begin{cases} \dfrac{1 - e^{-j\alpha}}{j\alpha}, & \alpha\ne 0\\ 1, & \alpha = 0. \end{cases}$$

$$\boxed{\;C^{\text{seg,F}}_{mnp} = L\,e^{-j\beta_0}\,f(\alpha).\;}$$

### 4.3 Legendre — reduce to monomial

Using the monomial expansion of $P_m$:
$$C^{\text{seg,L}}_{mnp} = L \sum_{k,\ell,r} A_{m,k} A_{n,\ell} A_{p,r}\,C^{\text{seg,geo}}_{m-2k,\,n-2\ell,\,p-2r} / L,$$

i.e. a weighted sum of geometric segment moments. (Divide by $L$ to avoid double-counting, since each geometric moment already carries $L$.)

### Polyline-3D
$$C^{\text{pl}}_{mnp} = \sum_{e=1}^{E} C^{\text{seg}}_{mnp}(e).$$

---

## 5. Triangle-in-3D surface moments (3 cases)

Triangle with vertices $v_1, v_2, v_3 \in \mathbb{R}^3$. Parameterize
$$\mathbf{x}(u,v) = v_1 + u\,\mathbf{e}_1 + v\,\mathbf{e}_2,\quad \mathbf{e}_1 = v_2-v_1,\ \mathbf{e}_2 = v_3-v_1,\ (u,v)\in\triangle_1.$$
Area element $dS = 2A\,du\,dv$, where $A = \tfrac{1}{2}\|\mathbf{e}_1\times\mathbf{e}_2\|$.

**Master formula**:
$$\boxed{\;C^{\text{tri}}_{mnp} = 2A \iint_{\triangle_1} \phi_{mnp}\!\big(\mathbf{x}(u,v)\big)\,du\,dv.\;}$$

### 5.1 Geometric — Dirichlet moment formula

Multinomial expansion of $x(u,v)^m y(u,v)^n z(u,v)^p$ yields terms of
the form $C_{a,b}\,u^a v^b$. The unit-simplex moment is the classical
Dirichlet integral:
$$\iint_{\triangle_1} u^a v^b\,du\,dv = \frac{a!\,b!}{(a+b+2)!}.$$

Written out: let $\mathbf{x}(u,v) = v_1 + u\mathbf{e}_1 + v\mathbf{e}_2$, so each of the three Cartesian coordinates is an affine function of $(u,v)$. Define $x = x_1 + u\,e_{1x} + v\,e_{2x}$ (similarly $y, z$). Then

$$x^m = \sum_{i_1+i_2+i_3=m}\binom{m}{i_1,i_2,i_3}\,x_1^{i_1}(u\,e_{1x})^{i_2}(v\,e_{2x})^{i_3},$$

and similarly for $y^n$, $z^p$. Multiplying the three gives terms
$u^a v^b$ with
$a = i_2 + j_2 + k_2$, $b = i_3 + j_3 + k_3$, weighted by a product of
multinomial coefficients and powers of $v_1$ components and edge-vector
components.

$$\boxed{\;C^{\text{tri,geo}}_{mnp} = 2A \sum_{\substack{i_1+i_2+i_3=m\\ j_1+j_2+j_3=n\\ k_1+k_2+k_3=p}} M_{ijk}\,\cdot\,\frac{(i_2+j_2+k_2)!\,(i_3+j_3+k_3)!}{(i_2+j_2+k_2+i_3+j_3+k_3+2)!},\;}$$

where $M_{ijk}$ is the product of multinomial coefficients and coordinate-edge powers. Implementation detail: precompute once per (m,n,p); vectorize the multinomial enumeration.

### 5.2 Fourier — **the simplex kernel** (this is the gap I skipped before)

Expand the phase:
$$-j\pi\,\mathbf{k}\cdot\mathbf{x}(u,v) = -j\pi\,\mathbf{k}\cdot v_1 \;-\; j\pi(\mathbf{k}\cdot\mathbf{e}_1)\,u \;-\; j\pi(\mathbf{k}\cdot\mathbf{e}_2)\,v,$$
with $\mathbf{k} = (m,n,p)$.

Define
$$\beta_0 = \pi\,\mathbf{k}\cdot v_1,\quad \alpha = \pi\,\mathbf{k}\cdot\mathbf{e}_1,\quad \beta = \pi\,\mathbf{k}\cdot\mathbf{e}_2.$$

Then
$$C^{\text{tri,F}}_{mnp} = 2A\,e^{-j\beta_0}\,I(\alpha,\beta),\qquad I(\alpha,\beta) := \iint_{\triangle_1} e^{-j(\alpha u + \beta v)}\,du\,dv.$$

**Evaluate $I(\alpha,\beta)$.** Integrate $v$ first from $0$ to $1-u$:
$$I = \int_0^1 e^{-j\alpha u}\,\frac{1-e^{-j\beta(1-u)}}{j\beta}\,du$$
$$= \frac{1}{j\beta}\Big[f_1(\alpha) - e^{-j\beta} f_1(\alpha-\beta)\Big],\qquad f_1(\xi) := \int_0^1 e^{-j\xi u}du = \frac{1-e^{-j\xi}}{j\xi}\ \ (\xi\ne 0).$$

So the **generic case** is
$$\boxed{\;I(\alpha,\beta) = \frac{f_1(\alpha) - e^{-j\beta}\,f_1(\alpha-\beta)}{j\beta}\quad (\alpha\ne 0,\ \beta\ne 0,\ \alpha\ne\beta).\;}$$

**Removable singularities** (all finite; compute as separate branches in code):

$$\boxed{\;
\begin{aligned}
I(0,\beta)\ (\beta\ne 0)\  &= \frac{1}{j\beta} - \frac{1-e^{-j\beta}}{(j\beta)^2},\\[4pt]
I(\alpha,0)\ (\alpha\ne 0)\ &= \frac{1}{j\alpha} - \frac{1-e^{-j\alpha}}{(j\alpha)^2},\\[4pt]
I(\alpha,\alpha)\ (\alpha\ne 0)\ &= \frac{f_1(\alpha) - e^{-j\alpha}}{j\alpha},\\[4pt]
I(0,0)\ &= \tfrac{1}{2}.
\end{aligned}\;}$$

Four branches total — that's the explicit Fourier triangle kernel we
needed. Sanity: $I(0,0) = 1/2$ = area of the unit simplex, agreeing with
$\phi \equiv 1$.

### 5.3 Legendre — reduce to monomial
Same pattern as §4.3: expand $P_m(x)P_n(y)P_p(z)$ into monomials using the $A_{m,k}$ coefficients, reuse §5.1.

### Triangulated surface
$$C^{\text{surf}}_{mnp} = \sum_{T\in\mathcal{T}} C^{\text{tri}}_{mnp}(T).$$

---

## 6. Planar polygon in 3D (flat face — shortcut)

A single planar polygon in 3D doesn't need to be triangulated for a
2D-in-plane closed form — but triangulation is the simplest path and
reuses §5 directly. We record the shortcut for completeness:

Let the polygon lie in the plane with normal $\hat{n}$, orthonormal basis $(\hat{u},\hat{v})$ on the plane, and offset $\mathbf{t}$ (a point on the plane). Any $\mathbf{x}$ on the plane writes as $\mathbf{x} = \mathbf{t} + s\,\hat{u} + r\,\hat{v}$ with in-plane coords $(s,r)$. Pulled back, the in-plane shape is a 2D polygon $\Omega_{sr}$.

- **Geometric**: the 3D monomial $x^m y^n z^p$ becomes a polynomial in
  $(s,r)$ of total degree $m+n+p$; apply the 2D polygon geometric-moment
  formula from the 2D PDF (§7.4) on each term, then sum.
- **Fourier**: $e^{-j\pi \mathbf{k}\cdot\mathbf{x}}
  = e^{-j\pi\mathbf{k}\cdot\mathbf{t}} \cdot e^{-j\pi(\mathbf{k}\cdot\hat{u})s}\,e^{-j\pi(\mathbf{k}\cdot\hat{v})r}$;
  apply the 2D polygon Fourier formula (§7.5 of the 2D PDF) with
  in-plane frequencies $(\mathbf{k}\cdot\hat{u}, \mathbf{k}\cdot\hat{v})$ and a global phase factor.
- **Legendre**: monomial reduction.

**Practical recommendation**: triangulate the polygon (fan or ear-clip,
since it's planar) and call §5. Fewer code paths.

---

## 7. Polyhedron volume moments (3 cases)

### 7.1 3D divergence theorem

$$\iiint_\Omega \phi_{mnp}\,dV = \oiint_{\partial\Omega} \mathbf{F}\cdot\hat{n}\,dS,\qquad \nabla\cdot\mathbf{F} = \phi_{mnp}.$$

Find an $\mathbf{F}$, convert to a sum over triangular faces with
outward normals, then reduce each face integral to the triangle cases of §5
(with a constant $\mathbf{F}\cdot\hat{n}$ pulled out per face where possible).

### 7.2 Geometric — $\mathbf{F}$ puts everything on $x$

Take $\mathbf{F} = \left(\dfrac{x^{m+1}y^n z^p}{m+1},\ 0,\ 0\right)$.
Then $\nabla\cdot\mathbf{F} = x^m y^n z^p$. On a face with outward
normal $\hat{n}=(n_x, n_y, n_z)$ ($n_x$ **constant per flat face**):

$$C^{\text{vol,geo}}_{mnp} = \sum_{T\in\partial\Omega} \frac{n_{x,T}}{m+1} \iint_T x^{m+1} y^n z^p\,dS.$$

Each inner surface integral is a §5.1 triangle-geometric moment with
shifted $m$-index. **Finite sum over faces × multinomial** — fully closed
form.

### 7.3 Fourier — ansatz, DC branch

Take $\mathbf{F} = \gamma\,(m, n, p)\,\phi^{\text{F}}_{mnp}$. Compute
divergence:
$$\nabla\cdot\mathbf{F} = \gamma\,(-j\pi)(m^2+n^2+p^2)\,\phi^{\text{F}}_{mnp}.$$
Match $\phi^{\text{F}}_{mnp}$:
$$\gamma = \frac{1}{-j\pi(m^2+n^2+p^2)}\qquad (m,n,p) \ne (0,0,0).$$

$$\boxed{\;C^{\text{vol,F}}_{mnp} = \frac{1}{-j\pi(m^2+n^2+p^2)}\,\sum_{T\in\partial\Omega} (m\,n_{x,T} + n\,n_{y,T} + p\,n_{z,T})\iint_T e^{-j\pi(mx+ny+pz)}\,dS,\;}$$
with the inner integral from §5.2.

**DC branch** (singular): for $(m,n,p) = (0,0,0)$, $\phi^{\text{F}}_{000}\equiv 1$, so
$$\boxed{\;C^{\text{vol,F}}_{000} = \iiint_\Omega 1\,dV = \operatorname{vol}(\Omega).\;}$$
Compute directly as a signed volume sum over face tetrahedra (standard: for each triangle $T=(v_1,v_2,v_3)$, $\tfrac{1}{6}\,v_1\cdot(v_2\times v_3)$, summed with outward normals). **This is the same singularity as 2D at $(m,n)=(0,0)$ — worth fixing in the 2D PDF too.**

### 7.4 Legendre — reduce to geometric

Expand $P_m(x)P_n(y)P_p(z)$ into monomials, reuse §7.2.

---

## 8. Polyhedron **surface** moments (alternative descriptor)

Separate from §7 — just sum the triangle moments from §5 over
$\partial\Omega$:

$$\boxed{\;C^{\text{surf}(\Omega)}_{mnp} = \sum_{T\in\partial\Omega} C^{\text{tri}}_{mnp}(T).\;}$$

This is a "skin" descriptor: encodes shape of the boundary, ignores
interior filling. **For a thin vs. thick building of the same outline,
surface and volume descriptors diverge in useful ways.** Both are cheap
given the §5 machinery; recommended to concatenate.

---

## 9. Summary of closed forms (12 + 1 extra)

| Shape ↓ \ Basis → | Geometric | Fourier | Legendre |
|---|---|---|---|
| Point | §3 | §3 | §3 |
| Point cloud | §3 (sum) | §3 (sum) | §3 (sum) |
| Segment | §4.1 | §4.2 | §4.3 |
| Polyline-3D | §4 (sum) | §4 (sum) | §4 (sum) |
| Triangle in 3D | §5.1 | §5.2 (4 sub-cases) | §5.3 |
| Triangulated surface | §5 (sum) | §5 (sum) | §5 (sum) |
| Planar polygon in 3D | §6 / §5 (triangulate) | §6 / §5 | §6 / §5 |
| Polyhedron (volume) | §7.2 | §7.3 + DC | §7.4 |
| Polyhedron (surface) | §8 | §8 | §8 |

Every cell is a finite sum of elementary functions. No residual
integrals. Twelve + one alternate descriptor for solids.

---

## 10. Open design choices (not in MVP, worth noting)

1. **Central moments**: replace $\mathbf{x}$ by $\mathbf{x}-\bar{\mathbf{x}}$ where $\bar{\mathbf{x}}$ is the centroid. Gives translation invariance for free. Can be computed from raw moments via the standard $\mu_{mnp}$ recursion.
2. **Scale normalization**: divide $C_{mnp}$ by $m_{000}^{\alpha(m,n,p)}$ with $\alpha = (m+n+p+3)/3$ for 3D volumes, $(m+n+p+2)/2$ for surfaces. Gives scale invariance.
3. **Orthonormal Legendre**: multiply by $\sqrt{(2m+1)(2n+1)(2p+1)/8}$. Better conditioning at high orders.
4. **Rotation invariance**: tensor-product bases are *not* rotation invariant. For rotation-aware descriptors use:
   - **3D Zernike** (Canterakis 1999; Novotni & Klein 2003) — orthogonal on the unit ball, not tensor-product-separable.
   - **Spherical harmonics shells** (Kazhdan et al. 2003) — per-radius SH magnitudes.
   These do not fit the closed-form-polyhedron machinery directly;
   they'd need a separate path (voxelize / sample / ray-cast). Worth
   adding as a fourth basis track if rotation invariance matters.
5. **Point-cloud weighting**: intensity (LiDAR returns) or confidence can be folded in via $w_i$ in §2.2.
6. **Higher-order volume descriptors** — we currently have both volume and surface moments; a third option is *edge* moments (1-skeleton of the polyhedron), useful for wireframe shape. Reuses §4. Mention only; not MVP.

---

## 11. What belongs in Chunk 2 (not yet written)

- Concrete numerical examples: a unit cube, a tetrahedron, an extruded triangle — compute all 12 cases at low orders and cross-check.
- Vectorized implementation notes (batching over faces, over index tensors).
- Numerical-stability discussion for high $m+n+p$ (where monomial conditioning collapses and Legendre wins).
