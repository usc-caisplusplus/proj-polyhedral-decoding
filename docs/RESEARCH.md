# Research framing: shape representation learning

This repository supports a **research project** on **fixed-dimensional geometric representations** for machine learning. It is **not** primarily a Poly2Vec paper reproduction. The Poly2Vec-style Fourier encoder (`encoder_type=poly2vec`) is one **baseline / alternative** encoding path; the main scientific focus is **moment-based and basis-function encodings** (`encoder_type=poly2moment`) and how they compare.

---

## Core problem

Standard models (e.g. MLPs) need **fixed-size inputs**. Geospatial objects are **polymorphic** and **variable length**:

| Object        | Naive coordinate description |
|---------------|------------------------------|
| Point         | 2 numbers                    |
| Segment       | 4 numbers                    |
| Polyline      | \(2N\) numbers               |
| Polygon       | \(2N\) numbers (plus holes)  |

**Research question:** How can we map **arbitrary 2D geometry** to a **fixed-dimensional vector** such that downstream networks can recover **meaningful spatial structure** (topology, distance, direction, containment, etc.)?

---

## Proposed view: geometry as a function, moments as coordinates

1. **Treat the shape as a function** on the plane. For a filled region (polygon), a standard choice is an **indicator** \(f(x,y) \in \{0,1\}\) (or a smoothed variant). Points and polylines can be described with measures supported on vertices or edges, depending on formulation.

2. **Project onto a finite basis.** Choose basis functions \(\phi_{m,n}\) (e.g. monomials, complex Fourier exponentials, Legendre polynomials on a normalized domain). Compute **integral moments**
   \[
   c_{m,n} = \iint f(x,y)\,\phi_{m,n}(x,y)\,dx\,dy
   \]
   (or the analogous line/point definitions). Stacking \(\{c_{m,n}\}\) for a finite index set yields a **fixed-size embedding** before any neural network.

3. **Learn on top of the embedding.** An MLP (and task head) operates on this vector—not on raw variable-length coordinates. The **neural part** does not need to “parse” irregular lists; it learns from a **structured harmonic / polynomial coordinate system** of the shape.

This is **theory-structured representation learning**: the transform is grounded in analysis; the MLP tests whether the representation is **sufficient and learnable** for a task.

---

## What the downstream tasks are for

Tasks such as **distance regression**, **topological relation classification**, and **direction prediction** are **probe tasks**. They are not the definition of the project; they **stress-test** whether the embedding encodes:

- metric structure (distance),
- qualitative relations (overlap, inside, disjoint, …),
- angular / orientational cues (direction).

Strong performance across probes supports the claim that the representation **captures geometry**, not that the end goal is “a directional classifier” per se.

Synthetic data from `generator/` and preprocessed `.pt` bundles under `data/` exist to **train and evaluate** these probes under controlled conditions.

---

## Scientific axes explored in code

1. **Choice of basis** — `moment_params.basis_type`: monomial, Fourier, Legendre (see `utils/moments_compute*.py`, `models/moment_encoder.py`).
2. **Truncation / resolution** — `max_order` (and thus effective input size); see `scripts/ablation_input_dim.py`.
3. **Normalization of moments** — e.g. `moment_norm` in `poly2moment` (stability vs scale).
4. **Comparison to a Fourier mesh encoder** — Poly2Vec-style GFM path (`models/fourier_encoder.py`, `models/poly2vec.py`); sweeps in `scripts/ablation_fourier.py` vary `gfm_params` such as `n_freqs`, `w_min`, `w_max`.

These map directly to paper-worthy questions: **which basis**, **how many modes**, and **whether moments compete with or complement** a purely spectral mesh encoding.

---

## Relationship to Poly2Vec (the ICML paper)

The paper *Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications* introduces a **unified Fourier-based** pipeline for GeoAI. This codebase **implements related ideas** in the Fourier branch but uses this repo to ask a **broader question**: *moment / basis integral embeddings vs. Fourier-feature embeddings* for the same polymorphic geometry types.

Cite Poly2Vec where the Fourier encoder or methodology overlaps; distinguish this work wherever the **moment formulation**, **basis comparisons**, or **probe-task design** are novel.

---

## Suggested contributions (for a future paper)

- A clear **functional / moment** formulation of polymorphic 2D shape encoding with multiple orthogonal bases.
- **Systematic ablations** over basis type and order (and optional comparison to the GFM Fourier path).
- **Probe-task suite** demonstrating that the learned pipeline recovers spatial relations without hand-designed geometry-specific heads beyond fixed-dimensional embeddings.

---

## Quick pointers

| Topic | Location |
|--------|----------|
| Moment bases & batch computation | `utils/moments_compute_vectorized.py`, `utils/moments_compute.py` |
| Poly2Moment model | `models/poly2moment.py`, `models/moment_encoder.py` |
| Poly2Vec / GFM baseline | `models/poly2vec.py`, `models/fourier_encoder.py`, `models/gfm_meshgrid.py` |
| Moment input-dimension ablation | `scripts/ablation_input_dim.py` |
| GFM (Fourier) ablation | `scripts/ablation_fourier.py` |
| Synthetic probes & normalization | `generator/make_spatial_dataset.py`, `generator/normalization.py` |

---

## Ethics / scope

This is **representation learning** research, not a claim to replace classical computational geometry algorithms for exact predicates. Downstream use should respect data licenses (e.g. OSM-derived assets) and task-specific limitations of learned probes.
