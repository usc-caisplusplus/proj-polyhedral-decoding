# 3D unified spatial dataset generator

Mirrors [generator/](../generator/) lifted to 3D. One pipeline for
**directional**, **distance**, and **topological** tasks on point,
polyline, and polyhedron primitives.

## Layout

- `make_spatial_dataset_3d.py` — main CLI and generation loop
- `geometry_generators_3d.py` — `point`, `polyline` (3D), `polyhedron` (extruded prism)
- `labels_3d.py` — **48-class direction** (16 azimuth × 3 elevation), 3D distance, 3D topology
- `io_utils_3d.py` — JSON geometry serialization (see §Schema), JSONL write
- `normalization_3d.py` — raw, bbox, centroid (m00 omitted in MVP)
- `topology_sampler_3d.py` — targeted sampling for balanced topology labels
- `hard_cases_3d.py` — near_disjoint, almost_contained, thin_prism, near_boundary, **vertical_near_touching** (new 3D-specific class), mixed

## Math backing

The closed-form moment formulas implemented in
`utils/moments_compute_3d.py` are derived and numerically verified in:

- [docs/references/Moments_3D_Shapes.md](../docs/references/Moments_3D_Shapes.md) — Chunk 1: theory + 12 closed forms
- [docs/references/Moments_3D_Shapes_Chunk2.md](../docs/references/Moments_3D_Shapes_Chunk2.md) — Chunk 2: worked examples + 22 numeric checks (`scripts/verify_3d_moments.py`)

## Run from repo root

```bash
# Distance (polyhedron-polyhedron)
py -m generator_3d.make_spatial_dataset_3d --task distance --pair_type polyhedron-polyhedron --n_samples 30000 --out_dir data_3d/distance_pp

# Directional (point-polyhedron, 48 classes)
py -m generator_3d.make_spatial_dataset_3d --task directional --pair_type point-polyhedron --n_samples 30000 --out_dir data_3d/dir_pph

# Topological random (heavily imbalanced toward 'disjoint' — usually not what you want)
py -m generator_3d.make_spatial_dataset_3d --task topological --pair_type polyhedron-polyhedron --n_samples 30000 --out_dir data_3d/topo_pp

# Topological, targeted (balanced across 6 classes)
py -m generator_3d.make_spatial_dataset_3d --task topological --pair_type polyhedron-polyhedron --topology_mode targeted --n_samples 30000 --out_dir data_3d/topo_pp_balanced

# With hard cases (vertical_near_touching is unique to 3D)
py -m generator_3d.make_spatial_dataset_3d --task topological --pair_type polyhedron-polyhedron --hard_case_mode vertical_near_touching --normalization bbox --out_dir data_3d/topo_pp_hard
```

## Options

| Option | Values | Description |
|--------|--------|-------------|
| `--task` | distance, directional, topological | Task type |
| `--pair_type` | point-point, point-polyhedron, polyhedron-point, polyhedron-polyhedron, polyline-polyhedron, polyline-polyline | Geometry pair |
| `--topology_mode` | random, targeted | Random or label-balanced |
| `--target_labels` | inside disjoint contains within touches intersects equals | Used with `topology_mode=targeted` |
| `--hard_case_mode` | none, near_disjoint, almost_contained, thin_prism, near_boundary, **vertical_near_touching**, mixed | Hard-case plugin |
| `--normalization` | raw, bbox, centroid | Coordinate normalization before export |

## Output schema

- `train.jsonl`, `val.jsonl`, `test.jsonl` — 70% / 15% / 15% split
- `config.json` — task, pair_type, n_samples, seed, options

Sample (topological polyhedron-polyhedron):

```json
{
  "id": "polyhedron_polyhedron_00000001",
  "task": "topological",
  "pair_type": "polyhedron-polyhedron",
  "geom1": {"type": "polyhedron",
            "footprint": [[0.1,0.2],[0.4,0.1],[0.5,0.3]],
            "holes": [],
            "z_bot": -0.2, "z_top": 0.1},
  "geom2": {"type": "polyhedron", "footprint": [...], "holes": [], "z_bot": 0.1, "z_top": 0.4},
  "label": "touches",
  "meta": {"complexity": "mid", "geom1_vertices": 3, "geom2_vertices": 5,
           "geom1_has_hole": false, "geom2_has_hole": false}
}
```

Distance sample has `"label": 0.1247` (a float). Directional sample has
`"label_id": 21, "label_48": "above-NE"` (48 decoupled az/el classes).

## Polyhedron model (MVP)

Polyhedra are **extruded prisms** — a 2D footprint (Shapely polygon)
vertically extruded between `z_bot` and `z_top`. This is deliberate:

1. Matches OSM/CityGML LoD1 buildings.
2. Makes topology exact without trimesh: `2D footprint relation × 1D z-range relation → 3D relation`.
3. Can be triangulated into outward-CCW faces on demand and fed directly into `utils/moments_compute_3d.moment_of_polyhedron_*`.

Non-vertical meshes (LoD2+) are a later extension; the `Geom3D` type
union and `io_utils_3d.to_json_geom_3d` dispatch have room for them.

## Direction: (azimuth, elevation) decoupled

The label is a pair:
- **Azimuth bin** (16 classes, same compass as 2D): E, ENE, NE, ..., ESE
- **Elevation bin** (3 classes): below (phi < -15°), level (|phi| ≤ 15°), above (phi > 15°)

Combined label ID: `el_idx * 16 + az_idx ∈ {0..47}`. Named form e.g.
`"above-NE"`. Downstream probes can train on either the joint 48-class
label or the two marginals independently.

## Topology: 3D DE-9IM subset

Supported labels per pair type:

| Pair | Labels |
|---|---|
| point-point | disjoint, equals |
| point-polyhedron / polyhedron-point | disjoint, inside (/ contains) |
| polyline-polyhedron | disjoint, touches, within, intersects |
| polyline-polyline | disjoint, intersects |
| polyhedron-polyhedron | disjoint, touches, intersects, contains, within, equals |

The 3D relation is computed as `combine(footprint_2d, z_range_1d)`;
see `_combine_3d_topology` in `labels_3d.py` for the rule table.

## Known limits

- `m00` normalization isn't implemented for 3D (would require computing
  per-primitive volume/area; easy add once moment code is wired into the
  pipeline — polyhedron volume is already available via `moment_of_polyhedron_monomial_3d(faces, 0, 0, 0)`).
- Polyline-polyline 3D topology is approximated (2D-projection intersect
  + z-range overlap). True 3D line-line intersection requires more care;
  exact for parallel projections of single segments, over-counts for
  complex polylines. For MVP this is fine; flag if the probe task
  depends on exactness.
- No multi-polyhedra (analog of 2D MultiPolygon) yet.
