# Unified spatial dataset generator

One pipeline for **directional**, **distance**, and **topological** tasks, with pluggable **targeted topology sampling**, **hard-case generation**, and **normalization**.

## Layout

- `make_spatial_dataset.py` — main CLI and generation loop
- `geometry_generators.py` — point, polyline, polygon, multipolygon (complexity: low/mid/high)
- `labels.py` — directional (16 compass), distance, topology labels
- `io_utils.py` — JSON geometry serialization, JSONL write, meta
- `normalization.py` — raw, bbox, m00, centroid
- `topology_sampler.py` — targeted sampling (touches, contains, intersects, disjoint, etc.)
- `hard_cases.py` — near_disjoint, almost_contained, thin_polygon, near_boundary, mixed

## Run from repo root

```bash
# Distance (point-point)
python run_make_spatial_dataset.py --task distance --pair_type point-point --n_samples 30000 --out_dir data/distance_point_point

# Directional (point-polygon)
python run_make_spatial_dataset.py --task directional --pair_type point-polygon --n_samples 30000 --out_dir data/dir_point_polygon

# Topological (polygon-polygon), random
python run_make_spatial_dataset.py --task topological --pair_type polygon-polygon --n_samples 30000 --out_dir data/topo_polygon_polygon

# Topological, targeted labels (balanced)
python run_make_spatial_dataset.py --task topological --pair_type polygon-polygon --topology_mode targeted --target_labels touches contains intersects disjoint --out_dir data/topo_polygon_polygon_balanced

# With hard cases and normalization
python run_make_spatial_dataset.py --task topological --pair_type polyline-polygon --hard_case_mode near_disjoint --normalization m00 --out_dir data/topo_pline_poly_hard
```

## Options

| Option | Values | Description |
|--------|--------|-------------|
| `--task` | distance, directional, topological | Task type |
| `--pair_type` | point-point, point-polygon, polygon-polygon, polyline-polygon, etc. | Geometry pair |
| `--topology_mode` | random, targeted | For topological: random or label-balanced |
| `--target_labels` | touches contains intersects disjoint (etc.) | Used when topology_mode=targeted |
| `--hard_case_mode` | none, near_disjoint, almost_contained, thin_polygon, near_boundary, mixed | Hard-case plugin |
| `--normalization` | raw, bbox, m00, centroid | Coordinate normalization before export |

## Output

- `train.jsonl`, `val.jsonl`, `test.jsonl` — one sample per line (70% / 15% / 15%)
- `config.json` — task, pair_type, n_samples, seed, options

Sample (topological):

```json
{
  "id": "polygon_polygon_00000001",
  "task": "topological",
  "pair_type": "polygon-polygon",
  "geom1": {"type": "polygon", "coords": [...], "holes": []},
  "geom2": {"type": "polygon", "coords": [...], "holes": []},
  "label": "touches",
  "meta": {"complexity": "high", "geom1_vertices": 18, "geom2_vertices": 24, "geom1_has_hole": true, "geom2_has_hole": false}
}
```

Distance sample includes `"label": 0.1247`. Directional includes `"label_id": 5, "label_16": "NNW"`.
