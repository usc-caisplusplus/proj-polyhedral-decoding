# Scripts

Run these from the **repository root** so paths like `./data/` and `config.json` resolve correctly.

| Script | Purpose |
|--------|---------|
| `setup_data.py` | Extract `Singapore.zip` / `NewYork.zip` from repo root into `./data/` |
| `ablation_input_dim.py` | Sweep `max_order` (Poly2Moment / moment encoder) across probe tasks |
| `ablation_fourier.py` | Sweep GFM `n_freqs` or `w_max` (Poly2Vec / Fourier branch) across tasks |
| `compare_encoders.py` | Small demos comparing Fourier vs moment encoders |
| `example_moment_encoder.py` | Examples for `GeometryMomentEncoder` and `Poly2Moment` |
| `evaluate_geo2vec_pairs.py` | Learn transductive Geo2Vec location/shape SDF embeddings and evaluate pair classification |

Place `Singapore.zip` and `NewYork.zip` at the repository root (ignored by git), then:

```bash
python scripts/setup_data.py
```

Examples:

```bash
python scripts/ablation_fourier.py --tasks regression --n_freqs_list 5 10 --epochs 25 --device cuda
python scripts/ablation_fourier.py --data_root ./data/NewYork --tasks regression --device cuda
python scripts/ablation_input_dim.py --tasks regression --max_orders 6 10 14 --dry_run
python scripts/compare_encoders.py
```

Smoke-test the Geo2Vec pair pipeline on a balanced subset:

```bash
python scripts/evaluate_geo2vec_pairs.py \
  --data-file data/NewYork/polygon_polygon_topological_relationship_data.pt \
  --type1 polygons --type2 polygons --task topology \
  --representation both --profile quick --max-pairs 1000
```

Run all pairs with the location-learning defaults from Geo2Vec:

```bash
python scripts/evaluate_geo2vec_pairs.py \
  --data-file data/NewYork/polygon_polygon_directional_relationship_360_data.pt \
  --type1 polygons --type2 polygons --task direction \
  --profile released --device cuda
```

Geo2Vec learns a free latent vector for each geometry rather than an inductive
geometry encoder. The location and shape SDF stages therefore see all
geometries without labels; only the downstream pair classifier is split
60/20/20. The default `--representation both` concatenates the two entity
vectors. Use `--representation location` for the location-only probes reported
in the Geo2Vec paper. Embeddings are cached under `results/geo2vec_cache/`.
Pair `.pt` files already use one shared city-wide `[-1, 1]` frame, so the
evaluator defaults to `--coordinate-mode pre_normalized`; do not fit new bounds
separately for each pair dataset when evaluating direction.

**GPU note:** If PyTorch prints *“no kernel image is available for execution on the device”* (e.g. very new NVIDIA architectures), install a PyTorch build that matches your GPU, or run with `--device cpu` until then.

Training entry point remains **`run.py`** at the repo root.
