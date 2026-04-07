# Shape Moments Representation

This repository studies **fixed-dimensional representations of 2D geospatial objects** (points, polylines, polygons) for learning-based spatial reasoning. Variable-length coordinates are mapped to a **fixed-size vector** using either:

- **Moment / basis encodings** (`encoder_type=poly2moment`): integrals against monomial, Legendre, or Fourier-type bases on the geometry.
- **Geometric Fourier mesh (GFM) encodings** (`encoder_type=poly2vec`): a spectral feature map used as a **strong baseline** and ablated separately.

Downstream tasks (distance, topology, direction, etc.) act as **probes** to test what the embedding captures. For the research framing and how this differs from a pure Poly2Vec reproduction, see **[docs/RESEARCH.md](docs/RESEARCH.md)**.

---

## Repository layout

| Path | Contents |
|------|----------|
| `models/` | Encoders (`poly2vec`, `poly2moment`, `fourier_encoder`, `moment_encoder`, …) |
| `utils/` | Training, moments, config, preprocessing helpers |
| `loaders/` | PyTorch datasets |
| `generator/` | Synthetic spatial datasets (JSONL / conversion) |
| `scripts/` | Ablation and demos — see **[scripts/README.md](scripts/README.md)** |
| `docs/` | [Research framing](docs/RESEARCH.md), [moment encoder notes](docs/MOMENT_ENCODER_README.md) |
| `config.json` | Default training / encoder hyperparameters (repo root) |

---

## Installation

Python ≥ 3.9 recommended:

```bash
conda create -n shape-moments python=3.9
conda activate shape-moments
pip install -r requirements.txt
```

---

## Quick start

**Training** (configure `encoder_type`, `moment_params` or `gfm_params`, paths, and task in `config.json`, or pass overrides like below):

```bash
python run.py \
  -dataset_name "NewYork" \
  -dataset_type1 "polygons" \
  -dataset_type2 "polygons" \
  -task "multi-relation" \
  -data_file "./data/NewYork/polygon_polygon_intersect_data.pt" \
  -encoder_type "poly2moment" \
  -data_path "./data/NewYork" \
  -num_classes 6
```

Use `-encoder_type "poly2vec"` for the GFM / Fourier branch.

**Ablations** (from repository root):

```bash
python scripts/ablation_input_dim.py --tasks regression --max_orders 6 10 14
python scripts/ablation_fourier.py --tasks regression --n_freqs_list 5 10 15 --epochs 25
```

---

## Datasets

Preprocessed assets used in experiments (including related work bundles) are linked here:

```
https://drive.google.com/drive/folders/119KvtA1K9CbII6TMMSPXkd1Yjrxhv9-P?usp=sharing
```

Place unpacked data under `./data/` (ignored by git). Example layout: `./data/Singapore/`, `./data/NewYork/`.

---

## Data preprocessing (OSM)

We follow OpenStreetMap-based pipelines for **Singapore** and **New York**; raw data can be obtained from [Geofabrik](https://download.geofabrik.de/) or via OSMnx.

**Tags (illustrative)**  
- Points (POIs): `amenity`, `shop`, `tourism`, `leisure`  
- Polylines (roads): `motorway`, `trunk`, `primary`  
- Polygons (buildings): `building`  

**Example OSMnx query**

```
pois_sg = ox.geometries_from_place("Singapore", tags={"amenity": True, "shop": True, "tourism": True, "leisure": True})
```

**Steps**

1. Put downloaded `.osm.pbf` (or your extract) under `./data/` as needed for your scripts.
2. Run normalization / extraction:

```bash
python utils/data_preprocessing.py
```

This produces pickles such as `poi_normalized.pkl`, `roads_normalized.pkl`, `buildings_normalized.pkl` under each dataset folder.

3. Optional pairwise / relationship `.pt` generation is handled in **`utils/dataset_generation.py`** (see that file for entry points):

```bash
python utils/dataset_generation.py
```

You should see files like `polygon_polygon_topological_relationship_data.pt` under the dataset directory when the script’s `__main__` block matches your needs.

---

## Related work: Poly2Vec (ICML 2025)

This codebase **implements and ablates** a **GFM / Fourier-style polymorphic encoder** in the spirit of **Poly2Vec**, which is a natural baseline when comparing against moment-based embeddings. The following refers to the **original Poly2Vec paper** and its suggested citation—not to the sole contribution of this repository.

> **Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications**  
> *Maria Despoina Siampou, Jialiang Li, John Krumm, Cyrus Shahabi, Hua Lu* — [arXiv:2408.14806](https://arxiv.org/pdf/2408.14806)

### Example: train with the Fourier / Poly2Vec encoder

```bash
python run.py \
  -dataset_name "NewYork" \
  -dataset_type1 "polygons" \
  -dataset_type2 "polygons" \
  -task "multi-relation" \
  -data_file "./data/NewYork/polygon_polygon_intersect_data.pt" \
  -encoder_type "poly2vec" \
  -data_path "./data/NewYork" \
  -num_classes 6
```

### Citation (Poly2Vec paper)

If you use ideas or implementations traceable to Poly2Vec / GFM, consider citing:

```bibtex
@inproceedings{siampoupoly2vec,
  title={Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications},
  author={Siampou, Maria Despoina and Li, Jialiang and Krumm, John and Shahabi, Cyrus and Lu, Hua},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
