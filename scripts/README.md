# Scripts

Run these from the **repository root** so paths like `./data/` and `config.json` resolve correctly.

| Script | Purpose |
|--------|---------|
| `ablation_input_dim.py` | Sweep `max_order` (Poly2Moment / moment encoder) across probe tasks |
| `ablation_fourier.py` | Sweep GFM `n_freqs` or `w_max` (Poly2Vec / Fourier branch) across tasks |
| `compare_encoders.py` | Small demos comparing Fourier vs moment encoders |
| `example_moment_encoder.py` | Examples for `GeometryMomentEncoder` and `Poly2Moment` |

Examples:

```bash
python scripts/ablation_fourier.py --tasks regression --n_freqs_list 5 10 --epochs 25
python scripts/ablation_input_dim.py --tasks regression --max_orders 6 10 14 --dry_run
python scripts/compare_encoders.py
```

Training entry point remains **`run.py`** at the repo root.
