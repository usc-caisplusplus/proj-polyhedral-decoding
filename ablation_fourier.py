"""
Ablation study for Poly2Vec (GFM / Fourier) hyperparameters across tasks.

Sweeps either:
  - ``n_freqs`` (geometric frequency count) with fixed ``w_min``, ``w_max`` from config, or
  - ``w_max`` (band upper end) with fixed ``n_freqs``, ``w_min`` from config.

``d_input`` is set from ``models.gfm_meshgrid.gfm_feature_dim`` so it always matches
the GFM mesh size.

Tasks match ``ablation_input_dim.py`` (same .pt files under ./data/Singapore).

Usage:
    python ablation_fourier.py --mode n_freqs --n_freqs_list 3 5 7 10 12 15
    python ablation_fourier.py --mode bandwidth --w_max_list 0.25 0.5 1.0 2.0
    python ablation_fourier.py --tasks regression --dry_run

Results: ./results/ablation_fourier/summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from ablation_input_dim import TASK_CONFIGS, run_one
from models.gfm_meshgrid import gfm_feature_dim
from utils.config import args_parser
from utils.utils import get_save_dir


ABLATION_DIR = "./results/ablation_fourier"
DEFAULT_N_FREQS = [3, 5, 7, 10, 12, 15]
DEFAULT_W_MAX = [0.25, 0.5, 1.0, 2.0]


def build_args_for_fourier(base_args, gfm_params: dict, task_cfg):
    args = copy.deepcopy(base_args)
    args.encoder_type = "poly2vec"
    args.gfm_params = dict(gfm_params)
    args.d_input = gfm_feature_dim(args.gfm_params)

    data_file, task, type1, type2, num_classes = task_cfg
    args.data_file = data_file
    args.task = task
    args.dataset_type1 = type1
    args.dataset_type2 = type2
    args.num_classes = num_classes
    return args


def main():
    parser = argparse.ArgumentParser(description="Ablation over GFM (Fourier) parameters")
    parser.add_argument(
        "--mode",
        choices=["n_freqs", "bandwidth"],
        default="n_freqs",
        help="n_freqs: sweep n_freqs; bandwidth: sweep w_max with fixed n_freqs from config",
    )
    parser.add_argument("--n_freqs_list", type=int, nargs="+", default=DEFAULT_N_FREQS)
    parser.add_argument("--w_max_list", type=float, nargs="+", default=DEFAULT_W_MAX)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()), choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--config", default="config.json", help="Base JSON (must include gfm_params)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs (default: config.json). Use a small value for quick smoke tests.",
    )
    cli = parser.parse_args()

    base_args = args_parser(cli.config)
    if cli.epochs is not None:
        base_args.epochs = cli.epochs
        base_args.patience = min(int(base_args.patience), max(1, cli.epochs))
    if getattr(base_args, "encoder_type", None) != "poly2vec":
        print(
            "Note: config encoder_type is not poly2vec; overriding to poly2vec for this ablation.",
            file=sys.stderr,
        )

    os.makedirs(ABLATION_DIR, exist_ok=True)

    base_gfm = dict(base_args.gfm_params)
    summary = {t: {} for t in cli.tasks}

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() and base_args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    for task_name in cli.tasks:
        task_cfg = TASK_CONFIGS[task_name]
        data_path = "./data/Singapore" + task_cfg[0]

        print(f"\n{'#' * 60}")
        print(f"  TASK: {task_name}")
        print(f"  Data: {data_path}")
        print(f"  mode: {cli.mode}")
        print(f"{'#' * 60}")

        data_tuple = torch.load(data_path)

        if cli.mode == "n_freqs":
            variants = []
            for nf in cli.n_freqs_list:
                gp = {**base_gfm, "n_freqs": nf}
                d_in = gfm_feature_dim(gp)
                variants.append((f"n_freqs={nf}", gp, d_in))
        else:
            variants = []
            nf = int(base_gfm["n_freqs"])
            for wm in cli.w_max_list:
                gp = {**base_gfm, "w_max": float(wm), "n_freqs": nf}
                d_in = gfm_feature_dim(gp)
                variants.append((f"w_max={wm}", gp, d_in))

        for label, gfm_params, d_input in variants:
            print(f"\n  {'=' * 54}")
            print(f"  {label}  ->  d_input={d_input}")
            print(f"  gfm_params={gfm_params}")
            print(f"  {'=' * 54}")

            args = build_args_for_fourier(base_args, gfm_params, task_cfg)

            run_name = (
                f"ablation_fourier/{cli.mode}/{task_name}/{label}_d_input={d_input}"
            )
            args.save_dir = get_save_dir(f"{ABLATION_DIR}/{run_name}", training=args.training)

            with open(os.path.join(args.save_dir, "args.json"), "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True, default=str)

            key = label.replace("=", "_")
            try:
                metrics = run_one(args, device, data_tuple, dry_run=cli.dry_run)
                print(f"  Test metrics: {metrics}")
                summary[task_name][key] = {
                    "d_input": d_input,
                    "gfm_params": gfm_params,
                    "status": "ok",
                    "metrics": metrics,
                }
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                summary[task_name][key] = {
                    "d_input": d_input,
                    "gfm_params": gfm_params,
                    "status": "error",
                    "error": str(e),
                }

    summary_path = os.path.join(ABLATION_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"\nAblation complete. Summary saved to {summary_path}")

    for task_name, task_results in summary.items():
        print(f"\n-- {task_name} --")
        for k, entry in task_results.items():
            st = entry["status"]
            m = entry.get("metrics", entry.get("error", ""))
            print(f"  {k:24}  {st:8}  {m}")


if __name__ == "__main__":
    main()
