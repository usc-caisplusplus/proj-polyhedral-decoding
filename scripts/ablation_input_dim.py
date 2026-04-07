"""
Ablation study over poly2moment input dimension (max_order) across multiple tasks.

For legendre/monomial basis: d_input = (max_order + 1)^2

Tasks swept:
  - regression:     point_point_dis_data.pt           (distance-prediction, points/points)
  - topo_multiclass: polygon_polygon_topological_relationship_data.pt (multi-relation, polygons/polygons, 6 cls)
  - directional:    polygon_polygon_directional_relationship_360_data.pt (direction-prediction, polygons/polygons, 8 cls)
  - cross_type:     polygon_polyline_topological_relationship_data.pt (multi-relation, polygons/polylines, 3 cls)

Usage (from repository root):

    python scripts/ablation_input_dim.py [--max_orders 2 4 6 8 10 12 14] [--tasks regression topo_multiclass directional cross_type] [--dry_run]

Results saved to ./results/ablation_input_dim/summary.json
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import argparse
import copy
import json
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from loaders.dataloader import GeometryRelationshipDataset
from utils.utils import CheckpointSaver, get_save_dir, load_model_checkpoint
from utils.config import args_parser
from models.TaskEncoder import TaskEncoder
from models.GeometryEncoder import GeometryEncoder
from utils.trainer import Trainer


# ── default sweep ─────────────────────────────────────────────────────────────
DEFAULT_MAX_ORDERS = [2, 4, 6, 8, 10, 12, 14]
ABLATION_DIR = "./results/ablation_input_dim"
VERBOSE = True

# Task configurations: name -> (data_file, task, type1, type2, num_classes)
TASK_CONFIGS = {
    "regression": (
        "/point_point_dis_data.pt",
        "distance-prediction",
        "points", "points",
        1,
    ),
    "topo_multiclass": (
        "/polygon_polygon_topological_relationship_data.pt",
        "multi-relation",
        "polygons", "polygons",
        6,
    ),
    "directional": (
        "/polygon_polygon_directional_relationship_360_data.pt",
        "direction-prediction",
        "polygons", "polygons",
        16,
    ),
    "cross_type": (
        "/polygon_polyline_topological_relationship_data.pt",
        "multi-relation",
        "polygons", "polylines",
        4,
    ),
}


def build_args_for_order(base_args, max_order: int, task_cfg):
    """Return a copy of base_args with max_order, d_input, and task fields updated."""
    args = copy.deepcopy(base_args)
    d_input = (max_order + 1) ** 2
    args.d_input = d_input
    args.moment_params = dict(args.moment_params)
    args.moment_params["max_order"] = max_order

    data_file, task, type1, type2, num_classes = task_cfg
    args.data_file = data_file
    args.task = task
    args.dataset_type1 = type1
    args.dataset_type2 = type2
    args.num_classes = num_classes
    return args


def run_one(args, device, data_tuple, dry_run=False):
    """Train and evaluate for one configuration. Returns test metrics dict."""
    X1, X2, L1, L2, Y = data_tuple

    (X1_tr, X1_oth, X2_tr, X2_oth,
     L1_tr, L1_oth, L2_tr, L2_oth,
     Y_tr,  Y_oth) = train_test_split(X1, X2, L1, L2, Y, test_size=0.4, random_state=42)

    (X1_val, X1_te, X2_val, X2_te,
     L1_val, L1_te, L2_val, L2_te,
     Y_val,  Y_te) = train_test_split(X1_oth, X2_oth, L1_oth, L2_oth, Y_oth,
                                       test_size=0.5, random_state=42)

    train_loader = DataLoader(GeometryRelationshipDataset(X1_tr,  X2_tr,  L1_tr,  L2_tr,  Y_tr),
                              batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(GeometryRelationshipDataset(X1_val, X2_val, L1_val, L2_val, Y_val),
                              batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(GeometryRelationshipDataset(X1_te,  X2_te,  L1_te,  L2_te,  Y_te),
                              batch_size=args.batch_size, shuffle=False)

    geometry_model = GeometryEncoder(args, device).to(device)
    model = TaskEncoder(args, geometry_model, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if args.task == "distance-prediction":
        criterion = torch.nn.MSELoss().to(device)
    elif args.task == "relation-prediction":
        criterion = torch.nn.BCELoss().to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)

    checkpoint_saver = CheckpointSaver(args.save_dir, metric_name="eval_loss", maximize_metric=False)
    trainer = Trainer(model, optimizer, criterion,
                      train_loader, val_loader, device,
                      args=args, checkpoint_saver=checkpoint_saver)

    if not dry_run:
        trainer.run(epochs=args.epochs, verbose=VERBOSE, patience=args.patience)

    best_path = os.path.join(args.save_dir, "best.pth.tar")
    if dry_run or not os.path.isfile(best_path):
        best_model = model
    else:
        best_model = load_model_checkpoint(best_path, model)
    test_metrics = trainer.test(test_loader, best_model)

    return {k: v.tolist() if hasattr(v, "tolist") else v for k, v in test_metrics.items()}


def main():
    parser = argparse.ArgumentParser(description="Ablation over poly2moment input dimension across tasks")
    parser.add_argument("--max_orders", type=int, nargs="+", default=DEFAULT_MAX_ORDERS)
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIGS.keys()),
                        choices=list(TASK_CONFIGS.keys()),
                        help="Which tasks to run")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip training; just test model instantiation and data loading")
    cli = parser.parse_args()

    base_args = args_parser(str(_REPO / "config.json"))
    device = torch.device("cuda" if torch.cuda.is_available() and base_args.device == "cuda" else "cpu")
    print(f"Device: {device}")

    os.makedirs(ABLATION_DIR, exist_ok=True)

    # summary[task_name][str(max_order)] = {d_input, status, metrics/error}
    summary = {t: {} for t in cli.tasks}

    for task_name in cli.tasks:
        task_cfg = TASK_CONFIGS[task_name]
        data_file, task, type1, type2, num_classes = task_cfg
        data_path = "./data/Singapore" + data_file

        print(f"\n{'#'*60}")
        print(f"  TASK: {task_name}  ({task}, {type1}/{type2}, {num_classes} cls)")
        print(f"  Data: {data_path}")
        print(f"{'#'*60}")

        data_tuple = torch.load(data_path)

        for max_order in cli.max_orders:
            d_input = (max_order + 1) ** 2
            print(f"\n  {'='*54}")
            print(f"  max_order={max_order}  ->  d_input={d_input}")
            print(f"  {'='*54}")

            args = build_args_for_order(base_args, max_order, task_cfg)

            run_name = (f"ablation_input_dim"
                        f"/{task_name}"
                        f"/max_order={max_order}"
                        f"_d_input={d_input}"
                        f"_basis={args.moment_params['basis_type']}"
                        f"_norm={args.moment_params['moment_norm']}")

            args.save_dir = get_save_dir(f"{ABLATION_DIR}/{run_name}", training=args.training)

            with open(os.path.join(args.save_dir, "args.json"), "w") as f:
                json.dump(vars(args), f, indent=4, sort_keys=True)

            try:
                metrics = run_one(args, device, data_tuple, dry_run=cli.dry_run)
                print(f"  Test metrics: {metrics}")
                summary[task_name][str(max_order)] = {"d_input": d_input, "status": "ok", "metrics": metrics}
            except Exception as e:
                print(f"  FAILED: {e}", file=sys.stderr)
                summary[task_name][str(max_order)] = {"d_input": d_input, "status": "error", "error": str(e)}

    # ── write summary ──────────────────────────────────────────────────────────
    summary_path = os.path.join(ABLATION_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nAblation complete. Summary saved to {summary_path}")

    # ── pretty-print table ─────────────────────────────────────────────────────
    for task_name, task_results in summary.items():
        print(f"\n-- {task_name} --")
        print(f"  {'max_order':>10}  {'d_input':>8}  {'status':>8}  metrics")
        print(f"  {'-'*64}")
        for order_str, entry in task_results.items():
            status = entry["status"]
            metrics_str = str(entry.get("metrics", entry.get("error", "")))
            print(f"  {order_str:>10}  {entry['d_input']:>8}  {status:>8}  {metrics_str}")


if __name__ == "__main__":
    main()
