#!/usr/bin/env python3
"""
3D unified spatial dataset generator.

Mirrors `generator/make_spatial_dataset.py`, lifted to 3D.

Tasks:      distance, directional (48 classes = 16 az * 3 el), topological
Pair types: point-point, point-polyhedron, polyhedron-point,
            polyhedron-polyhedron, polyline-polyhedron, polyline-polyline
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

from . import geometry_generators_3d as gen
from . import labels_3d as labels
from . import io_utils_3d as io_utils
from . import normalization_3d as normalization
from . import topology_sampler_3d as topo
from . import hard_cases_3d as hard_cases


COMPLEXITIES = ["low", "mid", "high"]
ALLOWED_TYPES = ["point", "polyline", "polyhedron"]
HARD_CASE_MODES = ["none", "near_disjoint", "almost_contained", "thin_prism",
                    "near_boundary", "vertical_near_touching", "mixed"]


def build_sample(
    task: str,
    pair_type: str,
    g1: Any,
    g2: Any,
    complexity: str,
    idx: int,
    norm_mode: str,
) -> Dict[str, Any]:
    g1_n, g2_n = normalization.normalize_pair_3d(g1, g2, norm_mode)
    sample: Dict[str, Any] = {
        "id": f"{pair_type.replace('-', '_')}_{idx:08d}",
        "task": task,
        "pair_type": pair_type,
        "geom1": io_utils.to_json_geom_3d(g1_n),
        "geom2": io_utils.to_json_geom_3d(g2_n),
        "meta": {
            "complexity": complexity,
            "geom1_vertices": io_utils.geom_meta_3d(g1_n)["vertices"],
            "geom2_vertices": io_utils.geom_meta_3d(g2_n)["vertices"],
            "geom1_has_hole": io_utils.geom_meta_3d(g1_n)["has_hole"],
            "geom2_has_hole": io_utils.geom_meta_3d(g2_n)["has_hole"],
        },
    }

    if task == "distance":
        sample["label"] = round(float(labels.distance_label_3d(g1_n, g2_n)), 6)
    elif task == "directional":
        bid, lbl = labels.directional_label_3d(g1_n, g2_n)
        sample["label_id"] = bid
        sample["label_48"] = lbl
    elif task == "topological":
        sample["label"] = labels.topology_label_3d(pair_type, g1_n, g2_n)
    return sample


def generate_records(
    rng: random.Random,
    task: str,
    pair_type: str,
    n_samples: int,
    norm_mode: str,
    topology_mode: str = "random",
    target_labels: Optional[List[str]] = None,
    hard_case_mode: str = "none",
) -> List[Dict[str, Any]]:
    type1, type2 = pair_type.split("-")
    records: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n_samples * 8
    target_labels = target_labels or []

    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        complexity = COMPLEXITIES[len(records) % len(COMPLEXITIES)]

        requested_topo: Optional[str] = None
        if task == "topological" and topology_mode == "targeted" and target_labels:
            requested_topo = target_labels[len(records) % len(target_labels)]
            try:
                g1, g2 = topo.sample_targeted_topology_pair_3d(
                    rng, pair_type, requested_topo, complexity
                )
            except Exception:
                g1, g2 = gen.make_pair(rng, type1, type2, complexity)
        else:
            g1, g2 = gen.make_pair(rng, type1, type2, complexity)

        if hard_case_mode != "none":
            g1, g2 = hard_cases.apply_hard_case_3d(rng, g1, g2, hard_case_mode, pair_type)

        sample = build_sample(task, pair_type, g1, g2, complexity, len(records), norm_mode)
        if task == "topological" and sample.get("label") is None:
            continue
        # If we asked for a specific topology label but the realized label differs
        # (e.g. normalization collapsed the intended relation), reject.
        if requested_topo is not None and sample.get("label") != requested_topo:
            continue
        records.append(sample)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3D unified spatial dataset generator (distance, directional, topological)"
    )
    parser.add_argument("--task", required=True, choices=["distance", "directional", "topological"])
    parser.add_argument("--out_dir", required=True, help="Output directory for train/val/test JSONL and config")
    parser.add_argument(
        "--pair_type",
        required=True,
        help="e.g. point-polyhedron, polyhedron-polyhedron, polyline-polyhedron",
    )
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalization",
        choices=["raw", "bbox", "centroid"],
        default="raw",
    )
    parser.add_argument(
        "--topology_mode",
        choices=["random", "targeted"],
        default="random",
        help="For topological task: random or targeted (balanced) sampling",
    )
    parser.add_argument(
        "--target_labels",
        nargs="*",
        default=[],
        help="For topology_mode=targeted: e.g. touches contains inside disjoint",
    )
    parser.add_argument(
        "--hard_case_mode",
        choices=HARD_CASE_MODES,
        default="none",
    )
    args = parser.parse_args()

    parts = args.pair_type.split("-")
    if len(parts) != 2 or parts[0] not in ALLOWED_TYPES or parts[1] not in ALLOWED_TYPES:
        raise SystemExit(f"Invalid --pair_type: {args.pair_type} (allowed: {ALLOWED_TYPES})")

    if args.task == "topological" and args.topology_mode == "targeted" and not args.target_labels:
        args.target_labels = labels.get_topology_labels_for_pair_type(args.pair_type)
        if not args.target_labels:
            args.target_labels = ["disjoint", "intersects"]

    rng = random.Random(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    records = generate_records(
        rng=rng,
        task=args.task,
        pair_type=args.pair_type,
        n_samples=args.n_samples,
        norm_mode=args.normalization,
        topology_mode=args.topology_mode,
        target_labels=args.target_labels,
        hard_case_mode=args.hard_case_mode,
    )

    rng.shuffle(records)
    n = len(records)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train = records[:n_train]
    val = records[n_train : n_train + n_val]
    test = records[n_train + n_val :]

    io_utils.write_jsonl_3d(os.path.join(args.out_dir, "train.jsonl"), train)
    io_utils.write_jsonl_3d(os.path.join(args.out_dir, "val.jsonl"), val)
    io_utils.write_jsonl_3d(os.path.join(args.out_dir, "test.jsonl"), test)

    config = {
        "task": args.task,
        "pair_type": args.pair_type,
        "n_samples": n,
        "seed": args.seed,
        "complexities": COMPLEXITIES,
        "normalization": args.normalization,
        "topology_mode": args.topology_mode,
        "target_labels": args.target_labels,
        "hard_case_mode": args.hard_case_mode,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Done. task={args.task}, pair_type={args.pair_type}, n={n}, out={args.out_dir}")


if __name__ == "__main__":
    main()
