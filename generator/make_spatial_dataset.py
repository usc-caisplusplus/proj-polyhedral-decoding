#!/usr/bin/env python3
"""
Unified spatial dataset generator.

Tasks: directional, distance, topological
Pair types: point-point, point-polygon, polygon-point, polyline-polyline, polyline-polygon, polygon-polygon
Plugins: topology_mode (random | targeted), hard_case_mode, normalization (raw | bbox | m00 | centroid)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Any

from . import geometry_generators as gen
from . import labels
from . import io_utils
from . import normalization
from . import topology_sampler as topo
from . import hard_cases


COMPLEXITIES = ["low", "mid", "high"]


def build_sample(
    task: str,
    pair_type: str,
    g1: Any,
    g2: Any,
    complexity: str,
    idx: int,
    norm_mode: str,
) -> Dict[str, Any]:
    """Build one sample dict (geom1/geom2 already normalized if norm_mode != raw)."""
    g1_n, g2_n = normalization.normalize_pair(g1, g2, norm_mode)

    sample: Dict[str, Any] = {
        "id": f"{pair_type.replace('-', '_')}_{idx:08d}",
        "task": task,
        "pair_type": pair_type,
        "geom1": io_utils.to_json_geom(g1_n),
        "geom2": io_utils.to_json_geom(g2_n),
        "meta": {
            "complexity": complexity,
            "geom1_vertices": io_utils.geom_meta(g1_n).get("vertices", 0),
            "geom2_vertices": io_utils.geom_meta(g2_n).get("vertices", 0),
            "geom1_has_hole": io_utils.geom_meta(g1_n).get("has_hole", False),
            "geom2_has_hole": io_utils.geom_meta(g2_n).get("has_hole", False),
        },
    }

    if task == "distance":
        sample["label"] = round(float(labels.distance_label(g1, g2)), 6)
    elif task == "directional":
        bid, lbl = labels.directional_label(g1, g2)
        sample["label_id"] = bid
        sample["label_16"] = lbl
    elif task == "topological":
        lab = labels.topology_label(pair_type, g1, g2)
        sample["label"] = lab
    return sample


def generate_records(
    rng: random.Random,
    task: str,
    pair_type: str,
    n_samples: int,
    norm_mode: str,
    topology_mode: str,
    target_labels: List[str],
    hard_case_mode: str,
) -> List[Dict[str, Any]]:
    type1, type2 = pair_type.split("-")
    records: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(records) < n_samples and attempts < max_attempts:
        attempts += 1
        complexity = COMPLEXITIES[len(records) % len(COMPLEXITIES)]

        if task == "topological" and topology_mode == "targeted" and target_labels:
            # Cycle through target labels for balance
            label = target_labels[len(records) % len(target_labels)]
            try:
                g1, g2 = topo.sample_targeted_topology_pair(rng, pair_type, label, complexity)
            except Exception:
                g1, g2 = gen.make_pair(rng, type1, type2, complexity)
        else:
            g1, g2 = gen.make_pair(rng, type1, type2, complexity)

        if hard_case_mode != "none":
            g1, g2 = hard_cases.apply_hard_case(rng, g1, g2, hard_case_mode, pair_type)

        sample = build_sample(task, pair_type, g1, g2, complexity, len(records), norm_mode)

        if task == "topological" and sample.get("label") is None:
            continue
        records.append(sample)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified spatial dataset generator (directional, distance, topological)"
    )
    parser.add_argument("--task", required=True, choices=["distance", "directional", "topological"])
    parser.add_argument("--out_dir", required=True, help="Output directory for train/val/test JSONL and config")
    parser.add_argument(
        "--pair_type",
        required=True,
        help="e.g. point-point, point-polygon, polygon-polygon, polyline-polygon",
    )
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)

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
        help="For topology_mode=targeted: e.g. touches contains intersects disjoint",
    )
    parser.add_argument(
        "--hard_case_mode",
        choices=["none", "near_disjoint", "almost_contained", "thin_polygon", "near_boundary", "mixed"],
        default="none",
    )
    parser.add_argument(
        "--normalization",
        choices=["raw", "bbox", "m00", "centroid"],
        default="raw",
    )
    args = parser.parse_args()

    # Validate pair_type
    allowed = ["point", "polyline", "polygon"]
    parts = args.pair_type.split("-")
    if len(parts) != 2 or parts[0] not in allowed or parts[1] not in allowed:
        raise SystemExit(f"Invalid --pair_type: {args.pair_type} (use e.g. point-polygon)")

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

    io_utils.write_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    io_utils.write_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    io_utils.write_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    config = {
        "task": args.task,
        "pair_type": args.pair_type,
        "n_samples": n,
        "seed": args.seed,
        "complexities": COMPLEXITIES,
        "topology_mode": args.topology_mode,
        "target_labels": args.target_labels,
        "hard_case_mode": args.hard_case_mode,
        "normalization": args.normalization,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Done. task={args.task}, pair_type={args.pair_type}, n={n}")


if __name__ == "__main__":
    main()
