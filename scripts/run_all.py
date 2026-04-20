"""
Run *all* dataset tasks on *both* datasets (NewYork, Singapore) for all moment
families (Legendre / Fourier / Monomial) by repeatedly writing `config.json`
and invoking `run.py`.

Why this approach:
- `run.py` always loads `config.json` via `args_parser("config.json")`
- `run.py` CLI overrides do NOT cover `moment_params` / most nested config

This script keeps core training code unchanged and just orchestrates runs +
produces a clean results folder layout.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# ---- specs -------------------------------------------------------------------


_GEOM_PLURAL = {"point": "points", "polyline": "polylines", "polygon": "polygons"}


@dataclass(frozen=True)
class TaskSpec:
    group: str  # topo | directional | distance
    task_id: str  # stable id used in results folder + tables (e.g. "dir_point_polyline")
    dataset: str  # value written to config.json's "dataset"
    task: str  # run.py task (direction-prediction, distance-prediction, ...)
    dataset_type1: str
    dataset_type2: str
    data_file: str  # with leading "/"
    num_classes: int


@dataclass(frozen=True)
class EncoderSpec:
    encoder_id: str  # legendre | fourier | monomial
    moment_params: dict[str, Any]


_ENCODERS: list[EncoderSpec] = [
    EncoderSpec(
        encoder_id="legendre",
        moment_params={"basis_type": "legendre", "max_order": 14, "moment_norm": "m00_scale"},
    ),
    EncoderSpec(
        encoder_id="fourier",
        moment_params={"basis_type": "fourier", "max_order": 14, "moment_norm": "m00_scale"},
    ),
    EncoderSpec(
        encoder_id="monomial",
        moment_params={"basis_type": "monomial", "max_order": 14, "moment_norm": "m00_scale"},
    ),
]


def _now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---- task discovery -----------------------------------------------------------


def _infer_types_from_filename(name: str) -> tuple[str, str]:
    """
    Examples:
    - point_point_dis_data.pt -> (points, points)
    - polygon_polyline_directional_relationship_360_data.pt -> (polygons, polylines)
    """
    m = re.match(r"^(point|polyline|polygon)_(point|polyline|polygon)_", name)
    if not m:
        raise ValueError(f"Cannot infer types from filename: {name}")
    t1, t2 = m.group(1), m.group(2)
    return _GEOM_PLURAL[t1], _GEOM_PLURAL[t2]


def _task_from_filename(name: str) -> tuple[str, str, str, int]:
    """
    Returns (group, task, dataset_id, num_classes_placeholder).

    Note: for multi-class tasks, `run.py` will infer `num_classes` if it can.
    We still provide a sane placeholder so config parsing stays typed.
    """
    if name.endswith("_dis_data.pt"):
        group = "distance"
        task = "distance-prediction"
        dataset_id = "dist_" + name.replace("_dis_data.pt", "")
        return group, task, dataset_id, 1

    if "directional_relationship_360" in name:
        group = "directional"
        task = "direction-prediction"
        dataset_id = "dir_" + name.replace("_directional_relationship_360_data.pt", "")
        return group, task, dataset_id, 16

    if name.endswith("_contain_data.pt"):
        group = "topo"
        task = "relation-prediction"
        dataset_id = "topo_" + name.replace("_contain_data.pt", "_contain")
        return group, task, dataset_id, 1

    if name.endswith("_intersect_data.pt"):
        group = "topo"
        task = "relation-prediction"
        dataset_id = "topo_" + name.replace("_intersect_data.pt", "_intersect")
        return group, task, dataset_id, 1

    if name.endswith("_topological_relationship_data.pt"):
        group = "topo"
        task = "multi-relation"
        dataset_id = "topo_" + name.replace("_topological_relationship_data.pt", "_multiclass")
        # placeholder; run.py will try to infer from labels
        placeholder = 6 if "polygon_polygon_topological_relationship" in name else 2
        return group, task, dataset_id, placeholder

    raise ValueError(f"Unrecognized dataset file pattern: {name}")


def discover_tasks(dataset_dir: Path) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for p in sorted(dataset_dir.glob("*.pt")):
        name = p.name
        group, task, dataset_id, num_classes = _task_from_filename(name)
        t1, t2 = _infer_types_from_filename(name)
        task_id = dataset_id  # keep stable id for tables + folders
        tasks.append(
            TaskSpec(
                group=group,
                task_id=task_id,
                dataset=dataset_id,
                task=task,
                dataset_type1=t1,
                dataset_type2=t2,
                data_file="/" + name,
                num_classes=num_classes,
            )
        )
    return tasks


# ---- run orchestration --------------------------------------------------------


_METRICS_SAVED_RE = re.compile(r"Test metrics saved to (.+/test_metrics\.json)\s*$", re.MULTILINE)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


def _unique_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 10_000):
        cand = Path(str(path) + f"_run{i}")
        if not cand.exists():
            return cand
    raise RuntimeError(f"Could not find unique directory for {path}")


def _run_one(*, project_root: Path, stdout_dir: Path) -> tuple[Path, str]:
    """
    Runs `python run.py` in `project_root` and returns:
    - the directory containing `test_metrics.json` + `args.json`
    - stdout text
    """
    proc = subprocess.run(
        [sys.executable, "run.py"],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        _ensure_dir(stdout_dir)
        (stdout_dir / "last_failed_stdout.txt").write_text(out, encoding="utf-8")
        raise RuntimeError(
            f"run.py failed with exit code {proc.returncode}. "
            f"See {stdout_dir / 'last_failed_stdout.txt'}"
        )
    m = _METRICS_SAVED_RE.search(out)
    if not m:
        _ensure_dir(stdout_dir)
        (stdout_dir / "last_stdout.txt").write_text(out, encoding="utf-8")
        raise RuntimeError(
            "Could not locate metrics path in run.py output. "
            f"Wrote stdout to {stdout_dir / 'last_stdout.txt'}"
        )
    metrics_path = Path(m.group(1)).resolve()
    run_dir = metrics_path.parent
    return run_dir, out


def _build_run_config(
    base: dict[str, Any],
    *,
    dataset_name: str,
    data_path: str,
    task: TaskSpec,
    encoder: EncoderSpec,
    epochs: int,
    patience: int,
    device: str | None,
) -> dict[str, Any]:
    cfg = dict(base)

    cfg["dataset_name"] = dataset_name
    cfg["data_path"] = data_path

    cfg["dataset"] = task.dataset
    cfg["task"] = task.task
    cfg["dataset_type1"] = task.dataset_type1
    cfg["dataset_type2"] = task.dataset_type2
    cfg["data_file"] = task.data_file
    cfg["num_classes"] = int(task.num_classes)

    cfg["encoder_type"] = "poly2moment"
    cfg["moment_params"] = dict(encoder.moment_params)

    # keep this aligned with moment basis to avoid downstream ambiguity
    cfg["sampling_strategy"] = str(encoder.moment_params.get("basis_type", cfg.get("sampling_strategy", "fourier")))

    cfg["epochs"] = int(epochs)
    cfg["patience"] = int(patience)
    cfg["training"] = True

    if device is not None:
        cfg["device"] = device

    # set save_dir to a temporary staging dir; we'll move/rename after the run
    cfg["save_dir"] = "./results/_staging"
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Run all tasks × datasets × encoders via config.json rewriting.")
    p.add_argument("--results_root", type=str, default="./results_v2", help="Clean output root")
    p.add_argument("--epochs", type=int, default=50, help="Epochs per run (use 1 for quick smoke tests)")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", type=str, default=None, help='Override config.json device ("cpu" or "cuda")')
    p.add_argument("--max_runs", type=int, default=None, help="Cap total runs (debugging)")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.json"
    original_config_text = config_path.read_text(encoding="utf-8")
    base_cfg = json.loads(original_config_text)
    results_root = (project_root / args.results_root).resolve()
    _ensure_dir(results_root)

    datasets: list[tuple[str, Path]] = [
        ("NewYork", project_root / "data" / "NewYork"),
        ("Singapore", project_root / "data" / "Singapore"),
    ]

    all_plans: list[tuple[str, TaskSpec, EncoderSpec]] = []
    for dataset_name, dataset_dir in datasets:
        tasks = discover_tasks(dataset_dir)
        for t in tasks:
            for e in _ENCODERS:
                all_plans.append((dataset_name, t, e))

    if args.max_runs is not None:
        all_plans = all_plans[: int(args.max_runs)]

    print(f"Planned runs: {len(all_plans)}")

    try:
        for idx, (dataset_name, task, enc) in enumerate(all_plans, start=1):
            run_label = f"{dataset_name} | {task.task_id} | {enc.encoder_id}"
            print(f"\n[{idx}/{len(all_plans)}] {run_label}")

            cfg = _build_run_config(
                base_cfg,
                dataset_name=dataset_name,
                data_path=f"./data/{dataset_name}",
                task=task,
                encoder=enc,
                epochs=int(args.epochs),
                patience=int(args.patience),
                device=args.device,
            )

            if args.dry_run:
                continue

            _write_json(config_path, cfg)

            stdout_dir = results_root / "_logs"
            run_dir, out = _run_one(project_root=project_root, stdout_dir=stdout_dir)

            # Move into a clean, shallow layout:
            # results_v2/<Dataset>/<Group>/<Task>/<Encoder>/
            dest = results_root / dataset_name / task.group / task.task_id / enc.encoder_id
            dest = _unique_dir(dest)
            _ensure_dir(dest.parent)
            shutil.move(str(run_dir), str(dest))

            # Keep stdout attached to the run for debugging/repro.
            (dest / "stdout.txt").write_text(out, encoding="utf-8")

            # Optional: if staging got recreated, keep it empty-ish.
            staging = project_root / "results" / "_staging"
            if staging.exists():
                # do not delete (safety), but we can leave it as-is
                pass

    finally:
        config_path.write_text(original_config_text, encoding="utf-8")

    print(f"\nDone. Results at: {results_root}")
    print("Next: python scripts/generate_tables.py --results_dir ./results_v2 --out_dir ./tables_v2")


if __name__ == "__main__":
    main()
