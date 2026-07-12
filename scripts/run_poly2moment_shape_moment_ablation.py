"""
Run Poly2Moment basis ablations for Geo2Vec shape and edge-count tasks.

This script systematically calls
`scripts/evaluate_poly2moment_shape_classification.py` for:

- each individual moment basis: monomial, Fourier, Legendre
- all three bases fused with projected concatenation

It then aggregates test metrics across seeds into a table with the same task
layout as the Geo2Vec shape table:

    Building Shape / Edge, MNIST Shape / Edge, Singapore Edge, NYC Edge

Examples:
    python scripts/run_poly2moment_shape_moment_ablation.py
    python scripts/run_poly2moment_shape_moment_ablation.py \
        --seeds 2024 2025 2026 --device cuda --epochs 100
    python scripts/run_poly2moment_shape_moment_ablation.py \
        --max-samples-per-class 20 --max-samples 200 --epochs 2 --device cpu
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_poly2moment_shape_classification.py"


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    display_name: str
    evaluator_args: tuple[str, ...]


@dataclass(frozen=True)
class ColumnSpec:
    column_id: str
    group_name: str
    metric_name: str
    task: str
    dataset: str
    json_metric: str
    scale: float
    higher_is_better: bool


METHODS: tuple[MethodSpec, ...] = (
    MethodSpec(
        method_id="monomial",
        display_name="Monomial",
        evaluator_args=("--single-basis", "monomial"),
    ),
    MethodSpec(
        method_id="fourier",
        display_name="Fourier",
        evaluator_args=("--single-basis", "fourier"),
    ),
    MethodSpec(
        method_id="legendre",
        display_name="Legendre",
        evaluator_args=("--single-basis", "legendre"),
    ),
    MethodSpec(
        method_id="projected_concat",
        display_name="Projected concat",
        evaluator_args=(
            "--basis-types",
            "monomial",
            "fourier",
            "legendre",
        ),
    ),
)


COLUMNS: tuple[ColumnSpec, ...] = (
    ColumnSpec(
        column_id="building_shape",
        group_name="Building",
        metric_name="Shape↑",
        task="classification",
        dataset="building",
        json_metric="accuracy",
        scale=100.0,
        higher_is_better=True,
    ),
    ColumnSpec(
        column_id="building_edge",
        group_name="Building",
        metric_name="Edge↓",
        task="edge-count",
        dataset="building",
        json_metric="mae",
        scale=1.0,
        higher_is_better=False,
    ),
    ColumnSpec(
        column_id="mnist_shape",
        group_name="MNIST",
        metric_name="Shape↑",
        task="classification",
        dataset="mnist",
        json_metric="accuracy",
        scale=100.0,
        higher_is_better=True,
    ),
    ColumnSpec(
        column_id="mnist_edge",
        group_name="MNIST",
        metric_name="Edge↓",
        task="edge-count",
        dataset="mnist",
        json_metric="mae",
        scale=1.0,
        higher_is_better=False,
    ),
    ColumnSpec(
        column_id="singapore_edge",
        group_name="Singapore",
        metric_name="Edge↓",
        task="edge-count",
        dataset="singapore",
        json_metric="mae",
        scale=1.0,
        higher_is_better=False,
    ),
    ColumnSpec(
        column_id="nyc_edge",
        group_name="NYC",
        metric_name="Edge↓",
        task="edge-count",
        dataset="nyc",
        json_metric="mae",
        scale=1.0,
        higher_is_better=False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run individual-moment and projected-concat Poly2Moment "
            "ablations for shape classification and edge-count tasks."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results" / "poly2moment_shape_moment_ablation",
        help="Directory for per-run JSON outputs and aggregate tables.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[2024],
        help="Random seeds to run. Multiple seeds produce mean±std cells.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[method.method_id for method in METHODS],
        default=None,
        help="Subset of methods to run. Defaults to all methods.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        choices=[column.column_id for column in COLUMNS],
        default=None,
        help="Subset of table columns/tasks to run. Defaults to all columns.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--classifier-hidden-size", type=int, default=256)
    parser.add_argument("--classifier-dropout", type=float, default=0.5)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument(
        "--max-order",
        type=int,
        default=None,
        help="Override config.json moment_params.max_order.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional stratified classification subset for smoke tests.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional random edge-count subset for smoke tests.",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=ROOT / "results" / "poly2moment_shape_cache",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun evaluations even when per-run JSON already exists.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after failed runs and aggregate completed results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without running them.",
    )
    return parser.parse_args()


def selected_methods(method_ids: list[str] | None) -> list[MethodSpec]:
    allowed = set(method_ids) if method_ids is not None else None
    return [
        method
        for method in METHODS
        if allowed is None or method.method_id in allowed
    ]


def selected_columns(column_ids: list[str] | None) -> list[ColumnSpec]:
    allowed = set(column_ids) if column_ids is not None else None
    return [
        column
        for column in COLUMNS
        if allowed is None or column.column_id in allowed
    ]


def run_output_path(
    results_dir: Path,
    method: MethodSpec,
    column: ColumnSpec,
    seed: int,
) -> Path:
    return (
        results_dir
        / "runs"
        / method.method_id
        / column.task
        / column.dataset
        / f"seed_{seed}.json"
    )


def build_command(
    args: argparse.Namespace,
    method: MethodSpec,
    column: ColumnSpec,
    seed: int,
    output_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(EVALUATOR),
        "--task",
        column.task,
        "--dataset",
        column.dataset,
        "--seed",
        str(seed),
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--classifier-hidden-size",
        str(args.classifier_hidden_size),
        "--classifier-dropout",
        str(args.classifier_dropout),
        "--feature-batch-size",
        str(args.feature_batch_size),
        "--classifier-batch-size",
        str(args.classifier_batch_size),
        "--feature-cache-dir",
        str(args.feature_cache_dir),
        "--output",
        str(output_path),
        *method.evaluator_args,
    ]
    if args.max_order is not None:
        cmd.extend(["--max-order", str(args.max_order)])
    if column.task == "classification" and args.max_samples_per_class is not None:
        cmd.extend(["--max-samples-per-class", str(args.max_samples_per_class)])
    if column.task == "edge-count" and args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    if args.rebuild_features:
        cmd.append("--rebuild-features")
    return cmd


def print_command(cmd: list[str]) -> None:
    print(" ".join(cmd))


def run_one(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    log_text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(log_text, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}; see {log_path}"
        )


def extract_metric(output_path: Path, column: ColumnSpec) -> float:
    with output_path.open("r", encoding="utf-8") as file:
        result = json.load(file)
    dataset_result = result["datasets"][column.dataset]
    value = dataset_result["metrics"]["test"][column.json_metric]
    return float(value) * column.scale


def summarize(values: list[float]) -> dict[str, float | int]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return {"n": 0, "mean": math.nan, "std": math.nan}
    return {
        "n": len(clean),
        "mean": mean(clean),
        "std": stdev(clean) if len(clean) > 1 else 0.0,
    }


def format_cell(summary: dict[str, float | int], bold: bool = False) -> str:
    if int(summary["n"]) == 0:
        return "—"
    text = f"{float(summary['mean']):.2f}±{float(summary['std']):.3f}"
    return f"**{text}**" if bold else text


def best_method_ids(
    table: dict[str, dict[str, dict[str, float | int]]],
    columns: list[ColumnSpec],
) -> dict[str, str | None]:
    best: dict[str, str | None] = {}
    for column in columns:
        scored: list[tuple[str, float]] = []
        for method_id, row in table.items():
            summary = row.get(column.column_id)
            if summary is None or int(summary["n"]) == 0:
                continue
            value = float(summary["mean"])
            if math.isfinite(value):
                scored.append((method_id, value))
        if not scored:
            best[column.column_id] = None
            continue
        scored.sort(key=lambda item: item[1], reverse=column.higher_is_better)
        best[column.column_id] = scored[0][0]
    return best


def write_raw_csv(
    path: Path,
    rows: list[dict[str, str | int | float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method_id",
        "method",
        "column_id",
        "dataset",
        "task",
        "metric",
        "seed",
        "value",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_csv(
    path: Path,
    table: dict[str, dict[str, dict[str, float | int]]],
    methods: list[MethodSpec],
    columns: list[ColumnSpec],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method_id", "method"]
    for column in columns:
        fieldnames.extend(
            [
                f"{column.column_id}_mean",
                f"{column.column_id}_std",
                f"{column.column_id}_n",
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method in methods:
            row: dict[str, str | int | float] = {
                "method_id": method.method_id,
                "method": method.display_name,
            }
            for column in columns:
                summary = table[method.method_id][column.column_id]
                row[f"{column.column_id}_mean"] = float(summary["mean"])
                row[f"{column.column_id}_std"] = float(summary["std"])
                row[f"{column.column_id}_n"] = int(summary["n"])
            writer.writerow(row)


def write_markdown_table(
    path: Path,
    table: dict[str, dict[str, dict[str, float | int]]],
    methods: list[MethodSpec],
    columns: list[ColumnSpec],
) -> None:
    best = best_method_ids(table, columns)
    header = ["Method"] + [
        f"{column.group_name} {column.metric_name}" for column in columns
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for method in methods:
        cells = [method.display_name]
        for column in columns:
            summary = table[method.method_id][column.column_id]
            cells.append(
                format_cell(summary, bold=best[column.column_id] == method.method_id)
            )
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def escape_latex(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("↑", r"$\uparrow$")
        .replace("↓", r"$\downarrow$")
    )


def latex_cell(summary: dict[str, float | int], bold: bool = False) -> str:
    if int(summary["n"]) == 0:
        return "--"
    text = f"{float(summary['mean']):.2f}$\\pm${float(summary['std']):.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def write_latex_table(
    path: Path,
    table: dict[str, dict[str, dict[str, float | int]]],
    methods: list[MethodSpec],
    columns: list[ColumnSpec],
) -> None:
    best = best_method_ids(table, columns)
    group_spans: list[tuple[str, int]] = []
    for column in columns:
        if group_spans and group_spans[-1][0] == column.group_name:
            group_spans[-1] = (group_spans[-1][0], group_spans[-1][1] + 1)
        else:
            group_spans.append((column.group_name, 1))

    alignment = "l" + "c" * len(columns)
    group_header = [""] + [
        f"\\multicolumn{{{span}}}{{c}}{{{escape_latex(group)}}}"
        for group, span in group_spans
    ]
    metric_header = [""] + [escape_latex(column.metric_name) for column in columns]

    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\toprule",
        " & ".join(group_header) + r" \\",
        " & ".join(metric_header) + r" \\",
        "\\midrule",
    ]
    for method in methods:
        cells = [escape_latex(method.display_name)]
        for column in columns:
            summary = table[method.method_id][column.column_id]
            cells.append(
                latex_cell(summary, bold=best[column.column_id] == method.method_id)
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    methods = selected_methods(args.methods)
    columns = selected_columns(args.columns)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    total = len(methods) * len(columns) * len(args.seeds)
    print(f"Planned runs: {total}")

    failures: list[str] = []
    for method in methods:
        for column in columns:
            for seed in args.seeds:
                output_path = run_output_path(args.results_dir, method, column, seed)
                log_path = output_path.with_suffix(".log")
                cmd = build_command(args, method, column, seed, output_path)
                label = (
                    f"{method.method_id} | {column.dataset} | "
                    f"{column.task} | seed={seed}"
                )
                if args.dry_run:
                    print_command(cmd)
                    continue
                if output_path.is_file() and not args.force:
                    print(f"Skipping existing run: {label}")
                    continue
                print(f"Running: {label}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    run_one(cmd, log_path)
                except RuntimeError as error:
                    failures.append(str(error))
                    print(f"FAILED: {error}")
                    if not args.keep_going:
                        raise

    if args.dry_run:
        return

    raw_rows: list[dict[str, str | int | float]] = []
    table: dict[str, dict[str, dict[str, float | int]]] = {}
    for method in methods:
        table[method.method_id] = {}
        for column in columns:
            values = []
            for seed in args.seeds:
                output_path = run_output_path(args.results_dir, method, column, seed)
                if not output_path.is_file():
                    continue
                value = extract_metric(output_path, column)
                values.append(value)
                raw_rows.append(
                    {
                        "method_id": method.method_id,
                        "method": method.display_name,
                        "column_id": column.column_id,
                        "dataset": column.dataset,
                        "task": column.task,
                        "metric": column.json_metric,
                        "seed": seed,
                        "value": value,
                    }
                )
            table[method.method_id][column.column_id] = summarize(values)

    write_raw_csv(args.results_dir / "raw_metrics.csv", raw_rows)
    write_summary_csv(args.results_dir / "summary.csv", table, methods, columns)
    write_markdown_table(args.results_dir / "result_table.md", table, methods, columns)
    write_latex_table(args.results_dir / "result_table.tex", table, methods, columns)

    with (args.results_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "seeds": args.seeds,
                "methods": [method.method_id for method in methods],
                "columns": [column.column_id for column in columns],
                "table": table,
                "failures": failures,
            },
            file,
            indent=2,
            sort_keys=True,
        )

    print(f"\nWrote {args.results_dir / 'result_table.md'}")
    print(f"Wrote {args.results_dir / 'result_table.tex'}")
    print(f"Wrote {args.results_dir / 'summary.csv'}")
    if failures:
        print(f"Completed with {len(failures)} failed run(s); see summary.json.")


if __name__ == "__main__":
    main()
