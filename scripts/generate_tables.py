"""
Aggregate metrics from a clean results tree and write Markdown tables.

Default layout produced by `scripts/run_all.py`:
results_v2/<Dataset>/<Group>/<Task>/<Encoder>[/...]/test_metrics.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


CITY_DISPLAY = {"Singapore": "Singapore", "NewYork": "New York"}

ALL_MOMENTS_CONCAT = "ALL_MOMENTS_CONCAT"
METHOD_ORDER = ["LEGENDRE", "FOURIER", "MONOMIAL", ALL_MOMENTS_CONCAT]
METHOD_DISPLAY = {
    "LEGENDRE": "Legendre moments",
    "FOURIER": "Fourier moments",
    "MONOMIAL": "Monomial (geometric moments)",
    ALL_MOMENTS_CONCAT: "All moments (concatenated)",
}


@dataclass(frozen=True)
class _RunRecord:
    metrics_path: Path
    args_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_runs(results_dir: Path) -> list[_RunRecord]:
    runs: list[_RunRecord] = []
    for metrics_path in sorted(results_dir.rglob("test_metrics.json")):
        args_path = metrics_path.parent / "args.json"
        if args_path.exists():
            runs.append(_RunRecord(metrics_path=metrics_path, args_path=args_path))
    return runs


def _task_group(args: dict[str, Any]) -> str:
    ds = str(args.get("dataset", "") or "")
    task = str(args.get("task", "") or "")
    if ds.startswith("topo_") or task in {"relation-prediction", "multi-relation"}:
        return "topo"
    if ds.startswith("dir_") or task == "direction-prediction":
        return "directional"
    if ds.startswith("dist_") or task == "distance-prediction":
        return "distance"
    return "other"


def _method_name(args: dict[str, Any]) -> str:
    mp = args.get("moment_params") or {}
    if not isinstance(mp, dict):
        mp = {}

    basis_types = mp.get("basis_types")
    if (
        isinstance(basis_types, (list, tuple))
        and len(basis_types) > 1
    ):
        return ALL_MOMENTS_CONCAT

    bt = str(mp.get("basis_type", "") or "")
    if bt == "legendre":
        return "LEGENDRE"
    if bt == "fourier":
        return "FOURIER"
    if bt == "monomial":
        return "MONOMIAL"
    return str(args.get("encoder_type", "METHOD")).upper()


def collect_metrics_df(results_dir: Path) -> pd.DataFrame:
    results_dir = results_dir.resolve()
    runs = _find_runs(results_dir)
    if not runs:
        raise SystemExit(f"No runs found under {results_dir} (need test_metrics.json + args.json)")

    base = results_dir.parent
    rows: list[dict[str, Any]] = []
    for r in runs:
        a = _read_json(r.args_path)
        m = _read_json(r.metrics_path)
        mp = a.get("moment_params") or {}
        if not isinstance(mp, dict):
            mp = {}
        rows.append(
            {
                "file": str(r.metrics_path.relative_to(base)),
                "save_dir": str(r.metrics_path.parent.relative_to(base)),
                "group": _task_group(a),
                "dataset_name": a.get("dataset_name"),
                "dataset": a.get("dataset"),
                "task": a.get("task"),
                "method": _method_name(a),
                "basis_type": mp.get("basis_type"),
                "basis_types": json.dumps(mp.get("basis_types"))
                if mp.get("basis_types") is not None
                else None,
                "feature_fusion": mp.get("feature_fusion"),
                "projection_dim": mp.get("projection_dim"),
                "max_order": mp.get("max_order"),
                "epochs": a.get("epochs"),
                **m,
            }
        )

    df = pd.DataFrame(rows)
    for c in ["accuracy", "MAE", "MSE", "f1", "aucroc", "auprc", "precision", "recall"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _dataframe_to_pipe_markdown(df: pd.DataFrame, *, index_header: str) -> str:
    cols = df.columns
    header_cells = [f"{a} / {b}" for a, b in cols] if isinstance(cols, pd.MultiIndex) else [str(c) for c in cols]
    header = [index_header] + header_cells
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for idx, row in df.iterrows():
        body = [str(idx)] + [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(body) + " |")
    return "\n".join(lines)


def _decorate_cells(piv: pd.DataFrame, *, higher_is_better: bool) -> pd.DataFrame:
    out = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=object)
    for col in piv.columns:
        s = pd.to_numeric(piv[col], errors="coerce")
        scored = [(idx, float(s.loc[idx])) for idx in s.index if not pd.isna(s.loc[idx])]
        scored.sort(key=lambda t: t[1], reverse=higher_is_better)
        best_idx = scored[0][0] if scored else None
        for idx in piv.index:
            v = piv.loc[idx, col]
            if pd.isna(v):
                out.loc[idx, col] = "—"
                continue
            txt = f"{float(v):.4f}"
            out.loc[idx, col] = f"**{txt}**" if idx == best_idx else txt
    return out


def _pretty_columns(piv: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(piv.columns, pd.MultiIndex):
        return piv
    new_cols = [(CITY_DISPLAY.get(str(a), str(a)), str(b)) for a, b in piv.columns]
    out = piv.copy()
    out.columns = pd.MultiIndex.from_tuples(new_cols)
    return out


def _write_table(
    df: pd.DataFrame,
    *,
    out_path: Path,
    group: str,
    metric: str,
    higher_is_better: bool,
    title: str,
) -> None:
    sub = df[df["group"] == group].copy()
    if sub.empty:
        out_path.write_text(f"{title}\n\n(No runs found for group `{group}`.)\n", encoding="utf-8")
        return

    # Prefer the best run if duplicates exist (e.g., reruns).
    sub = sub.sort_values(metric, ascending=not higher_is_better, na_position="last")
    sub = sub.drop_duplicates(subset=["method", "dataset_name", "dataset"], keep="first")

    tasks = sorted(sub["dataset"].dropna().unique().tolist())
    ordered_cols = [(ds, task) for ds in ["NewYork", "Singapore"] for task in tasks]

    piv = sub.pivot_table(
        index="method",
        columns=["dataset_name", "dataset"],
        values=metric,
        aggfunc="first",
    )
    piv = piv.reindex(columns=pd.MultiIndex.from_tuples(ordered_cols))
    piv = piv.reindex(index=METHOD_ORDER)

    piv = _decorate_cells(_pretty_columns(piv), higher_is_better=higher_is_better)
    piv = piv.rename(index=METHOD_DISPLAY)
    md = _dataframe_to_pipe_markdown(piv, index_header="Methods")
    out_path.write_text(title + "\n\n" + md + "\n", encoding="utf-8")


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_table(
        df,
        out_path=out_dir / "table_dir_accuracy.md",
        group="directional",
        metric="accuracy",
        higher_is_better=True,
        title="Directional prediction accuracy by moment family",
    )
    _write_table(
        df,
        out_path=out_dir / "table_dist_mae.md",
        group="distance",
        metric="MAE",
        higher_is_better=False,
        title="Distance prediction MAE (lower is better) by moment family",
    )
    _write_table(
        df,
        out_path=out_dir / "table_topo_accuracy.md",
        group="topo",
        metric="accuracy",
        higher_is_better=True,
        title="Topological relation accuracy by moment family",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Collect metrics and write Markdown tables.")
    p.add_argument("--results_dir", type=str, default="./results_v2", help="Scan for test_metrics.json + args.json")
    p.add_argument("--csv_out", type=str, default="all_metrics_v2.csv", help="Write aggregated CSV")
    p.add_argument("--from_csv", type=str, default=None, help="Skip scan; load this CSV instead")
    p.add_argument("--out_dir", type=str, default="./tables", help="Output directory for markdown tables")
    p.add_argument("--no_csv", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()

    if args.from_csv:
        df = pd.read_csv(Path(args.from_csv).resolve())
    else:
        df = collect_metrics_df(Path(args.results_dir))
        if not args.no_csv:
            csv_path = Path(args.csv_out).resolve()
            df.to_csv(csv_path, index=False)
            print(f"Wrote {len(df)} rows to {csv_path}")

    write_tables(df, out_dir)
    print(
        f"Wrote {out_dir / 'table_topo_accuracy.md'}, "
        f"{out_dir / 'table_dir_accuracy.md'}, {out_dir / 'table_dist_mae.md'}"
    )


if __name__ == "__main__":
    main()
