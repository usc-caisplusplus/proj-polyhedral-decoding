#!/usr/bin/env bash
# generate_3d_datasets.sh
#
# Generates and converts all valid 3D geometry pair datasets.
#
# Usage:
#   bash scripts/generate_3d_datasets.sh [OUT_DIR] [N_SAMPLES]
#
# Defaults:
#   OUT_DIR   = data/3d
#   N_SAMPLES = 20000  (per task per pair type; directional uses N_SAMPLES * num_az_bins/16)
#
# Run from the repo root.

set -euo pipefail

OUT_DIR="${1:-data/3d}"
N="${2:-20000}"

PYTHON="${PYTHON:-python}"

echo "Output dir : $OUT_DIR"
echo "N per task : $N"
echo "Python     : $PYTHON"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

generate_and_convert() {
    local TASK="$1"          # topological | directional | distance
    local PAIR="$2"          # e.g. polyhedron-polyhedron
    local TOPO_MODE="$3"     # random | targeted  (ignored for non-topo tasks)
    local N_SAMPLES="$4"
    local JSONL_DIR="$OUT_DIR/jsonl/${PAIR}/${TASK}"
    local PT_FILE="$OUT_DIR/${PAIR//-/_}_${TASK}_data.pt"

    echo ">>> GENERATE  task=$TASK  pair=$PAIR  n=$N_SAMPLES"

    if [ "$TASK" = "topological" ]; then
        $PYTHON -m generator_3d.make_spatial_dataset_3d \
            --task "$TASK" \
            --pair_type "$PAIR" \
            --n_samples "$N_SAMPLES" \
            --normalization bbox \
            --out_dir "$JSONL_DIR" \
            --topology_mode "$TOPO_MODE"
    else
        $PYTHON -m generator_3d.make_spatial_dataset_3d \
            --task "$TASK" \
            --pair_type "$PAIR" \
            --n_samples "$N_SAMPLES" \
            --normalization bbox \
            --out_dir "$JSONL_DIR"
    fi

    echo ">>> CONVERT   -> $PT_FILE"

    # Determine --task flag for converter (directional uses label_id, same key name)
    CONV_TASK="$TASK"

    $PYTHON utils/convert_jsonl_to_pt_3d.py \
        --jsonl "$JSONL_DIR/train.jsonl" "$JSONL_DIR/val.jsonl" "$JSONL_DIR/test.jsonl" \
        --out "$PT_FILE" \
        --task "$CONV_TASK" \
        --pair_type "$PAIR"

    echo ""
}

mkdir -p "$OUT_DIR/jsonl"

# ---------------------------------------------------------------------------
# All valid pair types and tasks
#
# Pair types (from generator_3d/topology_sampler_3d.py dispatcher table):
#   point-point
#   point-polyhedron
#   polyhedron-point
#   polyhedron-polyhedron
#   polyline-polyhedron
#   polyline-polyline
#
# Tasks per pair type:
#   topological  — all pairs
#   directional  — all pairs
#   distance     — all pairs
# ---------------------------------------------------------------------------

# ---- point-point -----------------------------------------------------------
generate_and_convert topological   point-point          targeted "$N"
generate_and_convert directional   point-point          -        "$N"
generate_and_convert distance      point-point          -        "$N"

# ---- point-polyhedron ------------------------------------------------------
generate_and_convert topological   point-polyhedron     targeted "$N"
generate_and_convert directional   point-polyhedron     -        "$N"
generate_and_convert distance      point-polyhedron     -        "$N"

# ---- polyhedron-point ------------------------------------------------------
generate_and_convert topological   polyhedron-point     targeted "$N"
generate_and_convert directional   polyhedron-point     -        "$N"
generate_and_convert distance      polyhedron-point     -        "$N"

# ---- polyhedron-polyhedron -------------------------------------------------
generate_and_convert topological   polyhedron-polyhedron targeted "$N"
# directional has 48 classes — generate more samples for balance
generate_and_convert directional   polyhedron-polyhedron -        $((N * 3))
generate_and_convert distance      polyhedron-polyhedron -        "$N"

# ---- polyline-polyhedron ---------------------------------------------------
generate_and_convert topological   polyline-polyhedron  targeted "$N"
generate_and_convert directional   polyline-polyhedron  -        $((N * 3))
generate_and_convert distance      polyline-polyhedron  -        "$N"

# ---- polyline-polyline -----------------------------------------------------
generate_and_convert topological   polyline-polyline    targeted "$N"
generate_and_convert directional   polyline-polyline    -        $((N * 3))
generate_and_convert distance      polyline-polyline    -        "$N"

echo "============================================================"
echo "All datasets written to $OUT_DIR/"
echo "============================================================"
ls -lh "$OUT_DIR/"*.pt
