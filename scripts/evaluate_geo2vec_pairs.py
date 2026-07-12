"""
Evaluate Geo2Vec embeddings on the repository's 2D pair datasets.

Geo2Vec is transductive: it learns one latent vector per geometry ID while
fitting that geometry's signed-distance field. This runner therefore:

1. deduplicates geometries occurring in a pair dataset,
2. learns unsupervised Geo2Vec location and/or shape embeddings,
3. freezes and caches those embeddings, and
4. trains a pair classifier on concatenated embeddings.

The SDF stage sees every geometry, including geometries occurring in held-out
pairs, but never sees pair labels. This matches a transductive representation
evaluation; it is not an unseen-geometry evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


ROOT = Path(__file__).resolve().parents[1]
GEO2VEC_ROOT = ROOT / "Geo2Vec"

# Geo2Vec uses top-level imports such as "models.Geo2Vec" and
# "utils.preprocess", so its directory must precede this repository root.
sys.path.insert(0, str(GEO2VEC_ROOT))

from models import MP_Sampling  # noqa: E402
from models.Geo2Vec import (  # noqa: E402
    Geo2Vec_Dataset,
    Geo2Vec_Model,
    SDFLoss,
    identity_collate,
)
from utils.preprocess import normalize_geometries, poly_preprocess  # noqa: E402


GEOMETRY_TYPE_ALIASES = {
    "point": "point",
    "points": "point",
    "polyline": "polyline",
    "polylines": "polyline",
    "line": "polyline",
    "lines": "polyline",
    "polygon": "polygon",
    "polygons": "polygon",
}

PROFILE_DEFAULTS = {
    # Intended for validating the pipeline before committing a large run.
    "quick": {
        "samples_per_unit": 100,
        "point_samples": 5,
        "uniform_grid_size": 5,
        "sdf_epochs": 2,
        "shape_samples_per_unit": 30,
        "shape_point_samples": 5,
        "shape_uniform_grid_size": 5,
        "shape_sdf_epochs": 2,
    },
    # Defaults in Geo2Vec/runners/list2embedding.py for both SDF branches.
    "released": {
        "samples_per_unit": 4000,
        "point_samples": 10,
        "uniform_grid_size": 30,
        "sdf_epochs": 2,
        "shape_samples_per_unit": 100,
        "shape_point_samples": 20,
        "shape_uniform_grid_size": 20,
        "shape_sdf_epochs": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate transductive Geo2Vec embeddings on a 2D pair .pt dataset."
    )
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--type1", required=True, choices=sorted(GEOMETRY_TYPE_ALIASES))
    parser.add_argument("--type2", required=True, choices=sorted(GEOMETRY_TYPE_ALIASES))
    parser.add_argument("--task", required=True, choices=("topology", "direction"))
    parser.add_argument("--profile", choices=tuple(PROFILE_DEFAULTS), default="quick")
    parser.add_argument(
        "--representation",
        choices=("location", "shape", "both"),
        default="both",
        help="Entity vector supplied to the downstream pair classifier",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--sdf-hidden-size", type=int, default=256)
    parser.add_argument("--sdf-num-freqs", type=int, default=16)
    parser.add_argument("--sdf-num-layers", type=int, default=8)
    parser.add_argument("--sdf-batch-size", type=int, default=20_480)
    parser.add_argument("--sdf-learning-rate", type=float, default=1e-3)
    parser.add_argument("--sdf-training-ratio", type=float, default=0.95)
    parser.add_argument("--sample-band-width", type=float, default=0.1)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument(
        "--coordinate-mode",
        choices=("pre_normalized", "fit_bounds", "none"),
        default="pre_normalized",
        help=(
            "pre_normalized preserves the pair files' shared [-1, 1] city frame "
            "and maps it to [0, 1]; fit_bounds reproduces Geo2Vec normalization "
            "for raw coordinates; none leaves coordinates unchanged"
        ),
    )
    parser.add_argument("--samples-per-unit", type=int, default=None)
    parser.add_argument("--point-samples", type=int, default=None)
    parser.add_argument("--uniform-grid-size", type=int, default=None)
    parser.add_argument("--sdf-epochs", type=int, default=None)

    parser.add_argument("--shape-samples-per-unit", type=int, default=None)
    parser.add_argument("--shape-point-samples", type=int, default=None)
    parser.add_argument("--shape-uniform-grid-size", type=int, default=None)
    parser.add_argument("--shape-sdf-epochs", type=int, default=None)
    parser.add_argument("--shape-num-freqs", type=int, default=8)
    parser.add_argument("--shape-code-reg-weight", type=float, default=1.0)
    parser.add_argument(
        "--shape-rotation-invariant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Augment shape positional encoding with radial Fourier features",
    )

    parser.add_argument("--classifier-hidden-size", type=int, default=256)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--classifier-epochs", type=int, default=100)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--classifier-patience", type=int, default=15)

    parser.add_argument("--embedding-cache", type=Path, default=None)
    parser.add_argument("--rebuild-embeddings", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_type(value: str) -> str:
    return GEOMETRY_TYPE_ALIASES[value]


def select_pair_indices(
    labels: torch.Tensor, max_pairs: int | None, seed: int
) -> torch.Tensor:
    if max_pairs is None or max_pairs >= len(labels):
        return torch.arange(len(labels))
    if max_pairs < labels.unique().numel() * 5:
        raise ValueError("--max-pairs must leave at least five examples per class")

    indices, _ = train_test_split(
        np.arange(len(labels)),
        train_size=max_pairs,
        random_state=seed,
        stratify=labels.cpu().numpy(),
    )
    return torch.from_numpy(np.sort(indices))


def coordinates_for(
    row: torch.Tensor, length: torch.Tensor, geometry_type: str
) -> np.ndarray:
    row = row.detach().cpu()
    if geometry_type == "point":
        coords = row.reshape(-1)[:2].numpy()
        if coords.shape != (2,):
            raise ValueError(f"Expected point coordinates with shape (2,), got {coords.shape}")
        return np.ascontiguousarray(coords, dtype=np.float64)

    count = int(length.item())
    coords = row[:count].numpy()
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coordinate sequence with shape (N, 2), got {coords.shape}")
    minimum = 3 if geometry_type == "polygon" else 2
    if len(coords) < minimum:
        raise ValueError(
            f"{geometry_type} requires at least {minimum} coordinates, got {len(coords)}"
        )
    return np.ascontiguousarray(coords, dtype=np.float64)


def shapely_geometry(coords: np.ndarray, geometry_type: str) -> Any:
    if not np.isfinite(coords).all():
        raise ValueError("Geometry contains NaN or infinite coordinates")
    if geometry_type == "point":
        return Point(float(coords[0]), float(coords[1]))
    if geometry_type == "polyline":
        return LineString(coords)
    if geometry_type == "polygon":
        return Polygon(coords)
    raise AssertionError(f"Unhandled geometry type: {geometry_type}")


def geometry_key(coords: np.ndarray, geometry_type: str) -> tuple[Any, ...]:
    # Pair files use float32 coordinates. Keying their exact valid-coordinate
    # bytes preserves intentionally translated/scaled variants as new entities.
    coords32 = np.ascontiguousarray(coords, dtype=np.float32)
    return geometry_type, coords32.shape, coords32.tobytes()


def deduplicate_pair_geometries(
    x1: torch.Tensor,
    x2: torch.Tensor,
    l1: torch.Tensor,
    l2: torch.Tensor,
    type1: str,
    type2: str,
) -> tuple[list[Any], torch.Tensor]:
    geometries: list[Any] = []
    key_to_id: dict[tuple[Any, ...], int] = {}
    pair_ids = torch.empty((len(x1), 2), dtype=torch.long)

    for pair_index in range(len(x1)):
        for side, (x, lengths, geometry_type) in enumerate(
            ((x1, l1, type1), (x2, l2, type2))
        ):
            coords = coordinates_for(x[pair_index], lengths[pair_index], geometry_type)
            key = geometry_key(coords, geometry_type)
            entity_id = key_to_id.get(key)
            if entity_id is None:
                entity_id = len(geometries)
                key_to_id[key] = entity_id
                geometries.append(shapely_geometry(coords, geometry_type))
            pair_ids[pair_index, side] = entity_id

    return geometries, pair_ids


def prepare_location_geometries(
    geometries: list[Any], coordinate_mode: str
) -> list[Any]:
    if coordinate_mode == "pre_normalized":
        # Poly2Vec pair tensors already share a city-wide [-1, 1] coordinate
        # frame. Apply the same fixed transform to both axes so topology and
        # centroid angles are preserved while matching Geo2Vec's [0, 1] range.
        return [
            affinity.scale(
                affinity.translate(geometry, xoff=1.0, yoff=1.0),
                xfact=0.5,
                yfact=0.5,
                origin=(0.0, 0.0),
            )
            for geometry in geometries
        ]
    if coordinate_mode == "fit_bounds":
        return normalize_geometries(geometries)
    if coordinate_mode == "none":
        return geometries
    raise ValueError(f"Unsupported coordinate mode: {coordinate_mode}")


def train_sdf_embeddings(
    geometries: list[Any],
    args: argparse.Namespace,
    *,
    branch: str,
    samples_per_unit: int,
    point_samples: int,
    uniform_grid_size: int,
    epochs: int,
    num_freqs: int,
    log_sampling: bool,
    polar_fourier: bool,
    code_reg_weight: float,
) -> torch.Tensor:
    if not geometries:
        raise ValueError("No geometries found")

    set_seed(args.seed if branch == "location" else args.seed + 1)
    device = torch.device(args.device)
    geometry_by_id = dict(enumerate(geometries))

    samples = MP_Sampling.MP_sample(
        geometry_by_id,
        args.num_processes,
        samples_perUnit=samples_per_unit,
        point_sample=point_samples,
        sample_band_width=args.sample_band_width,
        uniformed_sample_perUnit=uniform_grid_size,
    )
    full_dataset = Geo2Vec_Dataset(samples, geometry_by_id.keys())
    del samples

    train_size = round(args.sdf_training_ratio * len(full_dataset))
    train_size = min(max(train_size, 1), len(full_dataset) - 1)
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    loader_kwargs = {
        "batch_size": args.sdf_batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
        "collate_fn": identity_collate,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = Geo2Vec_Model(
        n_poly=len(geometries) + 1,
        z_size=args.embedding_dim,
        hidden_size=args.sdf_hidden_size,
        num_freqs=num_freqs,
        weight_decay=0.01,
        log_sampling=log_sampling,
        polar_fourier=polar_fourier,
        num_layers=args.sdf_num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.sdf_learning_rate)
    loss_fn = SDFLoss(code_reg_weight=code_reg_weight, sum=True)

    best_val_loss = float("inf")
    best_embeddings: torch.Tensor | None = None
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for entity_ids, coordinates, distances in train_loader:
            entity_ids = entity_ids.to(device, non_blocking=True)
            coordinates = coordinates.to(device, non_blocking=True)
            distances = distances.to(device, non_blocking=True)
            optimizer.zero_grad()
            predictions = model(entity_ids, coordinates)
            latent_codes = model.poly_embedding_layer(entity_ids)
            loss = loss_fn(predictions, distances, latent_codes)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for entity_ids, coordinates, distances in val_loader:
                entity_ids = entity_ids.to(device, non_blocking=True)
                coordinates = coordinates.to(device, non_blocking=True)
                distances = distances.to(device, non_blocking=True)
                val_loss += torch.nn.functional.l1_loss(
                    model(entity_ids, coordinates), distances, reduction="mean"
                ).item()

        print(
            f"{branch.capitalize()} SDF epoch {epoch}/{epochs}: "
            f"train_sum={train_loss / len(train_loader):.6f}, "
            f"val_sum={val_loss:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_embeddings = (
                model.poly_embedding_layer.weight[: len(geometries)]
                .detach()
                .cpu()
                .clone()
            )

    if best_embeddings is None:
        raise RuntimeError(f"Geo2Vec {branch} training did not produce embeddings")
    return best_embeddings


def learn_location_embeddings(
    geometries: list[Any], args: argparse.Namespace
) -> torch.Tensor:
    prepared = prepare_location_geometries(geometries, args.coordinate_mode)
    return train_sdf_embeddings(
        prepared,
        args,
        branch="location",
        samples_per_unit=args.samples_per_unit,
        point_samples=args.point_samples,
        uniform_grid_size=args.uniform_grid_size,
        epochs=args.sdf_epochs,
        num_freqs=args.sdf_num_freqs,
        log_sampling=False,
        polar_fourier=False,
        code_reg_weight=0.0,
    )


def learn_shape_embeddings(
    geometries: list[Any], args: argparse.Namespace
) -> torch.Tensor:
    # The paper assigns all points the same shape representation because a
    # point has no spatial extent. Use a shared zero vector and train shape
    # SDF latents only for lines and polygons.
    embeddings = torch.zeros((len(geometries), args.embedding_dim))
    non_point_ids = [
        entity_id
        for entity_id, geometry in enumerate(geometries)
        if geometry.geom_type != "Point"
    ]
    if not non_point_ids:
        print("Shape SDF skipped: all entities are points; using the shared zero vector")
        return embeddings

    prepared = [
        poly_preprocess(geometries[entity_id])[0] for entity_id in non_point_ids
    ]
    trained = train_sdf_embeddings(
        prepared,
        args,
        branch="shape",
        samples_per_unit=args.shape_samples_per_unit,
        point_samples=args.shape_point_samples,
        uniform_grid_size=args.shape_uniform_grid_size,
        epochs=args.shape_sdf_epochs,
        num_freqs=args.shape_num_freqs,
        log_sampling=True,
        polar_fourier=args.shape_rotation_invariant,
        code_reg_weight=args.shape_code_reg_weight,
    )
    embeddings[torch.tensor(non_point_ids)] = trained
    return embeddings


class PairClassifier(nn.Module):
    """Classifier architecture corresponding to Geo2Vec's MLP_relationship."""

    def __init__(self, embedding_dim: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, pair_embeddings: torch.Tensor) -> torch.Tensor:
        return self.network(pair_embeddings)


def split_indices(labels: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    indices = np.arange(len(labels))
    y = labels.numpy()
    train, other = train_test_split(
        indices, test_size=0.4, random_state=seed, stratify=y
    )
    val, test = train_test_split(
        other, test_size=0.5, random_state=seed, stratify=y[other]
    )
    return {
        "train": torch.from_numpy(train),
        "val": torch.from_numpy(val),
        "test": torch.from_numpy(test),
    }


@torch.no_grad()
def classifier_metrics(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    logits = model(features[indices].to(device))
    predictions = logits.argmax(dim=1).cpu().numpy()
    truth = labels[indices].numpy()
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "macro_precision": float(
            precision_score(truth, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(truth, predictions, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(truth, predictions).tolist(),
    }


def train_pair_classifier(
    embeddings: torch.Tensor,
    pair_ids: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    set_seed(args.seed + 2)
    device = torch.device(args.device)
    features = torch.cat(
        (embeddings[pair_ids[:, 0]], embeddings[pair_ids[:, 1]]), dim=1
    ).float()
    labels = labels.long()
    splits = split_indices(labels, args.seed)

    train_dataset = TensorDataset(
        features[splits["train"]], labels[splits["train"]]
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.classifier_batch_size,
        shuffle=True,
        generator=generator,
    )

    num_classes = int(labels.max().item()) + 1
    model = PairClassifier(
        embedding_dim=embeddings.shape[1],
        hidden_size=args.classifier_hidden_size,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.classifier_learning_rate, weight_decay=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, args.classifier_epochs + 1):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(
                model(features[splits["val"]].to(device)),
                labels[splits["val"]].to(device),
            ).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            val_accuracy = classifier_metrics(
                model, features, labels, splits["val"], device
            )["accuracy"]
            print(
                f"Classifier epoch {epoch}/{args.classifier_epochs}: "
                f"val_loss={val_loss:.6f}, val_accuracy={val_accuracy:.4f}"
            )
        if stale_epochs >= args.classifier_patience:
            print(f"Classifier stopped early at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Classifier training did not produce a model")
    model.load_state_dict(best_state)
    metrics = {
        split: classifier_metrics(model, features, labels, indices, device)
        for split, indices in splits.items()
    }
    sizes = {split: len(indices) for split, indices in splits.items()}
    return metrics, sizes


def resolve_profile(args: argparse.Namespace) -> None:
    defaults = PROFILE_DEFAULTS[args.profile]
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)


def cache_metadata(
    args: argparse.Namespace,
    data_file: Path,
    selected_indices: torch.Tensor,
) -> dict[str, Any]:
    stat = data_file.stat()
    selected_hash = hashlib.sha256(selected_indices.numpy().tobytes()).hexdigest()
    return {
        "data_file": str(data_file),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "type1": normalized_type(args.type1),
        "type2": normalized_type(args.type2),
        "representation": args.representation,
        "selected_indices_sha256": selected_hash,
        "seed": args.seed,
        "embedding_dim": args.embedding_dim,
        "sdf_hidden_size": args.sdf_hidden_size,
        "sdf_num_freqs": args.sdf_num_freqs,
        "sdf_num_layers": args.sdf_num_layers,
        "sdf_epochs": args.sdf_epochs,
        "sdf_training_ratio": args.sdf_training_ratio,
        "samples_per_unit": args.samples_per_unit,
        "point_samples": args.point_samples,
        "uniform_grid_size": args.uniform_grid_size,
        "sample_band_width": args.sample_band_width,
        "coordinate_mode": args.coordinate_mode,
        "shape_samples_per_unit": args.shape_samples_per_unit,
        "shape_point_samples": args.shape_point_samples,
        "shape_uniform_grid_size": args.shape_uniform_grid_size,
        "shape_sdf_epochs": args.shape_sdf_epochs,
        "shape_num_freqs": args.shape_num_freqs,
        "shape_code_reg_weight": args.shape_code_reg_weight,
        "shape_rotation_invariant": args.shape_rotation_invariant,
    }


def metadata_digest(metadata: dict[str, Any]) -> str:
    payload = json.dumps(metadata, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def torch_load_compat(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    resolve_profile(args)
    set_seed(args.seed)

    data_file = args.data_file.expanduser().resolve()
    raw = torch_load_compat(data_file)
    if not isinstance(raw, (tuple, list)) or len(raw) != 5:
        raise ValueError("Expected a 2D pair dataset containing (X1, X2, L1, L2, Y)")
    x1, x2, l1, l2, labels = raw
    labels = labels.long()

    selected_indices = select_pair_indices(labels, args.max_pairs, args.seed)
    x1 = x1[selected_indices]
    x2 = x2[selected_indices]
    l1 = l1[selected_indices]
    l2 = l2[selected_indices]
    labels = labels[selected_indices]

    metadata = cache_metadata(args, data_file, selected_indices)
    if args.embedding_cache is None:
        cache_dir = ROOT / "results" / "geo2vec_cache"
        args.embedding_cache = (
            cache_dir / f"{data_file.stem}_{metadata_digest(metadata)}.pt"
        )
    cache_path = args.embedding_cache.expanduser().resolve()

    if cache_path.exists() and not args.rebuild_embeddings:
        cached = torch_load_compat(cache_path)
        if cached.get("metadata") != metadata:
            raise ValueError(
                f"Embedding cache metadata does not match this run: {cache_path}. "
                "Use --rebuild-embeddings or a different --embedding-cache."
            )
        embeddings = cached["embeddings"]
        pair_ids = cached["pair_ids"]
        component_dims = cached["component_dims"]
        print(
            f"Loaded {len(embeddings)} cached Geo2Vec embeddings from {cache_path}"
        )
    else:
        geometries, pair_ids = deduplicate_pair_geometries(
            x1,
            x2,
            l1,
            l2,
            normalized_type(args.type1),
            normalized_type(args.type2),
        )
        print(
            f"Selected {len(labels)} pairs containing "
            f"{len(geometries)} unique geometry instances"
        )
        components = []
        component_dims = {}
        if args.representation in ("location", "both"):
            location_embeddings = learn_location_embeddings(geometries, args)
            components.append(location_embeddings)
            component_dims["location"] = location_embeddings.shape[1]
        if args.representation in ("shape", "both"):
            shape_embeddings = learn_shape_embeddings(geometries, args)
            components.append(shape_embeddings)
            component_dims["shape"] = shape_embeddings.shape[1]
        embeddings = torch.cat(components, dim=1)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": metadata,
                "embeddings": embeddings,
                "pair_ids": pair_ids,
                "component_dims": component_dims,
            },
            cache_path,
        )
        print(f"Saved Geo2Vec embeddings to {cache_path}")

    metrics, split_sizes = train_pair_classifier(
        embeddings, pair_ids, labels, args
    )
    result = {
        "method": f"geo2vec_{args.representation}_transductive",
        "task": args.task,
        "data_file": str(data_file),
        "geometry_types": [normalized_type(args.type1), normalized_type(args.type2)],
        "num_pairs": len(labels),
        "num_entities": len(embeddings),
        "entity_embedding_dim": embeddings.shape[1],
        "component_dims": component_dims,
        "num_classes": int(labels.max().item()) + 1,
        "split_sizes": split_sizes,
        "profile": args.profile,
        "coordinate_mode": args.coordinate_mode,
        "embedding_cache": str(cache_path),
        "metrics": metrics,
        "note": (
            "The unsupervised SDF stage sees all unique geometries but no pair labels; "
            "points use a shared zero shape vector; "
            "the downstream classifier uses a stratified 60/20/20 pair split."
        ),
    }

    if args.output is None:
        args.output = (
            ROOT
            / "results"
            / "geo2vec"
            / data_file.stem
            / f"{args.profile}_{args.representation}_metrics.json"
        )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")

    print(json.dumps(metrics["test"], indent=2))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
