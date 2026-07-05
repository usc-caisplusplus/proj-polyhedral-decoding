"""
Evaluate Poly2Moment on Geo2Vec's Building and MNIST shape classification tasks.

The source polygons are centered and scaled independently, matching Geo2Vec's
shape preprocessing. Deterministic moment features are computed once and may be
cached; Poly2Moment's learned fusion/projection layers and the downstream MLP
classifier are then trained jointly.

Examples:
    python scripts/evaluate_poly2moment_shape_classification.py --dataset building
    python scripts/evaluate_poly2moment_shape_classification.py --dataset mnist
    python scripts/evaluate_poly2moment_shape_classification.py --dataset both
    python scripts/evaluate_poly2moment_shape_classification.py \
        --dataset building --max-samples-per-class 20 --epochs 2
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.poly2moment import Poly2Moment  # noqa: E402


DATASET_PATHS = {
    "building": ROOT / "Geo2Vec" / "data" / "ShapeClassification.gpkg",
    "mnist": ROOT / "Geo2Vec" / "data" / "mnist_scaled_normalized.gpkg",
}


@dataclass
class ShapeRecords:
    coordinates: list[np.ndarray]
    labels: torch.Tensor
    label_names: list[str]
    source_indices: list[int]
    skipped: int
    polygons_with_holes: int


class CoordinateDataset(Dataset):
    def __init__(self, coordinates: list[np.ndarray], labels: torch.Tensor):
        self.coordinates = coordinates
        self.labels = labels

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> tuple[np.ndarray, torch.Tensor]:
        return self.coordinates[index], self.labels[index]


class Poly2MomentFeatureClassifier(nn.Module):
    """Trainable Poly2Moment head operating on cached deterministic features."""

    def __init__(
        self,
        encoder: Poly2Moment,
        num_classes: int,
        classifier_hidden_size: int,
        classifier_dropout: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.args.d_out, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_hidden_size, num_classes),
        )

    def forward(self, fixed_features: torch.Tensor) -> torch.Tensor:
        if self.encoder.use_multi_basis:
            groups = []
            offset = 0
            for basis in self.encoder.basis_types:
                width = self.encoder._basis_feature_dim(basis)
                group = fixed_features[:, offset : offset + width]
                offset += width
                if self.encoder.feature_fusion == "projected_concat":
                    group = self.encoder.basis_projectors[basis](group)
                groups.append(group)
            embedding = self.encoder.nn(torch.cat(groups, dim=1))
        else:
            stream_width = self.encoder.moment_dim + self.encoder._N_POS_FEATS
            real_features = fixed_features[:, :stream_width]
            imaginary_features = fixed_features[
                :, stream_width : 2 * stream_width
            ]
            real_stream = self.encoder.param_stream1(real_features)
            imaginary_stream = self.encoder.param_stream2(imaginary_features)
            embedding = self.encoder.nn(
                torch.cat((real_stream, imaginary_stream), dim=1)
            )
        return self.classifier(embedding)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Poly2Moment on Geo2Vec's single-polygon Building and "
            "MNIST shape classification datasets."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=("building", "mnist", "both"),
        default="both",
    )
    parser.add_argument(
        "--building-file",
        type=Path,
        default=DATASET_PATHS["building"],
    )
    parser.add_argument(
        "--mnist-file",
        type=Path,
        default=DATASET_PATHS["mnist"],
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        default=None,
        help="Optional stratified subset, useful for smoke tests",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--classifier-batch-size", type=int, default=128)
    parser.add_argument("--classifier-hidden-size", type=int, default=256)
    parser.add_argument("--classifier-dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-order", type=int, default=None)
    basis_group = parser.add_mutually_exclusive_group()
    basis_group.add_argument(
        "--basis-types",
        nargs="+",
        choices=("monomial", "fourier", "legendre"),
        default=None,
        help="Override config with a multi-basis representation",
    )
    basis_group.add_argument(
        "--single-basis",
        choices=("monomial", "fourier", "legendre"),
        default=None,
        help="Override config with one moment basis",
    )
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=ROOT / "results" / "poly2moment_shape_cache",
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "poly2moment_shape_classification.json",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_args(cli: argparse.Namespace) -> SimpleNamespace:
    with cli.config.open("r", encoding="utf-8") as file:
        values = json.load(file)
    moment_params = dict(values["moment_params"])
    if cli.max_order is not None:
        moment_params["max_order"] = cli.max_order
    if cli.basis_types is not None:
        moment_params["basis_types"] = list(cli.basis_types)
        moment_params["basis_type"] = cli.basis_types[0]
    elif cli.single_basis is not None:
        moment_params.pop("basis_types", None)
        moment_params.pop("feature_fusion", None)
        moment_params.pop("projection_dim", None)
        moment_params["basis_type"] = cli.single_basis
    values["moment_params"] = moment_params
    return SimpleNamespace(**values)


def label_sort_key(label: str) -> tuple[int, int | str]:
    try:
        return (0, int(label))
    except ValueError:
        return (1, label)


def normalized_polygon_coordinates(geometry: Any) -> tuple[np.ndarray, int] | None:
    from shapely import affinity
    from shapely.geometry import MultiPolygon, Polygon

    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        geometry = geometry.buffer(0)
    if isinstance(geometry, MultiPolygon):
        geometry = max(geometry.geoms, key=lambda part: part.area)
    if not isinstance(geometry, Polygon) or geometry.is_empty or geometry.area <= 0:
        return None

    min_x, min_y, max_x, max_y = geometry.bounds
    scale = max(max_x - min_x, max_y - min_y)
    if not np.isfinite(scale) or scale <= 0:
        return None
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    geometry = affinity.translate(geometry, xoff=-center_x, yoff=-center_y)
    geometry = affinity.scale(
        geometry,
        xfact=1.0 / scale,
        yfact=1.0 / scale,
        origin=(0.0, 0.0),
    )

    rings = [np.asarray(geometry.exterior.coords[:-1], dtype=np.float32)]
    for interior in geometry.interiors:
        rings.append(np.full((1, 2), np.nan, dtype=np.float32))
        rings.append(np.asarray(interior.coords[:-1], dtype=np.float32))
    coordinates = np.concatenate(rings, axis=0)
    if len(coordinates) < 3:
        return None
    return coordinates, len(geometry.interiors)


def load_shape_records(
    path: Path,
    max_samples_per_class: int | None,
    seed: int,
) -> ShapeRecords:
    try:
        import geopandas as gpd
    except ImportError as error:
        raise RuntimeError(
            "GeoPandas is required. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    if not path.is_file():
        raise FileNotFoundError(f"Shape dataset not found: {path}")
    frame = gpd.read_file(path)
    if "label" not in frame:
        raise ValueError(f"{path} has no 'label' column")

    valid: list[tuple[int, str, np.ndarray, int]] = []
    skipped = 0
    for source_index, row in frame.iterrows():
        result = normalized_polygon_coordinates(row.geometry)
        label = row["label"]
        label_is_missing = label is None or (
            isinstance(label, (float, np.floating)) and np.isnan(label)
        )
        if result is None or label_is_missing:
            skipped += 1
            continue
        coordinates, hole_count = result
        valid.append((int(source_index), str(label), coordinates, hole_count))
    if not valid:
        raise ValueError(f"No valid labeled polygons found in {path}")

    if max_samples_per_class is not None:
        if max_samples_per_class < 10:
            raise ValueError("--max-samples-per-class must be at least 10")
        rng = np.random.default_rng(seed)
        grouped: dict[str, list[int]] = {}
        for position, (_, label, _, _) in enumerate(valid):
            grouped.setdefault(label, []).append(position)
        selected_positions = []
        for positions in grouped.values():
            positions_array = np.asarray(positions)
            if len(positions_array) > max_samples_per_class:
                positions_array = rng.choice(
                    positions_array,
                    size=max_samples_per_class,
                    replace=False,
                )
            selected_positions.extend(positions_array.tolist())
        selected_positions.sort()
        valid = [valid[position] for position in selected_positions]

    label_names = sorted({item[1] for item in valid}, key=label_sort_key)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    return ShapeRecords(
        coordinates=[item[2] for item in valid],
        labels=torch.tensor(
            [label_to_id[item[1]] for item in valid],
            dtype=torch.long,
        ),
        label_names=label_names,
        source_indices=[item[0] for item in valid],
        skipped=skipped,
        polygons_with_holes=sum(item[3] > 0 for item in valid),
    )


def collate_coordinates(
    samples: list[tuple[np.ndarray, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coordinates, labels = zip(*samples)
    lengths = torch.tensor([len(item) for item in coordinates], dtype=torch.long)
    padded = torch.zeros(
        (len(coordinates), int(lengths.max().item()), 2),
        dtype=torch.float32,
    )
    for index, item in enumerate(coordinates):
        padded[index, : len(item)] = torch.from_numpy(item)
    return padded, lengths, torch.stack(labels)


@torch.no_grad()
def fixed_feature_batch(
    encoder: Poly2Moment,
    coordinates: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if encoder.use_multi_basis:
        groups = []
        for basis in encoder.basis_types:
            encoded = encoder.moment_encoders[basis].encode(
                coordinates,
                lengths,
                dataset_type="polygons",
            )
            groups.append(encoder._prepare_basis_features(encoded))
        return torch.cat(groups, dim=1)

    encoded = encoder.moment_encoder.encode(
        coordinates,
        lengths,
        dataset_type="polygons",
    )
    batch_size = encoded.shape[0]
    flat = encoded.reshape(batch_size, -1)
    if torch.is_complex(flat):
        m00 = flat[:, 0:1].abs().to(torch.float32).clamp(min=1e-6)
        real, positional = encoder._m00_scale(
            flat.real.to(torch.float32),
            m00=m00,
        )
        imaginary = flat.imag.to(torch.float32) / m00
    else:
        real, positional = encoder._m00_scale(flat.to(torch.float32))
        imaginary = real
    return torch.cat(
        (
            torch.cat((real, positional), dim=1),
            torch.cat((imaginary, positional), dim=1),
        ),
        dim=1,
    )


def feature_metadata(
    dataset_name: str,
    path: Path,
    records: ShapeRecords,
    model_args: SimpleNamespace,
) -> dict[str, Any]:
    stat = path.stat()
    selected_hash = hashlib.sha256(
        np.asarray(records.source_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    return {
        "format_version": 1,
        "normalization": "per_geometry_bbox_center_max_extent_v1",
        "dataset": dataset_name,
        "data_file": str(path.resolve()),
        "data_size": stat.st_size,
        "data_mtime_ns": stat.st_mtime_ns,
        "selected_indices_sha256": selected_hash,
        "labels": records.label_names,
        "moment_params": model_args.moment_params,
        "fusion": model_args.fusion,
    }


def cache_path(cache_dir: Path, metadata: dict[str, Any]) -> Path:
    serialized = json.dumps(metadata, sort_keys=True).encode("utf-8")
    fingerprint = hashlib.sha256(serialized).hexdigest()[:16]
    return cache_dir / f"{metadata['dataset']}_{fingerprint}.pt"


def load_or_compute_features(
    dataset_name: str,
    path: Path,
    records: ShapeRecords,
    encoder: Poly2Moment,
    model_args: SimpleNamespace,
    cli: argparse.Namespace,
) -> torch.Tensor:
    metadata = feature_metadata(dataset_name, path, records, model_args)
    destination = cache_path(cli.feature_cache_dir, metadata)
    if destination.is_file() and not cli.rebuild_features:
        payload = torch.load(destination, map_location="cpu")
        if payload.get("metadata") == metadata:
            features = payload["features"].float()
            if len(features) != len(records.labels):
                raise RuntimeError(f"Invalid feature cache length in {destination}")
            print(f"Loaded {len(features)} cached feature rows from {destination}")
            return features

    loader = DataLoader(
        CoordinateDataset(records.coordinates, records.labels),
        batch_size=cli.feature_batch_size,
        shuffle=False,
        collate_fn=collate_coordinates,
    )
    batches = []
    encoder.eval()
    for batch_index, (coordinates, lengths, _) in enumerate(loader, start=1):
        batches.append(
            fixed_feature_batch(encoder, coordinates, lengths).cpu().float()
        )
        if batch_index == 1 or batch_index % 50 == 0:
            encoded_count = min(
                batch_index * cli.feature_batch_size,
                len(records.labels),
            )
            print(
                f"{dataset_name}: encoded fixed moment features "
                f"{encoded_count}/{len(records.labels)}"
            )
    features = torch.cat(batches, dim=0)
    if not torch.isfinite(features).all():
        raise RuntimeError(f"{dataset_name} produced NaN or infinite moment features")

    cli.feature_cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "features": features}, destination)
    print(f"Saved fixed moment features to {destination}")
    return features


def split_indices(labels: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    indices = np.arange(len(labels))
    targets = labels.numpy()
    train, other = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=targets,
    )
    validation, test = train_test_split(
        other,
        test_size=0.50,
        random_state=seed,
        stratify=targets[other],
    )
    return {
        "train": torch.from_numpy(train),
        "validation": torch.from_numpy(validation),
        "test": torch.from_numpy(test),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(
        TensorDataset(features[indices], labels[indices]),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    predictions = []
    truth = []
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(batch_features)
        loss_sum += criterion(logits, batch_labels).item()
        predictions.append(logits.argmax(dim=1).cpu())
        truth.append(batch_labels.cpu())
    predicted = torch.cat(predictions).numpy()
    actual = torch.cat(truth).numpy()
    return {
        "loss": loss_sum / len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro")),
        "macro_precision": float(
            precision_score(actual, predicted, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(actual, predicted, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
    }


def train_classifier(
    encoder: Poly2Moment,
    features: torch.Tensor,
    labels: torch.Tensor,
    label_names: list[str],
    cli: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, int], int]:
    set_seed(cli.seed + 1)
    splits = split_indices(labels, cli.seed)
    train_loader = DataLoader(
        TensorDataset(features[splits["train"]], labels[splits["train"]]),
        batch_size=cli.classifier_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cli.seed),
    )
    model = Poly2MomentFeatureClassifier(
        encoder=encoder,
        num_classes=len(label_names),
        classifier_hidden_size=cli.classifier_hidden_size,
        classifier_dropout=cli.classifier_dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cli.learning_rate,
        weight_decay=cli.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    epochs_run = 0
    for epoch in range(1, cli.epochs + 1):
        epochs_run = epoch
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()

        validation = evaluate(
            model,
            features,
            labels,
            splits["validation"],
            device,
            cli.classifier_batch_size,
        )
        if validation["loss"] < best_validation_loss:
            best_validation_loss = validation["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch {epoch}/{cli.epochs}: "
                f"validation_loss={validation['loss']:.6f}, "
                f"validation_accuracy={validation['accuracy']:.4f}"
            )
        if stale_epochs >= cli.patience:
            print(f"Stopped early at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a model")
    model.load_state_dict(best_state)
    metrics = {
        split: evaluate(
            model,
            features,
            labels,
            indices,
            device,
            cli.classifier_batch_size,
        )
        for split, indices in splits.items()
    }
    sizes = {split: len(indices) for split, indices in splits.items()}
    return metrics, sizes, epochs_run


def selected_datasets(cli: argparse.Namespace) -> list[tuple[str, Path]]:
    paths = {
        "building": cli.building_file,
        "mnist": cli.mnist_file,
    }
    names = ("building", "mnist") if cli.dataset == "both" else (cli.dataset,)
    return [(name, paths[name]) for name in names]


def main() -> None:
    cli = parse_args()
    if cli.epochs <= 0 or cli.patience <= 0:
        raise ValueError("--epochs and --patience must be positive")
    if cli.feature_batch_size <= 0 or cli.classifier_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")

    set_seed(cli.seed)
    device = resolve_device(cli.device)
    model_args = load_model_args(cli)
    print(f"Device: {device}")
    print(f"Moment parameters: {model_args.moment_params}")

    results: dict[str, Any] = {
        "method": "poly2moment_shape_classification",
        "seed": cli.seed,
        "device": str(device),
        "config": str(cli.config.resolve()),
        "model": {
            "fusion": model_args.fusion,
            "d_hid": model_args.d_hid,
            "d_out": model_args.d_out,
            "dropout": model_args.dropout,
            "moment_params": model_args.moment_params,
        },
        "training": {
            "epochs": cli.epochs,
            "patience": cli.patience,
            "learning_rate": cli.learning_rate,
            "weight_decay": cli.weight_decay,
            "classifier_hidden_size": cli.classifier_hidden_size,
            "classifier_dropout": cli.classifier_dropout,
        },
        "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "datasets": {},
    }

    for dataset_offset, (dataset_name, path) in enumerate(selected_datasets(cli)):
        print(f"\nLoading {dataset_name} from {path}")
        records = load_shape_records(
            path,
            max_samples_per_class=cli.max_samples_per_class,
            seed=cli.seed,
        )
        counts = torch.bincount(
            records.labels,
            minlength=len(records.label_names),
        ).tolist()
        print(
            f"{dataset_name}: {len(records.labels)} polygons, "
            f"{len(records.label_names)} classes, "
            f"{records.polygons_with_holes} polygons with holes"
        )

        set_seed(cli.seed + dataset_offset)
        encoder = Poly2Moment(args=model_args, device=device).to(device)
        features = load_or_compute_features(
            dataset_name,
            path,
            records,
            encoder,
            model_args,
            cli,
        )
        metrics, split_sizes, epochs_run = train_classifier(
            encoder,
            features,
            records.labels,
            records.label_names,
            cli,
            device,
        )
        results["datasets"][dataset_name] = {
            "data_file": str(path.resolve()),
            "num_samples": len(records.labels),
            "num_classes": len(records.label_names),
            "class_names": records.label_names,
            "class_counts": dict(zip(records.label_names, counts)),
            "skipped_geometries": records.skipped,
            "polygons_with_holes": records.polygons_with_holes,
            "fixed_feature_dim": features.shape[1],
            "split_sizes": split_sizes,
            "epochs_run": epochs_run,
            "metrics": metrics,
        }
        print(
            f"{dataset_name} test: "
            f"accuracy={metrics['test']['accuracy']:.4f}, "
            f"macro_f1={metrics['test']['macro_f1']:.4f}"
        )

    cli.output.parent.mkdir(parents=True, exist_ok=True)
    with cli.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, sort_keys=True)
    print(f"\nSaved results to {cli.output}")


if __name__ == "__main__":
    main()
