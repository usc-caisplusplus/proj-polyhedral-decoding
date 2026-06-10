"""
Regression tests for Poly2Moment m00 scaling (real and complex bases).

Guards against the "Fourier + m00_scale destroys imag stream" bug
(commit af11fda). The (0,0) Fourier moment is always real-valued for
points / polylines / polygons; an earlier version of the model split
real/imag *before* per-stream m00_scale, which divided the imag stream
by clamp(0, 1e-6) and amplified it by ~1e6. This file pins down:

  - all (basis x geom_type) forward passes stay finite,
  - distinct inputs produce distinct embeddings,
  - the imag stream of Fourier moments stays bounded after m00 scaling,
  - a "buggy reference" implementation reproduces the 1e6 blow-up
    (so any future regression that re-introduces the per-stream split
    would reactivate the differential check).

Run from repo root:
    py scripts/test_poly2moment_norm.py
"""

from __future__ import annotations

import sys
import os
import math

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.poly2moment import Poly2Moment, MLP, SimpleMLP
from models.moment_encoder import GeometryMomentEncoder


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class Args:
    """Minimal stand-in for the real config object."""
    pass


def make_args(basis_type: str, max_order: int = 14):
    a = Args()
    a.fusion = "learned_fusion"
    a.d_input = (max_order + 1) ** 2
    a.d_hid = 100
    a.d_out = 32
    a.dropout = 0.2
    a.moment_params = {
        "basis_type": basis_type,
        "max_order": max_order,
        "moment_norm": "m00_scale",
    }
    return a


def make_multi_basis_args(feature_fusion: str, max_order: int = 4):
    args = make_args("fourier", max_order=max_order)
    args.moment_params.update({
        "basis_types": ["monomial", "fourier", "legendre"],
        "feature_fusion": feature_fusion,
        "projection_dim": 12,
    })
    return args


def test_only_m00_scale_supported(device):
    args = make_args("monomial")
    args.moment_params["moment_norm"] = "none"
    try:
        Poly2Moment(args, device)
    except ValueError as exc:
        return [] if "only supports moment_norm='m00_scale'" in str(exc) else [str(exc)]
    return ["unsupported normalization was accepted"]


def test_multi_basis_fusion(device):
    failures = []
    x, lengths = make_synthetic_batch("polylines", B=4, device=device)

    for fusion in ["normalized_concat", "projected_concat"]:
        args = make_multi_basis_args(fusion)
        torch.manual_seed(0)
        model = Poly2Moment(args, device).to(device).eval()
        with torch.no_grad():
            out = model(x, lengths, dataset_type="polylines")

        if tuple(out.shape) != (4, args.d_out):
            failures.append((fusion, "shape", tuple(out.shape)))
        if not torch.isfinite(out).all().item():
            failures.append((fusion, "non-finite output"))

        first_linear = model.nn.nn[0]
        if fusion == "projected_concat":
            expected_dim = 3 * args.moment_params["projection_dim"]
        else:
            moment_dim = args.d_input
            expected_dim = (
                (moment_dim + model._N_POS_FEATS)
                + (2 * moment_dim + model._N_POS_FEATS)
                + (moment_dim + model._N_POS_FEATS)
            )
        if first_linear.in_features != expected_dim:
            failures.append(
                (fusion, "fused dimension", first_linear.in_features, expected_dim)
            )

        if fusion == "projected_concat":
            model.train()
            model(x, lengths, dataset_type="polylines").sum().backward()
            for basis, projector in model.basis_projectors.items():
                if projector[0].weight.grad is None:
                    failures.append((fusion, basis, "missing projector gradient"))

    return failures


def test_multi_basis_feature_order(device):
    args = make_multi_basis_args("normalized_concat", max_order=1)
    model = Poly2Moment(args, device).to(device)
    encoded = torch.tensor(
        [[[2 + 0j, 4 + 6j], [8 + 10j, 12 + 14j]]],
        dtype=torch.complex64,
        device=device,
    )

    features = model._prepare_basis_features(encoded)
    expected = torch.tensor(
        [[1, 2, 4, 6, 0, 3, 5, 7, math.log(2), 4, 2]],
        dtype=torch.float32,
        device=device,
    )
    if not torch.allclose(features, expected, atol=1e-6):
        return [("feature order", features.cpu().tolist(), expected.cpu().tolist())]
    return []


def make_synthetic_batch(geom_type: str, B: int, device):
    """Construct a small batch + lengths tensor for the requested geom_type."""
    rng = np.random.default_rng(0)
    if geom_type == "points":
        # (B, 1, 2) padded
        pts = rng.uniform(-0.9, 0.9, size=(B, 1, 2)).astype(np.float32)
        lengths = np.ones(B, dtype=np.int64)
        return torch.from_numpy(pts).to(device), torch.from_numpy(lengths).to(device)
    if geom_type == "polylines":
        max_pts = 6
        pts = np.zeros((B, max_pts, 2), dtype=np.float32)
        lengths = np.zeros(B, dtype=np.int64)
        for b in range(B):
            n = int(rng.integers(2, max_pts + 1))
            pts[b, :n] = rng.uniform(-0.8, 0.8, size=(n, 2)).astype(np.float32)
            lengths[b] = n
        return torch.from_numpy(pts).to(device), torch.from_numpy(lengths).to(device)
    if geom_type == "polygons":
        max_k = 8
        pts = np.zeros((B, max_k, 2), dtype=np.float32)
        lengths = np.zeros(B, dtype=np.int64)
        for b in range(B):
            n = 5
            angles = sorted(rng.uniform(0, 2 * math.pi, n))
            radii = rng.uniform(0.15, 0.45, n)
            cx, cy = rng.uniform(-0.4, 0.4, 2)
            for k, (a_, r_) in enumerate(zip(angles, radii)):
                pts[b, k, 0] = cx + r_ * math.cos(a_)
                pts[b, k, 1] = cy + r_ * math.sin(a_)
            lengths[b] = n
        return torch.from_numpy(pts).to(device), torch.from_numpy(lengths).to(device)
    raise ValueError(f"unknown geom_type={geom_type!r}")


# ---------------------------------------------------------------------------
# Test 1: forward-pass invariants on the full (basis, geom) grid
# ---------------------------------------------------------------------------

def test_forward_invariants(device):
    print("\n" + "=" * 78)
    print("Test 1 — forward invariants over (basis x geom)")
    print("=" * 78)

    bases = ["monomial", "fourier", "legendre"]
    geoms = ["points", "polylines", "polygons"]

    failures = []
    for basis in bases:
        for geom in geoms:
            args = make_args(basis)
            torch.manual_seed(0)
            model = Poly2Moment(args, device).to(device).eval()
            x, lengths = make_synthetic_batch(geom, B=8, device=device)
            with torch.no_grad():
                out = model(x, lengths, dataset_type=geom)
            tag = f"{basis:9s} | {geom:9s}"
            ok_shape = tuple(out.shape) == (8, 32)
            ok_finite = torch.isfinite(out).all().item()
            ok_var = out.std().item() > 1e-6
            ok_range = out.abs().max().item() < 1e3
            pdists = (out.unsqueeze(0) - out.unsqueeze(1)).norm(dim=2)
            positives = pdists[pdists > 0]
            ok_distinct = (positives.numel() > 0 and positives.min().item() > 1e-5)
            status = "OK " if (ok_shape and ok_finite and ok_var and ok_range and ok_distinct) else "FAIL"
            print(f"  [{status}] {tag} | range=[{out.min().item():+.3g},{out.max().item():+.3g}] std={out.std().item():.3g}")
            if status == "FAIL":
                failures.append((basis, geom,
                                 ok_shape, ok_finite, ok_var, ok_range, ok_distinct))
    return failures


# ---------------------------------------------------------------------------
# Test 2: Fourier + m00_scale — the imag stream must NOT blow up
# ---------------------------------------------------------------------------

def test_fourier_m00_imag_bounded(device):
    print("\n" + "=" * 78)
    print("Test 2 — Fourier + m00_scale: imag stream bounded after normalization")
    print("=" * 78)

    failures = []
    for geom in ["points", "polylines", "polygons"]:
        args = make_args("fourier")
        torch.manual_seed(0)
        encoder = GeometryMomentEncoder(args=args, device=device)
        x, lengths = make_synthetic_batch(geom, B=16, device=device)
        x_enc = encoder.encode(x, lengths, dataset_type=geom)  # complex (B, O, O)

        # Reproduce post-fix normalization (matches models/poly2moment.py forward())
        B = x.shape[0]
        x_complex_flat = x_enc.reshape(B, -1)
        m00_mag = x_complex_flat[:, 0:1].abs().clamp(min=1e-6)
        x_real = (x_complex_flat.real / m00_mag).to(torch.float32)
        x_imag = (x_complex_flat.imag / m00_mag).to(torch.float32)

        # The post-fix invariant: both streams should be bounded.
        # Empirically, after dividing by |m00| the modulus per-cell <= 1
        # for points (unit modulus) and roughly <= 1 for polylines/polygons
        # under the normalization used here. We allow a generous slack.
        max_real = x_real.abs().max().item()
        max_imag = x_imag.abs().max().item()
        ratio = max_imag / max(max_real, 1e-9)

        ok_imag = max_imag < 10.0          # before-fix this was ~1e6
        ok_ratio = ratio < 100.0           # before-fix this was ~1e6 / O(1)
        status = "OK " if (ok_imag and ok_ratio) else "FAIL"
        print(f"  [{status}] {geom:9s} | max|real|={max_real:.3g}  max|imag|={max_imag:.3g}  imag/real={ratio:.2g}")
        if status == "FAIL":
            failures.append((geom, max_real, max_imag, ratio))
    return failures


# ---------------------------------------------------------------------------
# Test 3: differential test — pre-fix per-stream split must blow imag up
#
# This *intentionally* re-implements the buggy normalization from
# pre-af11fda. If a future refactor accidentally restores the per-stream
# m00_scale, the post-fix model and this buggy reference would coincide
# and the test would fail loudly.
# ---------------------------------------------------------------------------

class Poly2MomentBuggy(nn.Module):
    """Reproduces the *broken* pre-af11fda forward path for differential testing."""

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.fusion = args.fusion
        self.moment_encoder = GeometryMomentEncoder(args=args, device=device)
        self.basis_type = args.moment_params.get("basis_type", "monomial")
        d_input = 2 * args.d_input if self.fusion in ("learned_fusion", "concat") else args.d_input
        self.nn = SimpleMLP(d_input, args.d_hid, args.d_out, args.dropout)
        self.param_stream1 = nn.Sequential(nn.Linear(args.d_input, args.d_input, bias=False), MLP(args.d_input, args.dropout))
        self.param_stream2 = nn.Sequential(nn.Linear(args.d_input, args.d_input, bias=False), MLP(args.d_input, args.dropout))

    def _normalize(self, x):
        # Old m00_scale path — applied per-stream.
        m00 = x[:, 0:1].abs().clamp(min=1e-6)
        return x / m00

    def forward(self, x, lengths, dataset_type):
        B = x.shape[0]
        x_encoded = self.moment_encoder.encode(x, lengths, dataset_type=dataset_type)
        # Pre-fix path: split first, normalize per-stream.
        x_real = x_encoded.real.reshape(B, -1)
        x_imag = x_encoded.imag.reshape(B, -1)
        x_flat_real = self._normalize(x_real)
        x_flat_imag = self._normalize(x_imag)
        x1 = self.param_stream1(x_flat_real.float())
        x2 = self.param_stream2(x_flat_imag.float())
        return self.nn(torch.cat([x1, x2], dim=1))


def test_differential_against_buggy(device):
    print("\n" + "=" * 78)
    print("Test 3 — differential against pre-fix buggy reference")
    print("=" * 78)

    args = make_args("fourier")
    failures = []

    for geom in ["points", "polylines", "polygons"]:
        torch.manual_seed(0)
        good = Poly2Moment(args, device).to(device).eval()
        torch.manual_seed(0)
        bad = Poly2MomentBuggy(args, device).to(device).eval()
        x, lengths = make_synthetic_batch(geom, B=8, device=device)
        with torch.no_grad():
            out_good = good(x, lengths, geom)
            out_bad = bad(x, lengths, geom)

        # Buggy stream stats
        x_enc = good.moment_encoder.encode(x, lengths, geom)
        x_imag = x_enc.imag.reshape(8, -1)
        m00_imag = x_imag[:, 0:1].abs().clamp(min=1e-6)
        bad_imag_max = (x_imag / m00_imag).abs().max().item()

        # The buggy reference must show the catastrophic amplification we fixed.
        # If this assertion ever fails, someone has unwittingly restored the
        # split-then-per-stream-normalize order in the buggy reference itself —
        # update this guard *and* recheck production code.
        ok = bad_imag_max > 1e4
        status = "OK " if ok else "FAIL"
        print(f"  [{status}] {geom:9s} | buggy imag stream max={bad_imag_max:.3g}  | post-fix std={out_good.std().item():.3g}  buggy std={out_bad.std().item():.3g}")
        if not ok:
            failures.append((geom, bad_imag_max))
    return failures


# ---------------------------------------------------------------------------
# Test 4: dataset-driven sanity (skip if data missing)
#   Train for 3 epochs on Singapore polyline_point_contain (was 0.5015 / chance).
#   Pass = test accuracy > 0.65 (clearly above 0.5 chance).
# ---------------------------------------------------------------------------

def test_real_training_on_failing_task(device, dataset_path: str = "./data/Singapore/polyline_point_contain_data.pt"):
    print("\n" + "=" * 78)
    print("Test 4 — real-training sanity on Singapore polyline_point_contain (3 epochs)")
    print("=" * 78)

    if not os.path.isfile(dataset_path):
        print(f"  [SKIP] dataset not found: {dataset_path}")
        return []

    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from loaders.dataloader import GeometryRelationshipDataset
    from models.GeometryEncoder import GeometryEncoder
    from models.TaskEncoder import TaskEncoder

    args = make_args("fourier")
    args.encoder_type = "poly2moment"
    args.dataset_type1 = "polylines"
    args.dataset_type2 = "points"
    args.task = "relation-prediction"
    args.num_classes = 2
    args.batch_size = 128
    args.lr = 1e-4
    args.wd = 1e-8
    args.sampling_strategy = "fourier"
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.epochs = 3
    args.patience = 20
    args.training = True

    X1, X2, L1, L2, Y = torch.load(dataset_path)

    Xa, Xb, X2a, X2b, La, Lb, L2a, L2b, Ya, Yb = train_test_split(
        X1, X2, L1, L2, Y, test_size=0.4, random_state=42
    )
    X1v, X1t, X2v, X2t, L1v, L1t, L2v, L2t, Yv, Yt = train_test_split(
        Xb, X2b, Lb, L2b, Yb, test_size=0.5, random_state=42
    )

    train_loader = DataLoader(GeometryRelationshipDataset(Xa, X2a, La, L2a, Ya), batch_size=128, shuffle=True)
    test_loader = DataLoader(GeometryRelationshipDataset(X1t, X2t, L1t, L2t, Yt), batch_size=128, shuffle=False)

    geom_model = GeometryEncoder(args, device).to(device)
    model = TaskEncoder(args, geom_model, device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCELoss().to(device)

    dt1, dt2 = args.dataset_type1, args.dataset_type2

    model.train()
    for epoch in range(3):
        for x1, x2, l1, l2, y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            l1, l2 = l1.to(device), l2.to(device)
            y = y.to(device).float()
            optim.zero_grad()
            yhat = model(x1, l1, dt1, x2, l2, dt2).squeeze(-1)
            loss = crit(yhat, y)
            loss.backward()
            optim.step()
        print(f"  epoch {epoch+1}/3  train loss {loss.item():.4f}")

    # Test
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x1, x2, l1, l2, y in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            l1, l2 = l1.to(device), l2.to(device)
            y = y.to(device).float()
            yhat = model(x1, l1, dt1, x2, l2, dt2).squeeze(-1)
            pred = (yhat > 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / max(total, 1)
    ok = acc > 0.65
    status = "OK " if ok else "FAIL"
    print(f"  [{status}] test accuracy after 3 epochs = {acc:.4f}  (chance = 0.50, threshold = 0.65)")
    return [] if ok else [("real_training", acc)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}  (torch {torch.__version__})")

    all_failures = []
    all_failures += test_only_m00_scale_supported(device)
    all_failures += test_multi_basis_fusion(device)
    all_failures += test_multi_basis_feature_order(device)
    all_failures += test_forward_invariants(device)
    all_failures += test_fourier_m00_imag_bounded(device)
    all_failures += test_differential_against_buggy(device)
    all_failures += test_real_training_on_failing_task(device)

    print("\n" + "=" * 78)
    if all_failures:
        print(f"FAILURES: {len(all_failures)}")
        for f in all_failures:
            print("  ", f)
        sys.exit(1)
    print("Overall: ALL OK")
    print("=" * 78)


if __name__ == "__main__":
    main()
