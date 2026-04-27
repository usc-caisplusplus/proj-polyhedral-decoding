import torch
import torch.nn as nn
from models.moment_encoder_3d import GeometryMomentEncoder3D


class MLP(nn.Module):
    """MLP for learned fusion of moment features."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_model * 2)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class SimpleMLP(nn.Module):
    """Simple MLP for final embedding generation."""
    def __init__(self, d_input, d_hid, d_out, dropout):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(d_input, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out)
        )

    def forward(self, x):
        return self.nn(x)


class Poly2Moment3D(nn.Module):
    """
    Poly2Moment3D: 3D analogue of Poly2Moment.

    Encodes 3D geometries (points, polylines, flat polygons, polyhedra) using
    integer-based moment encodings over a 3D (O x O x O) grid and learns a
    fixed-length embedding via two parallel MLP streams followed by a final MLP.

    The model mirrors Poly2Moment exactly, but the flat moment dimension is
    O^3 instead of O^2 (where O = max_order + 1).

    Supported dataset_type values (forwarded to GeometryMomentEncoder3D):
        "points", "polylines", "polygons", "polyhedra"
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.fusion = args.fusion

        self.moment_encoder = GeometryMomentEncoder3D(args=args, device=self.device)

        self.basis_type = args.moment_params.get('basis_type', 'legendre')
        self.moment_norm = args.moment_params.get('moment_norm', 'none')

        # d_input must equal O^3 when using 3D moments
        # The caller is responsible for setting args.d_input correctly.
        if self.fusion == 'learned_fusion' or self.fusion == 'concat':
            d_input = 2 * args.d_input
        else:
            d_input = args.d_input

        self.nn = SimpleMLP(d_input, args.d_hid, args.d_out, args.dropout)

        if self.moment_norm in ('layer_norm', 'layer_norm_m00'):
            self.norm = nn.LayerNorm(args.d_input)

        self.param_stream1 = nn.Sequential(
            nn.Linear(args.d_input, args.d_input, bias=False),
            MLP(args.d_input, dropout=args.dropout)
        )
        self.param_stream2 = nn.Sequential(
            nn.Linear(args.d_input, args.d_input, bias=False),
            MLP(args.d_input, dropout=args.dropout)
        )

    def _normalize(self, x):
        """
        Apply the configured normalization strategy to a flat moment vector.
        Strategies match Poly2Moment:
            none           — pass through unchanged
            layer_norm     — per-sample zero-mean unit-variance
            sign_log       — sign(x)*log1p(|x|)
            m00_scale      — divide by absolute value of the (0,0,0) moment
            layer_norm_m00 — layer_norm + raw (0,0,0) moment restored at index 0
        """
        if self.moment_norm == 'none':
            return x
        elif self.moment_norm == 'layer_norm':
            return self.norm(x)
        elif self.moment_norm == 'sign_log':
            return torch.sign(x) * torch.log1p(torch.abs(x))
        elif self.moment_norm == 'm00_scale':
            m000 = x[:, 0:1].abs().clamp(min=1e-6)
            return x / m000
        elif self.moment_norm == 'layer_norm_m00':
            m000 = x[:, 0:1]
            normalized = self.norm(x)
            return torch.cat([m000, normalized[:, 1:]], dim=1)
        else:
            raise ValueError(f"Unknown moment_norm: {self.moment_norm}")

    def forward(self, x, lengths, dataset_type, face_counts=None):
        """
        Forward pass through Poly2Moment3D.

        Args:
            x:            (B, ..., 3) geometry coordinate tensor
            lengths:      (B,) actual vertex/point counts
            dataset_type: "points" | "polylines" | "polygons" | "polyhedra"
            face_counts:  (B,) triangular face counts — required for polyhedra

        Returns:
            x_emb: (B, d_out) embedding tensor
        """
        B = x.shape[0]

        x_encoded = self.moment_encoder.encode(x, lengths, dataset_type=dataset_type,
                                               face_counts=face_counts)

        # Flatten (B, O, O, O) -> (B, O^3)
        if torch.is_complex(x_encoded):
            x_real = x_encoded.real.reshape(B, -1)
            x_imag = x_encoded.imag.reshape(B, -1)
            x_flat_real = self._normalize(x_real)
            x_flat_imag = self._normalize(x_imag)
        else:
            x_flat = x_encoded.reshape(B, -1)
            if self.moment_norm == 'sign_log':
                x_flat_real = torch.log1p(torch.abs(x_flat))
                x_flat_imag = torch.sign(x_flat)
            else:
                x_flat_norm = self._normalize(x_flat)
                x_flat_real = x_flat_norm
                x_flat_imag = x_flat_norm

        x_stream1 = self.param_stream1(x_flat_real)
        x_stream2 = self.param_stream2(x_flat_imag)

        x_concat = torch.cat([x_stream1, x_stream2], dim=1)
        x_emb = self.nn(x_concat)

        return x_emb

    def encode(self, x, lengths, dataset_type, face_counts=None):
        return self.forward(x, lengths, dataset_type, face_counts=face_counts)
