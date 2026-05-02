import torch
import torch.nn as nn
from models.moment_encoder import GeometryMomentEncoder


class MLP(nn.Module):
    """MLP for learned fusion of moment features."""
    def __init__(self, d_model, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(d_model, d_model * 2)
        self.w_2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class SimpleMLP(nn.Module):
    """Simple MLP for final embedding generation."""
    def __init__(self, d_input, d_hid, d_out, dropout):
        super(SimpleMLP, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(d_input, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hid, d_out)
        )

    def forward(self, x):
        return self.nn(x)


class Poly2Moment(nn.Module):
    """
    Poly2Moment model: Learns an embedding vector for polymorphic 2D shapes
    using integer-based moment encodings (Monomial or Legendre basis).

    This is an alternative to Poly2Vec that uses moment-based features
    instead of Fourier-based features.
    """

    # Number of explicit positional/scale features appended when moment_norm == 'm00_scale':
    #   [log(m00), cx = M10/m00, cy = M01/m00]
    _N_POS_FEATS = 3

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.fusion = args.fusion

        # Initialize the moment encoder
        self.moment_encoder = GeometryMomentEncoder(args=args, device=self.device)

        # Get basis type for determining if moments are real or complex
        self.basis_type = args.moment_params.get('basis_type', 'monomial')
        # Normalization strategy: "none" | "layer_norm" | "sign_log" | "m00_scale"
        self.moment_norm = args.moment_params.get('moment_norm', 'none')

        # When using m00_scale we restore positional/scale info as 3 explicit features:
        #   [log(m00), cx = M10/m00, cy = M01/m00]
        # The streams therefore receive d_input + 3 values instead of d_input.
        stream_in = args.d_input + self._N_POS_FEATS if self.moment_norm == 'm00_scale' else args.d_input

        # Input dimension setup
        if self.fusion == 'learned_fusion' or self.fusion == 'concat':
            d_input = 2 * stream_in
        else:
            d_input = stream_in

        # Final embedding MLP
        self.nn = SimpleMLP(d_input, args.d_hid, args.d_out, args.dropout)

        # Layer norm (used when moment_norm is "layer_norm" or "layer_norm_m00")
        if self.moment_norm in ('layer_norm', 'layer_norm_m00'):
            self.norm = nn.LayerNorm(args.d_input)

        # Learned fusion parameters for moment streams
        self.param_stream1 = nn.Sequential(
            nn.Linear(stream_in, stream_in, bias=False),
            MLP(stream_in, dropout=args.dropout)
        )
        self.param_stream2 = nn.Sequential(
            nn.Linear(stream_in, stream_in, bias=False),
            MLP(stream_in, dropout=args.dropout)
        )

    def _normalize(self, x):
        """
        Apply the configured normalization strategy to a flat moment vector.
        Strategies:
            none           — pass through unchanged
            layer_norm     — per-sample zero-mean unit-variance (loses scale)
            sign_log       — sign(x) * log1p(|x|), preserves scale, compresses dynamic range
            m00_scale      — divide by m00; also returns [log(m00), cx, cy] as pos_feats
                             to restore positional/scale info stripped by the division
            layer_norm_m00 — layer_norm for shape + raw m00 restored at position 0

        Returns (normalized_x, pos_feats) where pos_feats is (B, 3) for m00_scale, else None.
        """
        if self.moment_norm == 'none':
            return x, None
        elif self.moment_norm == 'layer_norm':
            return self.norm(x), None
        elif self.moment_norm == 'sign_log':
            return torch.sign(x) * torch.log1p(torch.abs(x)), None
        elif self.moment_norm == 'm00_scale':
            m00 = x[:, 0:1].abs().clamp(min=1e-6)
            # flat index = m*O + n (row-major, indexing='ij')
            O = int(x.shape[1] ** 0.5)  # moment grid side length
            cx = x[:, O] / m00.squeeze(1)   # M[1,0] -> index 1*O + 0 = O
            cy = x[:, 1] / m00.squeeze(1)   # M[0,1] -> index 0*O + 1 = 1
            log_m00 = torch.log(m00)         # (B, 1)
            pos_feats = torch.stack([log_m00.squeeze(1), cx, cy], dim=1)  # (B, 3)
            return x / m00, pos_feats
        elif self.moment_norm == 'layer_norm_m00':
            # LayerNorm for shape, but restore the raw (0,0) moment at position 0
            m00 = x[:, 0:1]
            normalized = self.norm(x)
            return torch.cat([m00, normalized[:, 1:]], dim=1), None
        else:
            raise ValueError(f"Unknown moment_norm: {self.moment_norm}")

    def forward(self, x, lengths, dataset_type):
        """
        Forward pass through the Poly2Moment model.

        Args:
            x: Input geometry coordinates
            lengths: Actual lengths for variable-length geometries
            dataset_type: Type of geometry ("points", "lines", "polylines", "polygons")

        Returns:
            x_emb: Fixed-length embedding vector
        """
        B = x.shape[0]

        # Encode geometry using moment basis
        x_encoded = self.moment_encoder.encode(x, lengths, dataset_type=dataset_type)

        # Flatten the moment tensor: (B, O, O) -> (B, O*O)
        if torch.is_complex(x_encoded):
            # The (0,0) Fourier moment is always real-valued (imag part is exactly 0
            # for points, polylines, polygons). Splitting real/imag and then calling
            # _normalize independently with `m00_scale` would divide the imag stream
            # by clamp(0, 1e-6) = 1e-6, blowing it up by ~1e6. Compute the divisor
            # from the complex (0,0) magnitude and apply it once to both streams.
            x_complex_flat = x_encoded.reshape(B, -1)
            x_real = x_complex_flat.real.to(torch.float32)
            x_imag = x_complex_flat.imag.to(torch.float32)
            x_flat_real, pos_feats = self._normalize(x_real)
            x_flat_imag, _ = self._normalize(x_imag)
            # For Fourier m00_scale: imag part of M[0,0] is ~0, so _normalize divides
            # imag by a near-zero clamp. Override: reuse the real-stream m00 divisor.
            if self.moment_norm == 'm00_scale':
                m00_mag = x_real[:, 0:1].abs().clamp(min=1e-6)
                x_flat_imag = x_imag / m00_mag
        else:
            x_flat = x_encoded.reshape(B, -1)
            # sign_log: stream1=log-magnitude, stream2=sign (mirrors mag/phase split)
            if self.moment_norm == 'sign_log':
                x_flat_real = torch.log1p(torch.abs(x_flat))
                x_flat_imag = torch.sign(x_flat)
                pos_feats = None
            else:
                x_flat_real, pos_feats = self._normalize(x_flat)
                x_flat_imag = x_flat_real

        # Append positional/scale features restored from m00_scale normalization
        if pos_feats is not None:
            x_flat_real = torch.cat([x_flat_real, pos_feats], dim=1)
            x_flat_imag = torch.cat([x_flat_imag, pos_feats], dim=1)

        x_stream1 = self.param_stream1(x_flat_real)
        x_stream2 = self.param_stream2(x_flat_imag)

        # Concatenate the two streams
        x_concat = torch.cat([x_stream1, x_stream2], dim=1)

        # Generate final embedding
        x_emb = self.nn(x_concat)

        return x_emb

    def encode(self, x, lengths, dataset_type):
        """
        Encode geometry to embedding (alias for forward).

        Args:
            x: Input geometry coordinates
            lengths: Actual lengths for variable-length geometries
            dataset_type: Type of geometry

        Returns:
            Embedding vector
        """
        return self.forward(x, lengths, dataset_type)
