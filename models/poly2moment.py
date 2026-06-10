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

    # Number of explicit positional/scale features appended after m00 scaling:
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
        moment_norm = args.moment_params.get('moment_norm', 'm00_scale')
        if moment_norm != 'm00_scale':
            raise ValueError(
                f"Poly2Moment only supports moment_norm='m00_scale', got {moment_norm!r}"
            )

        # Restore positional/scale info as 3 explicit features:
        #   [log(m00), cx = M10/m00, cy = M01/m00]
        # The streams therefore receive d_input + 3 values instead of d_input.
        stream_in = args.d_input + self._N_POS_FEATS

        # Input dimension setup
        if self.fusion == 'learned_fusion' or self.fusion == 'concat':
            d_input = 2 * stream_in
        else:
            d_input = stream_in

        # Final embedding MLP
        self.nn = SimpleMLP(d_input, args.d_hid, args.d_out, args.dropout)

        # Learned fusion parameters for moment streams
        self.param_stream1 = nn.Sequential(
            nn.Linear(stream_in, stream_in, bias=False),
            MLP(stream_in, dropout=args.dropout)
        )
        self.param_stream2 = nn.Sequential(
            nn.Linear(stream_in, stream_in, bias=False),
            MLP(stream_in, dropout=args.dropout)
        )

    def _m00_scale(self, x, m00=None):
        """
        Divide a flat moment vector by |m00| and return scale/position features.

        A caller-provided m00 lets complex Fourier real and imaginary streams
        share the same divisor.
        """
        if m00 is None:
            m00 = x[:, 0:1].abs().clamp(min=1e-6)

        # flat index = m*O + n (row-major, indexing='ij')
        O = int(x.shape[1] ** 0.5)
        cx = x[:, O] / m00.squeeze(1)   # M[1,0] -> index 1*O + 0 = O
        cy = x[:, 1] / m00.squeeze(1)   # M[0,1] -> index 0*O + 1 = 1
        log_m00 = torch.log(m00)
        pos_feats = torch.stack([log_m00.squeeze(1), cx, cy], dim=1)
        return x / m00, pos_feats

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
            x_complex_flat = x_encoded.reshape(B, -1)
            x_real = x_complex_flat.real.to(torch.float32)
            x_imag = x_complex_flat.imag.to(torch.float32)
            m00 = x_complex_flat[:, 0:1].abs().to(torch.float32).clamp(min=1e-6)
            x_flat_real, pos_feats = self._m00_scale(x_real, m00=m00)
            x_flat_imag = x_imag / m00
        else:
            x_flat = x_encoded.reshape(B, -1)
            x_flat_real, pos_feats = self._m00_scale(x_flat)
            x_flat_imag = x_flat_real

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
