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

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.fusion = args.fusion

        # Initialize the moment encoder
        self.moment_encoder = GeometryMomentEncoder(args=args, device=self.device)

        # Get basis type for determining if moments are real or complex
        self.basis_type = args.moment_params.get('basis_type', 'monomial')

        # For monomial and Legendre bases, moments are real-valued
        # We still use a dual-stream architecture for consistency,
        # but we can split the moment tensor differently

        # Input dimension setup
        if self.fusion == 'learned_fusion' or self.fusion == 'concat':
            d_input = 2 * args.d_input
        else:
            d_input = args.d_input

        # Final embedding MLP
        self.nn = SimpleMLP(d_input, args.d_hid, args.d_out, args.dropout)

        # Learned fusion parameters for moment streams
        # For real-valued moments, we can split into positive and negative parts
        # or use even/odd order moments, or simply use two learned transformations
        self.param_stream1 = nn.Sequential(
            nn.Linear(args.d_input, args.d_input, bias=False),
            MLP(args.d_input, dropout=args.dropout)
        )
        self.param_stream2 = nn.Sequential(
            nn.Linear(args.d_input, args.d_input, bias=False),
            MLP(args.d_input, dropout=args.dropout)
        )

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

        # Flatten the moment tensor
        # x_encoded has shape (B, n_m, n_n), flatten to (B, n_m * n_n)
        x_flat = x_encoded.reshape(B, -1)

        # For real-valued moments, we create two streams:
        # Stream 1: Focus on lower-order moments (coarse features)
        # Stream 2: Focus on higher-order moments (fine details)
        # Alternatively, we can use positive and negative parts

        # Simple approach: Apply learned transformations to create two streams
        x_stream1 = self.param_stream1(x_flat)
        x_stream2 = self.param_stream2(x_flat)

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
