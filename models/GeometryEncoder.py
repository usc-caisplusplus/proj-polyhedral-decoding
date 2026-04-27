import torch
import torch.nn as nn
from models.poly2vec import Poly2Vec
from models.poly2moment import Poly2Moment
from models.poly2moment_3d import Poly2Moment3D

class GeometryEncoder(nn.Module):
    """
    GeometryEncoder: A wrapper of the whole framework for the spatial reasoning task.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device

        self.encoder_type = args.encoder_type
        if self.encoder_type == "poly2vec":
            self.encoder = Poly2Vec(args=args, device=self.device)
        elif self.encoder_type == "poly2moment":
            self.encoder = Poly2Moment(args=args, device=self.device)
        elif self.encoder_type == "poly2moment_3d":
            self.encoder = Poly2Moment3D(args=args, device=self.device)
        else:
            raise NotImplementedError(f"Encoder {self.encoder_type} not supported")

    def forward(self, x, lengths, dataset_type, face_counts=None):
        if self.encoder_type == "poly2moment_3d":
            x_emb = self.encoder.encode(x, lengths, dataset_type=dataset_type,
                                        face_counts=face_counts)
        else:
            x_emb = self.encoder.encode(x, lengths, dataset_type=dataset_type)

        return x_emb