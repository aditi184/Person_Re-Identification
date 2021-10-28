import torch
import torch.nn as nn

#TODO: Add necessary imports

class DummyModel(nn.Module):
    def __init__(self, batch_size, H, W, D):
        super(DummyModel, self).__init__()
        self.H = H
        self.W = W
        self.D = D
        self.B = batch_size

    def forward(self, x):
        out = torch.rand((self.B, self.D, self.H, self.W))
        return out

# TODO: Define your model architecture
class ReidModel(nn.Module):
    pass


