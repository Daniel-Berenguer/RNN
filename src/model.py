import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GRU_Layer

class Model(nn.Module):
    def __init__(self, num_chars, hSize):
        super().__init__()
        self.num_chars = num_chars
        self.gru = GRU_Layer(num_chars, hSize=hSize, nout=num_chars)


    def forward(self, X, train=True):
        # One hot encode batch
        X = X.long()
        X = F.one_hot(X, self.num_chars)
        X = X.float()

        logits = self.gru.forward(X, train=train)
        
        return logits