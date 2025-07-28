import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from src.layers import GRU_Layer

class Model(nn.Module):
    def __init__(self, num_tokens, hSize):
        super().__init__()
        self.num_tokens = num_tokens
        self.gru = GRU_Layer(num_tokens, hSize=hSize, nout=num_tokens)
        # Embeddings
        self.embeddings = nn.Embedding(num_tokens, num_tokens)


    def forward(self, X, train=True):
        X = X.long()
        emb = self.embeddings(X)
        logits = self.gru.forward(emb, train=train)
        
        return logits