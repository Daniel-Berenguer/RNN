import torch
import math
import torch.nn as nn

class BatchNorm(nn.Module):
    def __init__(self, nout):
        super().__init__()
        self.a = "a"
        # Parameters
        self.gain = nn.Parameter(torch.ones(nout))
        self.bias = nn.Parameter(torch.ones(nout))

        # Initialize running mean std
        self.runningMean = nn.Parameter(torch.zeros(nout), requires_grad=False)
        self.runningStd = nn.Parameter(torch.ones(nout), requires_grad=False)

    def forward(self, X, train=True):
        if train:
            batchMean = X.mean(dim=0)
            batchStd = X.std(dim=0)
            out = self.gain * ((X - batchMean) / (batchStd + 1e-5)) + self.bias

            with torch.no_grad():
                # Update running mean and std
                self.runningMean *= 0.99
                self.runningMean += 0.01 * batchMean
                self.runningStd *= 0.99
                self.runningStd += 0.01 * batchStd

        else:
            with torch.no_grad():
                out = self.gain * ((X - self.runningMean) / (self.runningStd + 1e-5)) + self.bias

        return out



class GRU_Layer(nn.Module):
    def __init__(self, nin, hSize, nout, bN=True):
        super().__init__()
        self.hSize = hSize
        self.bN = bN

        # Parameter Initialization
        # Next State Calc
        if not bN:
            self.b = nn.Parameter(torch.zeros(hSize)) # Bias
        else:
            self.bN_h = BatchNorm(hSize)
        self.U = nn.Parameter(torch.randn(nin, hSize) * 5/3 * 1/math.sqrt(nin)) # Input multiplier
        self.W = nn.Parameter(torch.randn(hSize, hSize)) # Prev HState multiplier

        # Update Gate Calc
        if not bN:
            self.bu = nn.Parameter(torch.zeros(hSize)) # Bias
        else:
            self.bN_u = BatchNorm(hSize)

        self.Uu = nn.Parameter(torch.randn(nin, hSize) * math.sqrt(1 / nin)) # Input multiplier
        self.Wu = nn.Parameter(torch.randn(hSize, hSize)) # Prev HState multiplier

        # Reset Gate Calc
        if not bN:
            self.br = nn.Parameter(torch.zeros(hSize)) # Bias
        else:
            self.bN_r = BatchNorm(hSize)

        self.Ur = nn.Parameter(torch.randn(nin, hSize) * math.sqrt(1 / nin)) # Input multiplier
        self.Wr = nn.Parameter(torch.randn(hSize, hSize)) # Prev HState multiplier

        # Output Calc
        self.V = nn.Parameter(torch.randn(hSize, nout))
        self.bo = nn.Parameter(torch.zeros(nout))
        

    def forward(self, X, train=True, show=False):
        # X are the sequences
        # X shape is (batch_sz, sequence_len, input_size)

        # Hidden State Initialization
        h = torch.zeros(self.hSize)

        for t in range(X.shape[1]):
            X_t = X[:, t, :] # t'th x elements of sequences
            
            # Calculate gates
            preU =  X_t @ self.Uu + h @ self.Wu
            preR =  X_t @ self.Ur + h @ self.Wr
            if self.bN:
                preU = self.bN_u.forward(preU, train=train)
                preR = self.bN_r.forward(preR, train=train)
            else:
                preU += self.bu
                preR += self.br

            u = torch.sigmoid(preU)
            r = torch.sigmoid(preR)

            # Update hidden state
            preH =  X_t @ self.U + (h * r) @ self.W
            if self.bN:
                preH = self.bN_h.forward(preH, train=train)
            else:
                preH += self.b
            h = (1 - u) * h + u * torch.tanh(preH)
            
        out = h @ self.V + self.bo
        return out
    
