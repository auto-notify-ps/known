

#import numpy as np
#import matplotlib.pyplot as plt
#import os, datetime
import torch as tt
import torch.nn as nn
from .common import build_dense_sequential

class MLP(nn.Module):
    """ Multi layer Perceptron with 1 input vector and 1 output vector"""

    def __init__(self, in_dim, layer_dims, out_dim, actF, actL):
        super(MLP, self).__init__()
        self.model = build_dense_sequential(in_dim, layer_dims, out_dim, actF, actL )

    def forward(self, x):
        return self.model(x)


class MLPn(nn.Module):
    """ Multi layer Perceptron with n input vectors and 1 output vector """

    def __init__(self, in_dims, layer_dims, out_dim, actF, actL):
        super(MLPn, self).__init__()
        self.model = build_dense_sequential(sum(in_dims), layer_dims, out_dim, actF, actL )

    def forward(self, x): #<--- here x is a tuple
        return self.model(tt.concat(x, dim=-1))
