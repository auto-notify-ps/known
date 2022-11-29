

#import numpy as np
#import matplotlib.pyplot as plt
#import os, datetime
import torch as tt
import torch.nn as nn
from io import BytesIO


def build_dense_sequential(in_dim, layer_dims, out_dim, actF, actL ):
    """ a fully connected dense layer, commonly attached at the end of other networks """

    layers = [nn.Linear(in_dim, layer_dims[0]), actF()]
    for i in range(len(layer_dims)-1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        layers.append(actF())
    layers.append(nn.Linear(layer_dims[-1], out_dim))
    _ = None if actL is None else layers.append(actL())
    return nn.Sequential( *layers )

def save_w8s(path, model):tt.save(model.state_dict(), path)

def load_w8s(path, model): model.load_state_dict(tt.load(path))

def make_clone(model, detach=False, set_eval=False):
    """ Clone a model using memory buffer
        NOTE: use detach=True to sets the 'requires_grad' to 'False' on all of the parameters of the cloned model. """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters(): p.requires_grad=False
    if set_eval: model_copy.eval()
    buffer.close()
    del buffer
    return model_copy

def make_clones(model, n_copies, detach=False, set_eval=False):
    """ Clone a model multiple times using memory buffer
        NOTE: use detach=True to sets the 'requires_grad' to 'False' on all of the parameters of the cloned model. """
    buffer = BytesIO()
    
    tt.save(model, buffer)
    model_copies = []
    for _ in range(n_copies):
        buffer.seek(0)
        model_copy = tt.load(buffer)
        if detach:
            for p in model_copy.parameters(): p.requires_grad=False
        if set_eval: model_copy.eval()
        model_copies.append(model_copy)
    buffer.close()
    del buffer
    return tuple(model_copies)
