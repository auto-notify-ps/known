

#import numpy as np
#import matplotlib.pyplot as plt
import os, datetime
import torch as tt
import torch.nn as nn
from io import BytesIO


now = datetime.datetime.now
fake = lambda members: type('object', (object,), members)()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""" [D] [torch.nn] 
    Some basic Neural Net models and helpers functions using torch.nn """
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def save_models(dir_name, file_names, models):
    os.makedirs(dir_name, exist_ok=True)
    for θ, f in zip(models, file_names):
        tt.save(θ, os.path.join(dir_name, f))

def load_models(dir_name, file_names):
    return tuple( [tt.load(os.path.join(dir_name, f)) for f in file_names ])

def save_model(path, model):
    tt.save(model, path)

def load_model(path):
    return tt.load(path)

def clone_model(model, detach=False):
    """ use detach=True to sets the 'requires_grad' to 'False' on all of the parameters of the cloned model. """
    buffer = BytesIO()
    tt.save(model, buffer)
    buffer.seek(0)
    model_copy = tt.load(buffer)
    if detach:
        for p in model_copy.parameters():
            p.requires_grad=False
    model_copy.eval()
    del buffer
    return model_copy

def build_sequential(in_dim, layer_dims, out_dim, actF, actL ):
    layers = [nn.Linear(in_dim, layer_dims[0]), actF()]
    for i in range(len(layer_dims)-1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        layers.append(actF())
    layers.append(nn.Linear(layer_dims[-1], out_dim))
    _ = None if actL is None else layers.append(actL())
    return nn.Sequential( *layers )

class MLP(nn.Module):
    """ Multi layer Perceptron """
    def __init__(self, in_dim, layer_dims, out_dim, actF, actL):
        super(MLP, self).__init__()
        self.net = build_sequential(in_dim, layer_dims, out_dim, actF, actL )
    def forward(self, x):
        return self.net(x)
