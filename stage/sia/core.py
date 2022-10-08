import os
from turtle import forward
import numpy as np

import torch as tt
import torch.nn as nn
import torch.functional as ff
import torch.optim as oo
import torchvision
from torchvision import models
import cv2

from .common import MLP, build_sequential,  clone_model


#%%___________________________________


# prepare +ve and -ve samples

# randomly sample an image from taining set
# ... another from  same class ====> label = 1 (+ve class)
# ... another from  different class ====> label = 0 (-ve class)

# to load and dataset and seperate out images


def load_dataset(path):
    
    # assume that each class has itsown directory
    
    classes = os.listdir(path)
    # classes should be numbered in a list
    class_labels = []
    class_images = []
    for c in classes:
        cpath = os.path.join(path,c)
        if os.path.isdir(cpath):
            class_samples = os.listdir(cpath)
            class_labels.append(c)
            class_images.append([os.path.join(cpath, s) for s in class_samples])
    return class_labels, class_images


class DataSet:
    def __init__(self, path, seed=None):
        self.class_labels, self.class_images = load_dataset(path)
        self.class_sample_count = [ len(cL) for cL in self.class_images ]
        self.rng = np.random.default_rng(seed)
        
        self.C = len(self.class_labels) #<------ no of classes
        self.info()
        

    def info(self):
        for k,v in zip(self.class_labels, self.class_sample_count):
            print(k, v)
        

    # for pairwise learning
    
    # 1. TOTALLY RANDOM CLASSES
    def equi_sample(self, n):
        # within each class select 'n' random images
        cS=[]
        for c in range(self.C):    
            cL =  self.class_images[c]
            cS.append([cL[i] for i in self.rng.integers(0, len(cL), size=n)])
        return cS

    def make_pairs(self, cS):
        pos_paired, neg_paired = [], []

        pos_pair_count, neg_pair_count = 0, 0
        for c,cL in enumerate(cS):
            ncL = len(cL)

            # positive pairing
            for i in range(ncL):
                for j in range(i, ncL):
                    pos_paired.append( (cL[i], cL[j], 1) )
                    pos_pair_count+=1
                    
            # negative pairs
            
            for i in range(ncL):
                for d in range(len(cS)):
                    if d==c: continue
                    cN = cS[d]
                    for b in cN:
                        neg_paired.append((cL[i], b, 0))
                        neg_pair_count+=1


        return pos_paired, neg_paired, pos_pair_count, neg_pair_count

    def make_paire(self, cS):
        pos_paired, neg_paired = [], []

        pos_pair_count, neg_pair_count = 0, 0
        
        for c,cL in enumerate(cS):
            ncL = len(cL)
            equi = int((ncL*(ncL+1)*0.5)/(self.C-1))

            # positive pairing
            for i in range(ncL):
                for j in range(i, ncL):
                    pos_paired.append( (cL[i], cL[j], 1) )
                    pos_pair_count+=1
                    
            # negative pairs
            
            for i in range(ncL):
                for d in range(len(cS)):
                    if d==c: continue
                    cN = cS[d]
                    cNL = len(cN)
                    cnL_chosen = self.rng.integers(0, cNL, size=equi )
                    for b in cnL_chosen:
                        neg_paired.append((cL[i], cN[b], 0))
                        neg_pair_count+=1


        return pos_paired, neg_paired, pos_pair_count, neg_pair_count

#%%___________________________________

def metric_l1(h1, h2):
    return tt.abs(h1-h2)

def metric_l2(h1, h2):
    return tt.sqrt(tt.pow(h1-h2, 2))

class Sia(nn.Module):
    def __init__(self, encoder, encoder_out, metricF, dense_layers, dense_actF=nn.ReLU ):
        super(Sia, self).__init__()
        self.encoder = nn.Sequential(encoder, nn.Flatten())
        
        self.encoder_out = encoder_out
        self.metricF = metricF
        self.dense = build_sequential(self.encoder_out, dense_layers, 1, actF=dense_actF, actL=nn.Sigmoid)
    
    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z = self.metricF(h1, h2)
        #print(f'{x1.shape=}, {x2.shape=}, {h1.shape=}, {h2.shape=},  {z.shape=}')
        return self.dense(z)



def vgg11_bn():
    return clone_model(models.vgg11_bn(pretrained=False).features)
