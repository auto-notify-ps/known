import os

import numpy as np

import torch as tt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as oo
import torchvision
from torchvision import models
import cv2

from .common import MLP, build_sequential,  clone_model
import random
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import PIL.ImageOps   
#%%___________________________________


# prepare +ve and -ve samples

# randomly sample an image from taining set
# ... another from  same class ====> label = 1 (+ve class)
# ... another from  different class ====> label = 0 (-ve class)

# to load and dataset and seperate out images

class SiameseNetworkDataset(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        
    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])

        #img0 = img0.convert("L")
        #img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, tt.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



class SiameseNetworkDatasetBase(Dataset):
    def __init__(self,imageFolderDataset,transform=None):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform

        
    def __getitem__(self, index):
        imgT = self.imageFolderDataset.imgs[index]
        #print(f'{imgT=}')
        img = Image.open(imgT[0])

        if self.transform is not None:
            img = self.transform(img)
        
        return img, tt.from_numpy(np.array(imgT[1], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)



#%%___________________________________

def vgg11_bn():
    return clone_model(models.vgg11_bn(pretrained=False).features)

def simple_cnn():
    return nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11,stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3,stride=1),
            nn.ReLU(inplace=True)
        )

#create the Siamese Neural Network
class SiameseNetwork(nn.Module):


    def __init__(self, encoder, dense):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.encoder = nn.Sequential(encoder, nn.Flatten())

        # Setting up the Fully Connected Layers
        ld = len(dense)
        layers= []
        for d in range(ld-1):
            layers.append(nn.Linear(dense[d], dense[d+1]))
            if d<ld-2: layers.append(nn.ReLU(inplace=True))

        self.dense = nn.Sequential( *layers )
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.encoder(x)
        #output = output.view(output.size()[0], -1)
        output = self.dense(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2

# Define the Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = tt.mean((label) * tt.pow(euclidean_distance, 2) +
                                    (1-label) * tt.pow(tt.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive