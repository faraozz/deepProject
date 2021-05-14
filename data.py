"""Preparing dataset."""
from __future__ import print_function

import argparse
from datetime import datetime
import os

import torch
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchnet as tnt
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

_IMAGENET_DATASET_DIR = "./tiny-imagenet-200"



train_data_dir = _IMAGENET_DATASET_DIR + '/' + 'train'

transforms_list_train = [transforms.Scale(256),transforms.CenterCrop(224),lambda x: np.asarray(x),]
transform = transforms.Compose(transforms_list_train)

#Get the dataset
dataset_origin = ImageFolder(train_data_dir, transform)


#Turn images
def rotate_img(img, angle):
    if angle == 0: # 0 degrees rotation
        return img
    elif angle == 90: # 90 degrees rotation
        return np.flipud(np.transpose(img, (1,0,2))).copy()
    elif angle == 180: # 180 degrees rotation
        return np.fliplr(np.flipud(img)).copy()
    elif angle == 270: # 270 degrees rotation
        return np.transpose(np.flipud(img), (1,0,2)).copy()
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


mean_pix = [0.485, 0.456, 0.406]
std_pix = [0.229, 0.224, 0.225]
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
#y = torch.ones([len(dataset_origin)*4, 1], dtype=torch.float64)
#X = torch.ones([len(dataset_origin)*4,1], dtype=torch.float64)

img, _=dataset_origin[0]
rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
rotation_labels = torch.LongTensor([0, 1, 2, 3])
rotated_imgs = torch.stack(rotated_imgs, dim=0)
X = rotated_imgs
y=rotation_labels

dataset = [(X[0],y[0]), (X[1],y[1]),(X[2],y[2]),(X[3],y[3])]

for i in range(1,len(dataset_origin)):
    img, _=dataset_origin[i]
    rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    rotated_imgs = torch.stack(rotated_imgs, dim=0)
    
    #X = torch.cat((X,rotated_imgs),0)
    #y = torch.cat((y,rotation_labels),0)
    #print(X.shape)
    #print(y.shape)
    dataset.append((rotated_imgs[0],rotation_labels[0]))
    dataset.append((rotated_imgs[1],rotation_labels[1]))
    dataset.append((rotated_imgs[2],rotation_labels[2]))
    dataset.append((rotated_imgs[3],rotation_labels[3]))
    #return torch.stack(rotated_imgs, dim=0), rotation_labels
print(len(dataset))
print(dataset[0][0].shape)
print(dataset[1][0].shape)
print(dataset[2][0].shape)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=6)
print(len(train_dataloader))

for X,y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

#class DataPrep(dataset_origin):
#    def _init_(self, dataset_origin):
#        self.dataset_origin = dataset_origin
#    def load_fct(self,idx):
#        img, _ = self.dataset_origin[idx]
#        rotated_imgs = [self.transform(img0),self.transform(rotate_img(img0,  90)),self.transform(rotate_img(img0, 180)),self.transform(rotate_img(img0, 270))]
#        rotation_labels = torch.LongTensor([0, 1, 2, 3])
#        return torch.stack(rotated_imgs, dim=0), rotation_labels
#    
#    def _call_(self):
#        rot_dataset = tnt.dataset.ListDataset(elem_list=range(len(self.dataset_origin)),load=load_fct)
#        data_loader = rot_dataset.parallel(batch_size=6,collate_fn=default_collate, num_workers=0,shuffle=shuffle)
#        return data_loader
#
#dataloader = DataPrep(dataset_origin)
#
#for b in dataloader(0):
#    data, label = b
#    print(data, label)
#    break

#Load dataset