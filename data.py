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
val_data_dir = _IMAGENET_DATASET_DIR + '/' + 'val'
test_data_dir = _IMAGENET_DATASET_DIR + '/' + 'test'

transforms_list_train = [transforms.Scale(256),transforms.CenterCrop(224),lambda x: np.asarray(x),]
transform = transforms.Compose(transforms_list_train)

#Get the dataset
dataset_origin = ImageFolder(train_data_dir, transform)
val_dataset_origin = ImageFolder(val_data_dir, transform)
test_dataset_origin = ImageFolder(test_data_dir, transform)


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

#For imageNet
mean_pix = [0.485, 0.456, 0.406]
std_pix = [0.229, 0.224, 0.225]
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])


#train
img, _=dataset_origin[0]
rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
rotation_labels = torch.LongTensor([0, 1, 2, 3])
rotated_imgs = torch.stack(rotated_imgs, dim=0)
X = rotated_imgs
y=rotation_labels

dataset = [(X[0],y[0]), (X[1],y[1]),(X[2],y[2]),(X[3],y[3])]

#validation
img1, _=val_dataset_origin[0]
rotated_imgs = [transform(img1),transform(rotate_img(img1,  90)),transform(rotate_img(img1, 180)),transform(rotate_img(img1, 270))]
rotation_labels = torch.LongTensor([0, 1, 2, 3])
rotated_imgs = torch.stack(rotated_imgs, dim=0)
X = rotated_imgs
y=rotation_labels

val_dataset = [(X[0],y[0]), (X[1],y[1]),(X[2],y[2]),(X[3],y[3])]

#test
img2, _=dataset_origin[0]
rotated_imgs = [transform(img2),transform(rotate_img(img2,  90)),transform(rotate_img(img2, 180)),transform(rotate_img(img2, 270))]
rotation_labels = torch.LongTensor([0, 1, 2, 3])
rotated_imgs = torch.stack(rotated_imgs, dim=0)
X = rotated_imgs
y=rotation_labels

test_dataset = [(X[0],y[0]), (X[1],y[1]),(X[2],y[2]),(X[3],y[3])]

for i in range(1,50):
    img, _=dataset_origin[i]
    rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    rotated_imgs = torch.stack(rotated_imgs, dim=0)
    
    dataset.append((rotated_imgs[0],rotation_labels[0]))
    dataset.append((rotated_imgs[1],rotation_labels[1]))
    dataset.append((rotated_imgs[2],rotation_labels[2]))
    dataset.append((rotated_imgs[3],rotation_labels[3]))
    
for i in range(1,50):
    img, _=val_dataset_origin[i]
    rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    rotated_imgs = torch.stack(rotated_imgs, dim=0)
    
    val_dataset.append((rotated_imgs[0],rotation_labels[0]))
    val_dataset.append((rotated_imgs[1],rotation_labels[1]))
    val_dataset.append((rotated_imgs[2],rotation_labels[2]))
    val_dataset.append((rotated_imgs[3],rotation_labels[3]))

for i in range(1,50):
    img, _=test_dataset_origin[i]
    rotated_imgs = [transform(img),transform(rotate_img(img,  90)),transform(rotate_img(img, 180)),transform(rotate_img(img, 270))]
    rotation_labels = torch.LongTensor([0, 1, 2, 3])
    rotated_imgs = torch.stack(rotated_imgs, dim=0)
    
    test_dataset.append((rotated_imgs[0],rotation_labels[0]))
    test_dataset.append((rotated_imgs[1],rotation_labels[1]))
    test_dataset.append((rotated_imgs[2],rotation_labels[2]))
    test_dataset.append((rotated_imgs[3],rotation_labels[3]))

print(val_dataset[0][0].shape)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=6)


for X,y in train_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

