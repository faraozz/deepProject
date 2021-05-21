"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import numpy as np
import torch
from os.path import join
from dataset import Dataset
from dataset import RotDataset
from NetworkInNetwork import NiN
#from AlexNet import alexnet
import torch.nn as nn
import torch.nn.functional as F

NUMBER_OF_CLASSES = 4  # set here! 200 for standard, 3 for mini, 4 for rotations

# CUDA for torch
use_cuda = True  # for now, avoid GPU stuff
device = torch.device("cuda")
torch.backends.cudnn.benchmark = False

# parameters
model_name = "test"  # name to save model under
max_epochs = 25
params = {
    'batch_size': 6,
    'shuffle': False,  # already shuffled
    'num_workers': 0
}
#model = NiN(3, NUMBER_OF_CLASSES)  # 3 input channel (colour images), NUMBER_OF_CLASSES output channels (= NUMBER_OF_CLASSES outputs)
#model = alexnet(NUMBER_OF_CLASSES,pretrained = True)
#model = model.to(device)

AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
AlexNet_model.classifier[4] = nn.Linear(4096,1024)
AlexNet_model.classifier[6] = nn.Linear(1024,NUMBER_OF_CLASSES)
AlexNet_model.to(device)

optimizer = torch.optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

# Datasets
trainpath = "../data/traindata.pkl"
valpath = "../data/valdata.pkl"

# Generators
#training_set = Dataset(trainpath)  # for regular labels
training_set = RotDataset(trainpath)  # for rotations
training_generator = torch.utils.data.DataLoader(training_set, **params)
train_size = len(training_set)

#validation_set = Dataset(valpath)  # for regular labels
validation_set = RotDataset(valpath)  # for rotations
validation_generator = torch.utils.data.DataLoader(validation_set, **params)
val_size = len(validation_set)

# loop over epochs
for epoch in range(max_epochs):
    total_train_loss = 0
    total_val_loss = 0
    train_correct = 0
    val_correct = 0
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.int64)
        local_labels = torch.squeeze(local_labels) # remove dimension of size 1

        # Model computations
        #print(local_batch.shape)
        #print(AlexNet_model)
        pred = AlexNet_model(local_batch)
        loss = criterion(pred, local_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        #print(loss.item())

        # accuracy
        pred_indices = pred.argmax(1)
        train_correct += pred_indices.eq(local_labels).sum().item()
    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device, dtype=torch.float), local_labels.to(device, dtype=torch.int64)
            local_labels = torch.squeeze(local_labels) # remove dimension of size 1

            # Model computations
            
            pred = AlexNet_model(local_batch)
            loss = criterion(pred, local_labels)

            optimizer.zero_grad()
            total_val_loss += loss.item()

            # accuracy
            pred_indices = pred.argmax(1)
            val_correct += pred_indices.eq(local_labels).sum().item()

    print("Epoch {}: train_loss = {}, train_acc = {}, val_loss = {}, val_acc = {}".format(epoch,
                                                                                          total_train_loss/train_size,
                                                                                          train_correct/train_size,
                                                                                          total_val_loss/val_size,
                                                                                          val_correct/val_size))

# Save model
torch.save(AlexNet_model.state_dict(), join("../models/checkpoints", model_name))

