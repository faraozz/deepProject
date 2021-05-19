"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import numpy as np
import torch
from os.path import join
from training.dataset import Dataset
from models.NetworkInNetwork import NiN

# CUDA for torch
use_cuda = False  # for now, avoid GPU stuff
device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

# parameters
model_name = "test"  # name to save model under
max_epochs = 10
params = {
    'batch_size': 64,
    'shuffle': False,  # already shuffled
    'num_workers': 2
}
model = NiN(1, 4)  # 3 input channel (colour images), 4 output channels (= 4 outputs)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# Datasets
trainpath = "../data/traindata.npz"
valpath = "../data/valdata.npz"

# Generators
training_set = Dataset(trainpath)
training_generator = torch.utils.data.Dataloader(training_set, **params)

validation_set = Dataset(valpath)
validation_generator = torch.utils.data.Dataloader(validation_set, **params)

# loop over epochs
for epoch in range(max_epochs):
    train_loss = None
    val_loss = None
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        optimizer.zero_grad()
        pred = model(local_batch)
        loss = criterion(pred, local_labels)
        loss.backward()
        optimizer.step()
        train_loss = loss.data[0]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            optimizer.zero_grad()
            pred = model(local_batch)
            loss = criterion(pred, local_labels)
            val_loss = loss.data[0]

    print("Epoch {}: train_loss = {}, val_loss = {}".format(epoch, train_loss, val_loss))

# Save model
torch.save(model.state_dict(), join("../models/checkpoints", model_name))

