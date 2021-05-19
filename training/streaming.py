"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import numpy as np
import torch
from os import listdir
from os.path import join
from training.dataset import Dataset

# CUDA for torch
use_cuda = False  # for now, avoid GPU stuff
device = torch.device("cpu")
torch.backends.cudnn.benchmark = True

# parameters
max_epochs = 10
params = {
    'batch_size': 64,
    'shuffle': False,  # already shuffled
    'num_workers': 2
}

# Datasets
trainpath = "../data/traindata.npz"
valpath = "../data/valdata.npz"

traindata = np.load(trainpath)
valdata = np.load(valpath)

trainX_paths, trainY_labels = traindata['X_paths'], traindata['Y_ints']
valX_paths, valY_labels = valdata['X_paths'], valdata['Y_ints']

# Generators
training_set = Dataset(trainX_paths, trainY_labels)
training_generator = torch.utils.data.Dataloader(training_set, **params)

validation_set = Dataset(valX_paths, valY_labels)
validation_generator = torch.utils.data.Dataloader(validation_set, **params)

# loop over epochs
for epoch in range(max_epochs):
    for local_batch, local_labels in training_generator:
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations

