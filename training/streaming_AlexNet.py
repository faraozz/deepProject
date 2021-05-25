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

if __name__ == "__main__":

    NUMBER_OF_CLASSES = 200  # set here! 200 for standard, 3 for mini, 4 for rotations

    # CUDA for torch
    use_cuda = True  # for now, avoid GPU stuff
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False

    # parameters
     # name to save model under
    max_epochs = 20
    params = {
        'batch_size': 128,
        'shuffle': False,  # already shuffled
        'num_workers':4
    }
    lrt = 0.001
    momentumt = 0.9
    weight_decayt = 0.0005
    pretrainedt = False
    model_name = "regular_classification_batch_size"+str(params["batch_size"])+"_epochs"+str(max_epochs)+"_mom"+str(momentumt)+"_weightdec"+str(weight_decayt)+"_lrt"+str(lrt)+"_pretrained"+str(pretrainedt)
    #model = NiN(3, NUMBER_OF_CLASSES)  # 3 input channel (colour images), NUMBER_OF_CLASSES output channels (= NUMBER_OF_CLASSES outputs)
    #model = alexnet(NUMBER_OF_CLASSES,pretrained = True)
    #model = model.to(device)

    AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=pretrainedt)
    AlexNet_model.classifier[4] = nn.Linear(4096,1024)
    AlexNet_model.classifier[6] = nn.Linear(1024,NUMBER_OF_CLASSES)
    AlexNet_model.to(device)

    optimizer = torch.optim.SGD(AlexNet_model.parameters(), lr=lrt, momentum=momentumt, weight_decay = weight_decayt)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.2)

    # Datasets
    trainpath = "E:/kth/deep/deepProject/data/traindata.pkl"
    valpath = "E:/kth/deep/deepProject/data/valdata.pkl"

    # Generators
    #training_set = Dataset(trainpath)  # for regular labels
    training_set = Dataset(trainpath)  # for rotations
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    train_size = len(training_set)

    #validation_set = Dataset(valpath)  # for regular labels
    validation_set = Dataset(valpath)  # for rotations
    validation_generator = torch.utils.data.DataLoader(validation_set, **params)
    val_size = len(validation_set)

    training_loss_list = np.zeros((1, max_epochs))
    training_acc_list = np.zeros((1, max_epochs))
    validation_loss_list = np.zeros((1, max_epochs))
    validation_acc_list = np.zeros((1, max_epochs))

    # loop over epochs
    print("training started")
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
        training_loss_list[0][epoch] = total_train_loss/train_size
        training_acc_list[0][epoch] = train_correct/train_size
        validation_loss_list[0][epoch] = total_val_loss/val_size
        validation_acc_list[0][epoch] = val_correct/val_size
        scheduler.step()

    # Save model
    torch.save({'model_state_dict': AlexNet_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),'training_loss': training_loss_list, 'training_acc': training_acc_list, 'validation_loss': validation_loss_list, 'validation_acc': validation_acc_list}, join("E:/kth/deep/deepProject/training/models/checkpoints", model_name))
