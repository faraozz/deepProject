
from os.path import join
import numpy as np
import torch
import torch.nn as nn
from models.NetworkInNetwork import NiN

####################### variables to set before running ##############################
NUMBER_OF_CLASSES = 10

#modelname = "mnist_6_0.001_0.9_10"
#modeldir = "../models"
#modelpath = join(modeldir, modelname)

#modeltype = "alexnet"
#trained_model_classes = 4
######################################################################################

#
# load model
#

def load_model(modeltype, modelpath, trained_model_classes, new_model_classes=None):
    """
    Load a model of a particular type and load stored parameters
    :param modeltype: torch object, the neural network object or which parameters are loaded
    :param modelpath: string, path of the model parameters
    :param trained_model_classes: the number of output classes of the trained model being loaded
    :param new_model_classes: optional scalar, the number of output parameters in the new model (replacing the
    original output layer)
    :return: torch object, a neural network model with the parameters loaded
    """
    if modeltype == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)
        #model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=1, padding=2)
        model.classifier[4] = nn.Linear(4096, 1024)
        model.classifier[6] = nn.Linear(1024, trained_model_classes)
        checkpoint = torch.load(modelpath)
        model.load_state_dict(checkpoint['model_state_dict'])
        if new_model_classes is not None:  # if a different number of model classes, replace the last layer entirely
            model.classifier[6] = nn.Linear(1024, new_model_classes)
    elif modeltype == "nin":
        model = NiN(3, trained_model_classes)
        model.load_state_dict(torch.load(modelpath))
        if new_model_classes is not None:
            pass  # TODO
    else:
        raise Exception("not a valid model type!")
    return model

#
# freeze all except final n layers
#

def freeze_last_layers(model, n_free=1):
    """
    Given a model, freeze all except the last n_free layers so that only these last layers can be trained
    :param model: torch object, a neural network model (preferably with trained parameters)
    :param n_free: int, the number of last layers to free for training
    :return: torch object, the neural network model with the last n_free layers free for training
    """
    for param in model.parameters():
        param.requires_grad = False

    num_layers = len(model.features)
    for i, param in enumerate(model.parameters()):
      if i >= num_layers - n_free:
        param.requires_grad = True

    return model

