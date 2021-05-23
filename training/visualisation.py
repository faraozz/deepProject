"""
Visualisation functions, particularly to visualise feature maps.
Feature map activation tutorial: https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/
"""

from os.path import join
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.image import imread

def convnet_feature_activations(im, model, layers=None, features=None, savepath="../results/layer_vis"):
    """
    given a convolutional network, pass an image through the layers and visualise the activations
    :param im: torch array, input image
    :param model: torch convolutional neural network model
    :param layer: optional list of ints, specify (a) specific layer(s) to visualise
    :param features: optional list of ints, specify (a) specific feature(s) to visualise
    :param savepath: optional string, provide a path to save the images to. Otherwise default it used.
    :return:
    """
    n_layers = 0
    conv_layers = []

    model_children = list(model.children())

    for child in model_children:
        if type(child) == nn.Conv2d:
            n_layers += 1
            conv_layers.append(child)
        elif type(child)==nn.Sequential:
            for layer in child.children():
                if type(layer)==nn.Conv2d:
                    n_layers += 1
                    conv_layers.append(layer)

    results = [conv_layers[0](im)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))  # push image through convolutional layers
    outputs = results

    for num_layer in range(len(outputs)):
        if layers is not None and num_layer not in layers:
            continue  # if a layer has been specified, skip all other layers
        layer_vis = outputs[num_layer][0, :, :, :]
        layer_vis = layer_vis.data
        plt.title("layer {}".format(num_layer))
        for i, filter in enumerate(layer_vis):
            if features is not None and i not in features:
                continue  # if specific features are sought, ignore all others
            if i == 16:
                break
            plt.subplot(2, 8, i+1)
            plt.imshow(filter)
            plt.axis("off")
        plt.savefig(join(savepath, "layer_{}.png".format(num_layer)))
        plt.show()

    # TODO: scale and overlay activations onto original images

