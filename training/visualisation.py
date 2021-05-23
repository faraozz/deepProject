"""
Visualisation functions, particularly to visualise feature maps.
Feature map activation tutorial: https://androidkt.com/how-to-visualize-feature-maps-in-convolutional-neural-networks-using-pytorch/
"""

from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib.image import imread
from skimage.transform import resize
torch.manual_seed(1)

def process_mat(mat, min=0, max=1, transpose=False):
    mat = np.squeeze(mat)
    if transpose:
        mat = mat.transpose()
    mat = ((mat - np.min(mat)) / (np.max(mat) - np.min(mat)) + min) * max
    return mat


def convnet_feature_activations(im, model, t_layer=0, layers=None, features=None, savepath="../results/layer_vis"):
    """
    given a convolutional network, pass an image through the layers and visualise the activations
    :param im: torch array, input image
    :param model: torch convolutional neural network model
    :param t_layer: int, the layer to visualise
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
        elif type(child) == nn.Sequential:
            for layer in child.children():
                if type(layer) == nn.Conv2d:
                    n_layers += 1
                    conv_layers.append(layer)

    results = [conv_layers[0](im)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))  # push image through convolutional layers
    outputs = results

    t_feature_maps = []
    for num_layer in range(len(outputs)):
        if layers is not None and num_layer not in layers:
            continue  # if a layer has been specified, skip all other layers
        layer_vis = outputs[num_layer][0, :, :, :]
        layer_vis = layer_vis.data
        if num_layer == t_layer:
            t_feature_maps = layer_vis
        #plt.title("layer {}".format(num_layer))
        for i, filter in enumerate(layer_vis):
            if features is not None and i not in features:
                continue  # if specific features are sought, ignore all others
            if i == 16:
                break
            # plt.subplot(2, 8, i+1)
            # plt.imshow(filter)
            # plt.axis("off")
        # plt.savefig(join(savepath, "layer_{}.png".format(num_layer)))
        # plt.show()
    return t_feature_maps


def visualise_image_activations(im, model, target=0, power_scale=2, savepath="../results/layer_vis"):
    target_features = convnet_feature_activations(im, model, t_layer=target, savepath=savepath)
    im_mat = im.numpy()
    tf_mat = target_features.numpy()

    tf_mat = tf_mat ** (power_scale ** target)
    tf_mat = np.sum(tf_mat, axis=0)
    im_mat, tf_mat = process_mat(im_mat), process_mat(tf_mat)

    tf_mat = resize(tf_mat, im_mat.shape[1:])

    overlaid_mat = im_mat * np.stack([tf_mat] * 3, axis=0)
    overlaid_mat = process_mat(overlaid_mat)
    plt.imshow(overlaid_mat.transpose())
    plt.title("layer {}".format(target + 1))
    plt.savefig(join(savepath, "layer{}_attention_map.png".format(target)))
    plt.show()

def demo():
    impath = "../results/n02123159_tiger_cat.jpg"
    savepath = "../results/layer_vis"

    #
    # read image
    #
    im = imread(impath)
    plt.imshow(im)
    plt.show()

    immeans, imstds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    normalise = transforms.Normalize(immeans, imstds)
    unnormalise = transforms.Normalize(-1 * np.asarray(immeans) / np.asarray(imstds),
                                       1.0 / np.asarray(imstds))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalise
    ])
    im = transform(im)
    im = im.unsqueeze(0)

    # load model
    AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)

    #
    # display results
    #
    im_mat = unnormalise(im)
    im_mat = np.squeeze(im_mat.numpy())
    plt.imshow(im_mat.transpose())
    plt.title("original image")
    plt.savefig(join(savepath, "original_image.png"))
    plt.show()

    visualise_image_activations(im, AlexNet_model, target=0, savepath=savepath)
    visualise_image_activations(im, AlexNet_model, target=1, savepath=savepath)
    visualise_image_activations(im, AlexNet_model, target=2, savepath=savepath)
    visualise_image_activations(im, AlexNet_model, target=3, savepath=savepath)

