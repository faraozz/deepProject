"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
from matplotlib import image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        im_paths, labels, label2id = data
        n_labels = len(label2id.keys())

        self.labels = labels
        self.im_paths = im_paths
        self.label2id = label2id
        self.n_labels = n_labels

    def __len__(self):
        """
        The total number of samples
        :return:
        """
        return len(self.im_paths)

    def __getitem__(self, index):
        """
        Generate one sample of data
        :param index:
        :return:
        """
        # select sample
        impath = self.im_paths[index]

        # load data and get label
        X = image.imread(impath).transpose()
        if len(X.shape) == 2:  # add extra channels in the case of monochrome images
            X = np.stack([X, X, X], axis=0)
        X = torch.from_numpy(X.astype(np.float64))
        class_id = self.label2id[self.labels[index]]
        class_id = torch.from_numpy(np.asarray([class_id]))
        y = class_id

        return X, y

class RotDataset(torch.utils.data.Dataset):
    def __init__(self, datapath):
        with open(datapath, "rb") as f:
            data = pickle.load(f)

        im_paths, _, _ = data

        self.im_paths = im_paths

    def __len__(self):
        """
        The total number of samples
        :return:
        """
        return len(self.im_paths)

    def __getitem__(self, index, rot=None):
        """
        Generate one sample of data
        :param index:
        :param rot: optional, index of which rotation to use (out of {0, 1, 2, 3})
        :return: X, y. Note that the CrossEntropy function expects the target value to be a class index
        """
        if rot is None:
            rot = np.random.randint(0, 4)

        # select sample
        impath = self.im_paths[index]

        # load data and get label
        X = image.imread(impath)
        if len(X.shape) == 2:
            X = np.stack([X, X, X], axis=2)  # axis 2 because NOT transposed yet
        X = self.rotate_img(X, rot)
        #ylab = [0 0 0 0]
        #ylab(rot)
        y = torch.from_numpy(np.asarray([rot]))


        return X, y

    def rotate_img(self, img, angle):
        # For imageNet
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_pix, std=std_pix)
        ])
        if angle == 0:  # 0 degrees rotation
            return transform(np.array(img))
        elif angle == 1:  # 90 degrees rotation
            return transform(np.flipud(np.transpose(img, (1, 0, 2))).copy())
        elif angle == 2:  # 180 degrees rotation
            return transform(np.fliplr(np.flipud(img)).copy())
        elif angle == 3:  # 270 degrees rotation
            return transform(np.transpose(np.flipud(img), (1, 0, 2)).copy())
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

