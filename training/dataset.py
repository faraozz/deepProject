"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, im_paths, labels):
        self.labels = labels
        self.im_paths = im_paths

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
        X = torch.load(impath)
        y = self.labels[index]

        return X, y

