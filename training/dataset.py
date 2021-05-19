"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import torch
import numpy as np
import torchvision.transforms as transforms

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

class RotDataset(torch.utils.data.Dataset):
    def __init__(self, im_paths):
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
        :return:
        """
        if rot is None:
            rot = np.random.randint(0, 4)

        # select sample
        impath = self.im_paths[index]

        # load data and get label
        X = torch.load(impath)
        X = self.rotate_img(X, rot)
        y = rot

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
            return transform(img)
        elif angle == 1:  # 90 degrees rotation
            return transform(np.flipud(np.transpose(img, (1, 0, 2))).copy())
        elif angle == 2:  # 180 degrees rotation
            return transform(np.fliplr(np.flipud(img)).copy())
        elif angle == 3:  # 270 degrees rotation
            return transform(np.transpose(np.flipud(img), (1, 0, 2)).copy())
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

