"""
tiny imagenet has 200 classes and 500 images per class
"""

import numpy as np
from os import listdir
from os.path import join

datapath = "D:/datasets/tiny-imagenet-200"
trainpath = join(datapath, "train")

trainclasspaths = listdir(trainpath)
K = len(trainclasspaths)


