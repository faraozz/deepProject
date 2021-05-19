"""
Load tiny imagenet 200 images and convert them into a single numpy array
"""

import numpy as np
import pickle
from os import listdir
from os.path import join

np.random.seed(1)

datapath = "D:/datasets/tiny-imagenet-200"
savepath = "../data"
trainpath = join(datapath, "train")
valpath = join(datapath, "val")

trainclasspaths = listdir(trainpath)
K = len(trainclasspaths)
n_classims = 500  # 500 images per class

#
# prepare training data
#
X_train_paths = []
Y_train_labs = []
label2id = {}
for i, classpath in enumerate(trainclasspaths):
    ims = listdir(join(trainpath, classpath, "images"))
    label2id[int(classpath[1:])] = i
    for im in ims:
        X_train_paths.append(join(trainpath, classpath, "images", im))
        Y_train_labs.append(classpath)

Y_train_ints = [int(y[1:]) for y in Y_train_labs]

# shuffle datapoints
indices = np.arange(len(X_train_paths))
np.random.shuffle(indices)
X_train_paths, Y_train_ints = np.asarray(X_train_paths), np.asarray(Y_train_ints)
X_train_paths, Y_train_ints = X_train_paths[indices], Y_train_ints[indices]

#
# Prepare validation data
#

vallabelpath = join(valpath, "val_annotations.txt")
with open(vallabelpath, "r") as f:
    vallabeldata = f.readlines()

val_labels = [line.strip().split("\t")[1] for line in vallabeldata]
val_ints = np.asarray([int(label[1:]) for label in val_labels])
val_paths = [join(valpath, "images", line.strip().split("\t")[0]) for line in vallabeldata]
val_paths = np.asarray(val_paths)

#
# save data
#

#np.savez(join(savepath, "traindata.npz"), X_paths=X_train_paths, Y_ints=Y_train_ints)
#np.savez(join(savepath, "valdata.npz"), X_paths=val_paths, Y_ints=val_ints)

with open(join(savepath, "traindata.pkl"), "wb") as f:
    pickle.dump([X_train_paths, Y_train_ints, label2id], f)

with open(join(savepath, "valdata.pkl"), "wb") as f:
    pickle.dump([val_paths, val_ints, label2id], f)

