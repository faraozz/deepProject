"""
tiny imagenet has 200 classes and 500 images per class
With some help from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

import numpy as np
import torch
from os.path import join
from dataset import Dataset
from dataset import RotDataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from NetworkInNetwork import NiN
import torch.nn as nn
import torch.nn.functional as F
import pickle
from os import listdir
from torchvision.io import read_image
from matplotlib import image
from dataset import Dataset

# Function to rotate a given image
def rotate_img(img, angle):
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


# Custom dataset (doing the rotation of each image)
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.list = listdir(img_dir)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_path = join(self.img_dir, "test_"+str(idx)+".JPEG")
        #image = read_image(img_path)
        #label = self.img_labels.iloc[idx, 1]
        X = image.imread(img_path)
        if len(X.shape) == 2:  # add extra channels in the case of monochrome images
            X = np.stack([X, X, X], axis=0)
        rot = np.random.randint(0, 4)
        X = np.reshape(X,(64,64,3))
        X = rotate_img(X, rot)
        label = torch.from_numpy(np.asarray(rot))
        sample = {"image": X, "label": label}
        return sample


if __name__ == '__main__':
        # Prepare test data

        np.random.seed(1)

        # Change datapath to be correct  ###########################################################################################
        datapath = "../tiny-imagenet-200/"
        testpath = join(datapath, "test/images/")



        # Change path to model to be correct #######################################################################################
        MODEL_PATH="models/best models/regular_classification_batch_size128_epochs20_mom0.9_weightdec0.0005_lrt0.001_pretrainedFalse"

        # CUDA for torch
        use_cuda = True
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False

        # Adapt parameters #########################################################################################################
        params = {
            'batch_size': 128,
            'shuffle': False,  # already shuffled
            'num_workers': 4
        }
        NUMBER_OF_CLASSES = 200

        # Load model
        AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)
        AlexNet_model.classifier[4] = nn.Linear(4096,1024)
        AlexNet_model.classifier[6] = nn.Linear(1024,NUMBER_OF_CLASSES)
        AlexNet_model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
        AlexNet_model.to(device)


        # Load test dataset
        test_dataset_origin = Dataset(testpath)

        test_dataloader = torch.utils.data.DataLoader(test_dataset_origin, **params)

        test_size = len(test_dataloader)


        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_dataloader:
                images, labels = data["image"].to(device,dtype=torch.float), data["label"].to(device, dtype=torch.int64)
                outputs = AlexNet_model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))




