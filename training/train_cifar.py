"Inspired from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from os.path import join
from training.transfer_learning import freeze_last_layers, load_model

CIFAR_PATH = '../cifar-10-batches-py'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#Downloading training data
train_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=6, shuffle=True, num_workers=0)

#Downloading test data
test_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=False, transform=transform)

testloader = torch.utils.data.DataLoader(test_data, batch_size=6, shuffle=False, num_workers=0)

#Now using the AlexNet
modelpath = "../models/checkpoints"
loaded_model_classes = 4  # 200 for imagenet, 4 for rotated imagenet
AlexNet_model = load_model("alexnet", modelpath, loaded_model_classes, 10)
AlexNet_model = freeze_last_layers(AlexNet_model)

#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Move the input and AlexNet_model to GPU for speed if available
AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training of AlexNet')

#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Save model
model_name = "cifar10_6_0.001_0.9_10"
torch.save(AlexNet_model.state_dict(), join("../models/checkpoints/", model_name))