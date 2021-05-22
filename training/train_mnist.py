"Inspired from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from os.path import join

MNIST_PATH = '../mnist'

t = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0), std=(1))]
                       )

#Downloading training data
dataset = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download=True, transform=t)
#print(len(dataset))

torch.manual_seed(43)
val_size = 6000
train_size = len(dataset) - val_size

train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
#print(len(train_data), len(val_data))

trainloader = torch.utils.data.DataLoader(train_data, batch_size=6, shuffle=True, num_workers=0)

validloader = torch.utils.data.DataLoader(val_data, batch_size=6, shuffle = True, num_workers=0)

#Downloading test data
test_data = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download=True, transform=t)
#print(len(test_data))
testloader = torch.utils.data.DataLoader(test_data, batch_size=6, shuffle=False, num_workers=0)

#Now using the AlexNet
AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)

AlexNet_model.features[0] = nn.Conv2d(1, 64, kernel_size=(11,11), stride = 1, padding=2)

#Updating the second classifier
AlexNet_model.classifier[4] = nn.Linear(4096,1024)

#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
AlexNet_model.classifier[6] = nn.Linear(1024,10)

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
    val_running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        #print(inputs.shape)
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

    for i, data in enumerate(validloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        #print(inputs.shape)
        # forward + backward + optimize
        output = AlexNet_model(inputs)
        val_loss = criterion(output, labels)

        # print statistics
        val_running_loss += val_loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, val_running_loss / 2000))
            val_running_loss = 0.0

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
model_name = "mnist_6_0.001_0.9_10"
torch.save(AlexNet_model.state_dict(), join("../models/", model_name))