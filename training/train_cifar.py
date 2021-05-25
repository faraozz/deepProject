"Inspired from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from os.path import join
from training.transfer_learning import freeze_last_layers, load_model
torch.manual_seed(1)

def save_history(history, session_id="test", savepath=""):
    # save history
    csv_history = ",".join([key for key in history.keys()]) + "\n"
    for i in range(len(history[list(history.keys())[0]])):
        for key in history.keys():
            csv_history += str(history[key][i]) + ","
        csv_history = csv_history[:-1] + "\n"
    with open(join(savepath, session_id + "_data.csv"), "w") as f:
        f.write(csv_history)

CIFAR_PATH = '../cifar-10-batches-py'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# training parameters
batch_size = 128
learning_rate = 0.001
momentum = 0.9
epochs = 25

# Downloading training data
train_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=True, transform=transform)

# Downloading validation data
val_size = 5000
train_size = len(train_data) - val_size
train_ds, val_ds = random_split(train_data, [train_size, val_size])
print(len(train_ds), len(val_ds))

# Downloading test data
test_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True, transform=transform)


# prepare data loaders
# train data
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
# validation data
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
# test data
test_loader = DataLoader(test_data, batch_size*2, num_workers=4, pin_memory=True)

#Now using the AlexNet
modelpath = "batch_size128_epochs35_mom0.9_weightdec0.0005_lrt0.001_pretrainedFalse"
loaded_model_classes = 4  # 200 for imagenet, 4 for rotated imagenet
AlexNet_model = load_model("alexnet", modelpath, loaded_model_classes, 10)
AlexNet_model = freeze_last_layers(AlexNet_model)


#Instantiating CUDA device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Move the input and AlexNet_model to GPU for speed if available
AlexNet_model = AlexNet_model.to(device)

#Loss
criterion = nn.CrossEntropyLoss()

#Optimizer(SGD)
optimizer = optim.SGD(AlexNet_model.parameters(), lr=learning_rate, momentum=momentum)
history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

for epoch in range(epochs):  # loop over the dataset multiple times
    total_train_loss = 0
    total_val_loss = 0
    train_correct = 0
    val_correct = 0
    trainstep, valstep = 0, 0
    # training
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output = AlexNet_model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        # accuracy
        pred_indices = output.argmax(1)
        train_correct += pred_indices.eq(labels).sum().item()

        # print statistics
        total_train_loss += loss.item()
        trainstep += 1
        #if i % 2000 == 1999:    # print every 2000 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        #          (epoch + 1, i + 1, running_loss / 2000))
        #    running_loss = 0.0
    # validation
    with torch.set_grad_enabled(False):
        for i, data in enumerate(val_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            output = AlexNet_model(inputs)
            loss = criterion(output, labels)

            # accuracy
            pred_indices = output.argmax(1)
            val_correct += pred_indices.eq(labels).sum().item()

            # print statistics
            total_val_loss += loss.item()
            valstep += 1
    trainloss = total_train_loss / trainstep
    valloss = total_val_loss / valstep
    trainacc = train_correct / len(train_ds)
    valacc = val_correct / len(val_ds)
    history['epoch'].append(epoch)
    history['train_loss'].append(trainloss)
    history['train_acc'].append(trainacc)
    history['val_loss'].append(valloss)
    history['val_acc'].append(valacc)
    print("epoch {}: training_loss={}, training_acc={}, validation_loss={}, validation_acc={}".format(epoch, trainloss, trainacc, valloss, valacc))

print('Finished Training of AlexNet')

#Testing Accuracy
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = AlexNet_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# Save model
model_name = "transfer_cifar10_{}_{}_{}_{}".format(batch_size, learning_rate, momentum, epochs)
save_history(history, session_id=model_name + ".csv")
torch.save(AlexNet_model.state_dict(), join("", model_name))

