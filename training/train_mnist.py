"Inspired from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from os.path import join
import numpy as np
from transfer_learning import freeze_last_layers, load_model


if __name__ == "__main__":
    MNIST_PATH = '../mnist'

    t = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=(0), std=(1))]
                           )

    #Downloading training data
    dataset = torchvision.datasets.MNIST(root=MNIST_PATH, train=True, download=True, transform=t)
    #print(len(dataset))
    max_epochs = 40
    params = {
        'batch_size': 15,
        'shuffle': True,  # already shuffled
        'num_workers':4
    }

    lrt = 0.001
    momentumt = 0.9
    weight_decayt = 0.0005
    #model_name = "mnisttransfer_batch_size"+str(params["batch_size"])+"_epochs"+str(max_epochs)+"_mom"+str(momentumt)+"_weightdec"+str(weight_decayt)+"_lrt"+str(lrt)+"_pretrainedFalse"
    model_name = "mnisttransfernonrot_batch_size"+str(params["batch_size"])+"_epochs"+str(max_epochs)+"ADAM_pretrainedFalse"

    val_size = 6000
    train_size = len(dataset) - val_size

    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    #print(len(train_data), len(val_data))

    trainloader = torch.utils.data.DataLoader(train_data, **params)

    validloader = torch.utils.data.DataLoader(val_data, **params)

    #Downloading test data
    test_data = torchvision.datasets.MNIST(root=MNIST_PATH, train=False, download=True, transform=t)
    #print(len(test_data))
    testloader = torch.utils.data.DataLoader(test_data, **params)
    test_size = len(test_data)
    train_size = len(train_data)
    val_size = len(val_data)
    print(train_size)
    print(val_size)
    print(test_size)
    #Now using the AlexNet
    #AlexNet_model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=False)

    #AlexNet_model.features[0] = nn.Conv2d(1, 64, kernel_size=(11,11), stride = 1, padding=2)

    #Updating the second classifier
    #AlexNet_model.classifier[4] = nn.Linear(4096,1024)

    #Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
    #AlexNet_model.classifier[6] = nn.Linear(1024,10)
    modelpath = "E:/kth/deep/deepProject/training/models/best models/regular_classification_batch_size128_epochs20_mom0.9_weightdec0.0005_lrt0.001_pretrainedFalse"
    loaded_model_classes = 200  # 200 for imagenet, 4 for rotated imagenet
    AlexNet_model = load_model("alexnet", modelpath, loaded_model_classes, 10)
    AlexNet_model = freeze_last_layers(AlexNet_model)
    #Instantiating CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Move the input and AlexNet_model to GPU for speed if available
    AlexNet_model.to(device)

    #Loss
    criterion = nn.CrossEntropyLoss()

    #Optimizer(SGD)
    #optimizer = optim.SGD(AlexNet_model.parameters(), lr=lrt, momentum=momentumt, weight_decay=weight_decayt)
    optimizer = torch.optim.Adam(AlexNet_model.parameters(), lr = lrt)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    training_loss_list = np.zeros((1, max_epochs))
    training_correct_list = np.zeros((1, max_epochs))
    validation_loss_list = np.zeros((1, max_epochs))
    validation_acc_list = np.zeros((1, max_epochs))

    print("training started")
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0.0
        val_running_loss = 0.0
        val_running_correct = 0.0
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
            pred_indices = output.argmax(1)
            running_correct += pred_indices.eq(labels).sum().item()
            #if i % 2000 == 1999:    # print every 2000 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch + 1, i + 1, running_loss / 2000))
            #    running_loss = 0.0

        with torch.set_grad_enabled(False):
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
                pred_indices = output.argmax(1)
                val_running_correct += pred_indices.eq(labels).sum().item()
                #if i % 2000 == 1999:    # print every 2000 mini-batches
                #    print('[%d, %5d] loss: %.3f' %
                #          (epoch + 1, i + 1, val_running_loss / 2000))
                #    val_running_loss = 0.0
        print("Epoch {}: train_loss = {}, train_acc = {}, val_loss = {}, val_acc = {}".format(epoch, running_loss/train_size, running_correct/train_size, val_running_loss/val_size, val_running_correct/val_size))
        scheduler.step()
        training_loss_list[0][epoch] = running_loss/train_size
        training_correct_list[0][epoch] = running_correct/train_size
        validation_loss_list[0][epoch] = val_running_loss/val_size
        validation_acc_list[0][epoch] = val_running_correct/val_size

    print('Finished Training of AlexNet')
    # Save model
    torch.save({'model_state_dict': AlexNet_model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),'training_loss': training_loss_list, 'training_acc': training_correct_list, 'validation_loss': validation_loss_list, 'validation_acc': validation_acc_list}, join("E:/kth/deep/deepProject/training/models/checkpoints", model_name))

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
#model_name = "mnist_6_0.001_0.9_10"
#torch.save(AlexNet_model.state_dict(), join("../models/", model_name))
