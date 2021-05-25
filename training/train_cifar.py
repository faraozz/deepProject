"Inspired from https://analyticsindiamag.com/implementing-alexnet-using-pytorch-as-a-transfer-learning-model-in-multi-class-classification/"

import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from os.path import join
from transfer_learning import freeze_last_layers, load_model

if __name__ == "__main__":
    CIFAR_PATH = '../cifar-10-batches-py'
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # parameters
     # name to save model under
    max_epochs = 40
    params = {
        'batch_size': 15,
        'shuffle': True,  # already shuffled
        'num_workers':0
    }
    lrt = 0.001
    momentumt = 0.9
    weight_decayt = 0.0005
    #model_name = "cifartransfer_batch_size"+str(params["batch_size"])+"_epochs"+str(max_epochs)+"_mom"+str(momentumt)+"_weightdec"+str(weight_decayt)+"_lrt"+str(lrt)+"_pretrainedFalse"
    model_name = "cifartransfernorot_batch_size"+str(params["batch_size"])+"_epochs"+str(max_epochs)+"ADAM_pretrainedFalse"
    #Downloading training data
    train_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=True, download=True, transform=transform)
    train_data, validation_data = torch.utils.data.random_split(train_data, [45000, 5000])
    train_size = len(train_data)
    val_size = len(validation_data)

    trainloader = torch.utils.data.DataLoader(train_data, **params)
    validationloader = torch.utils.data.DataLoader(validation_data, **params)
    #Downloading test data
    test_data = torchvision.datasets.CIFAR10(root=CIFAR_PATH, train=False, download=True, transform=transform)
    test_size = len(test_data)

    testloader = torch.utils.data.DataLoader(test_data, **params)

    #Now using the AlexNet
    modelpath = "E:/kth/deep/deepProject/training/models/best models/regular_classification_batch_size128_epochs20_mom0.9_weightdec0.0005_lrt0.001_pretrainedFalse"
    loaded_model_classes = 200  # 200 for imagenet, 4 for rotated imagenet
    AlexNet_model = load_model("alexnet", modelpath, loaded_model_classes, 10)
    AlexNet_model = freeze_last_layers(AlexNet_model)

    #Instantiating CUDA device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #Move the input and AlexNet_model to GPU for speed if available
    AlexNet_model.to(device)

    #optimizer = torch.optim.SGD(AlexNet_model.parameters(), lr=lrt, momentum=momentumt, weight_decay = weight_decayt)
    optimizer = torch.optim.Adam(AlexNet_model.parameters(), lr = lrt)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

    training_loss_list = np.zeros((1, max_epochs))
    training_correct_list = np.zeros((1, max_epochs))
    validation_loss_list = np.zeros((1, max_epochs))
    validation_acc_list = np.zeros((1, max_epochs))






    print("training started")
    for epoch in range(max_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_correct = 0.0
        validation_loss = 0.0
        validation_correct = 0.0

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
            pred_indices = output.argmax(1)
            running_correct += pred_indices.eq(labels).sum().item()
            #if i % 2000 == 1999:    # print every 2000 mini-batches
            #    print('[%d, %5d] loss: %.3f' %
            #          (epoch + 1, i + 1, running_loss / 2000))
                #running_loss = 0.0

        with torch.set_grad_enabled(False):
            for i, data in enumerate(validationloader, 0):
                inputs, labels=data[0].to(device), data[1].to(device)
                labels = torch.squeeze(labels) # remove dimension of size 1

                # Model computations
                optimizer.zero_grad()
                pred = AlexNet_model(inputs)
                loss = criterion(pred, labels)
                validation_loss += loss.item()

                # accuracy
                pred_indices = pred.argmax(1)
                validation_correct += pred_indices.eq(labels).sum().item()


        print("Epoch {}: train_loss = {}, train_acc = {}, val_loss = {}, val_acc = {}".format(epoch, running_loss/train_size, running_correct/train_size, validation_loss/val_size, validation_correct/val_size))
        scheduler.step()
        training_loss_list[0][epoch] = running_loss/train_size
        training_correct_list[0][epoch] = running_correct/train_size
        validation_loss_list[0][epoch] = validation_loss/val_size
        validation_acc_list[0][epoch] = validation_correct/val_size
                                                                                          #total_train_loss/train_size,
                                                                                          #train_correct/train_size)
                                                                                          #total_val_loss/val_size,
                                                                                          #val_correct/val_size))
    print('Finished Training of AlexNet')
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
