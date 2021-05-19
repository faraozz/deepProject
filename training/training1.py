import numpy as np
import torch
from models.NetworkInNetwork import NiN

#loss function?
#def loss(labels, preds):
#    loss = 0
#    for i in range(0, len(labels)):
#        loss = loss -(1/4)*np.log(np.preds[i]-labels[i])
#    return loss

def trainmodel(trainingdata, validationdata, model, optimizer, criterion, num_epochs):
    #save the best accuracy and model
    best_model = copy.deepcopy(model.state_dict())
    best_validation_acc = 0.0
    #training for epochs
    for epoch in range(0, num_epochs):

        #shuffle the trainingdata
        #np.random.shuffle(trainingdata)

        print("Epoch number: ", epoch, "/", num_epochs)
        #if we are training or validating
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
                data = trainingdata
            else:
                model.eval()
                data = validationdata

            for databatch, labelbatch in data:
                #batchsize
                #set gradients to zero
                optimizer.zero_grad()


                #track history if we are training
                #with torch.set_grad_enabled(phase=='train'):

                #outputs and predictions + calculates loss
                outputs = model(databatch)
                _, preds = torch.max(outputs, 1)
                loss = criterion(labels, preds)

                #backward + optimizer only if in training
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                #calculate running loss and number of correctly classified samples
                running_loss += loss.item() * np.shape(databatch, 0)
                running_classification_accuracy +=torch.sum(preds==label)



            #update learningrate
            #if phase=='train':
            #    scheduler.step()
            #divide by the number of samples.. not sure how this will work either
            epoch_loss = running_loss/len(trainingdata)
            epoch_acc = running_classification_accuracy.double()/len(trainingdata)

            print("Loss: ", epoch_loss)
            print("Accuracy: ", epoch_acc)

            #if the new model has higher validation accuracy, we save this model.
            if phase=='validation' and epoch_acc>best_validation_acc:
                best_acc = epoch_acc
                best_model  = copy.deepcopy(model.state_dict())

    print("Training complete")
    print("Best validation accuracy: ", best_acc)
    model.load_state_dict(best_model_wts)
    return model

model = NiN()
optimizer = torch.optim.SGD(network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


trainmodel(trainingdata, validationdata, model, optimizer, criterion, 10)
