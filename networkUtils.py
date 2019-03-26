# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 09:56:46 2019

@author: gebruiker
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import copy

from sklearn.model_selection import KFold, RepeatedKFold
from random import shuffle
from scriptFunctions import dataSplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
from sklearn import preprocessing

from scriptFunctions import saveRunningTime

# The structure of the model 
# It has 3 hidden layers
# Nodes in the layers can easily be changed, they only need to paired
# Paired = nn.Linear(81, 36) --> nn.Linear(36, 2), the second and first number are paired 
class network(nn.Module):
    # 42, 55, 162
    def __init__(self):
        super(network, self).__init__()
        self.fc1 = nn.Linear(81,108) # Input layer #162 #35
        self.fc2 = nn.Linear(108, 36) # 162, 108
        self.fc3 = nn.Linear(36, 36) # Hidden Layers # 108, 36
        #self.fc4 = nn.Linear(84, 32)
        
        self.fc5 = nn.Linear(36, 2) # Output layer #36, 2
        
        """
        self.linear1 = nn.Linear(in_features=1, out_features=5)
        self.bn1 = nn.BatchNorm1d(num_features=5)
        self.linear2 = nn.Linear(in_features=5, out_features=5)
        """

    # The forward function
    def forward(self, x):  # Input is a 1D tensor
        #y = F.relu(self.bn1(self.linear1(input)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        #x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Old version of the training loop, has no validation.
# Also has the old variables which I'm not going to change.
def train_alt_model(model, criterion, optimizer, scheduler, num_epochs, device, test_and_val_data_dict):
    
    dataloader = test_and_val_data_dict.get("train")
    running_loss = 0.0
    for epoch in range(num_epochs):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)
            #print(labels.item())
            
            inputs = inputs.float()
            labels = labels.float()
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = (model(inputs)).float()
            #print(outputs.item())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            if i % 800 == 799:    # print every 800 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    
    return model


def train_model(model, criterion, optimizer, scheduler, num_epochs, device, dataloaders, KFoldSeed):
    # Time the training starts
    since = time.time()
    
    # Repeated as then it works better with multiple epochs
    rkf = RepeatedKFold(10, num_epochs, KFoldSeed)
    
    # Variables used for the progress, time and accuracy predictions
    totalDataLoopSize = (len(dataloaders["train"]) + len(dataloaders["val"]))*(num_epochs*10)
    dataLoopIndex = 0
    estimateTimeList = []

    # Making a copy of the model
    best_model_wts = copy.deepcopy(model.state_dict())
    snd_best_model_wts = copy.deepcopy(model.state_dict())
    # Best accuracy of the model during training
    best_acc = 0.0
    # 2nd best accuracy
    snd_best_acc = 0.0
    # List of all accuracies
    allAccuracies = []
    
    # Combine the 2 dictionary lists
    completeData = dataloaders["train"] + dataloaders["val"]
    
    # Epoch loop
#    for epoch in range(num_epochs):
#        epochTime = time.time()
        

    for epoch, (trainIndexes, valIndexes) in enumerate(rkf.split(completeData)):
        epochTime = time.time()
        
        trainData = [completeData[index] for index in trainIndexes]
        valData = [completeData[index] for index in valIndexes]

        dataloaders = {'train':trainData, 'val':valData}
        dataset_sizes = {"train": len(dataloaders["train"]), "val": len(dataloaders["val"])}
        
#        if epoch == 1:
#            print("Train size: ",dataset_sizes['train'])
#            print("Val size: ",dataset_sizes['val'])

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data. 
            for inputs, labels in dataloaders[phase]:
                # Moving the data to the device the model in running
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Converting the data to floats
                inputs = inputs.float()
                labels = labels.float()

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = (model(inputs)).float()
                    
                    predictionProb = (sum((outputs.cpu()).detach().numpy()) / len((outputs.cpu()).detach().numpy()))
                    #prediction_prob = sum(output.detach().numpy())
                    #prediction_prob = output.detach().numpy().max()
                    predicted = np.where(predictionProb>0.5,1,0)
                    predictedTens = torch.from_numpy(predicted).long().to(device)
                    
                    _, preds = torch.max(outputs, 0)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    


                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictedTens.data == (labels.long()).data)
                
                dataLoopIndex += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            # Sort the accuracy list in the way the highest numbers are first.
            allAccuracies = sorted(allAccuracies, reverse=True)

            #print('{} Loss: {:.4f} Acc: {:.4f}'.format(
             #   phase, epoch_loss, epoch_acc))

            # Deep copy the models with the best and the second best accuracy
            if phase == 'val':
                if epoch == 0:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif epoch > 0 and epoch_acc.item() > allAccuracies[0]:
                    #print(epoch_acc)
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                elif epoch > 0 and epoch_acc.item() > allAccuracies[1]:
                    #print("snd",epoch_acc)
                    snd_best_acc = epoch_acc
                    snd_best_model_wts = copy.deepcopy(model.state_dict())
#            
#            if phase == 'val' and epoch_acc > best_acc:
#                print(epoch_acc)
#                best_acc = epoch_acc
#                best_model_wts = copy.deepcopy(model.state_dict())
            
#            #Second best model saved aswell
#            if phase == 'val' and epoch > 0:
#                if epoch_acc.item() > allAccuracies[2]:
#                    print(epoch_acc.item() > allAccuracies[2])
#                    print(epoch_acc.item())
#                    print(allAccuracies[2])
#                    snd_best_acc = epoch_acc
#                    snd_best_model_wts = copy.deepcopy(model.state_dict())
                    
            # Add the accuracy to the accuracy list
            allAccuracies.append(epoch_acc.item())
            
           
            
        # Quite useless section which only displays the current progress and estimated time left.
        progressPerc = (dataLoopIndex / totalDataLoopSize) * 100
        #currentElapsedTime = time.time() - since
        elapsedEpochTime = time.time() - epochTime
        estimateTimeList.append(elapsedEpochTime)
        currentEpoch = epoch + 1
        estimateTimeLeft = ((num_epochs*10) - (currentEpoch-1)) * (sum(estimateTimeList) / len(estimateTimeList))
        print("Current progress: {:.2f}% complete, completed in {:.0f}m {:.0f}s. Estimated time left: {:.0f}m {:.0f}s".format(
                progressPerc, elapsedEpochTime // 60, elapsedEpochTime % 60, estimateTimeLeft // 60, estimateTimeLeft % 60), end="\r")

    # Prints the total time it took to train.
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    averageAcc = (sum(allAccuracies) / (len(allAccuracies)))
    print('Best val Acc: {:4f}\n Last Acc: {:4f}\n Average acc: {:4f}'.format(best_acc, epoch_acc, averageAcc))
    #print(allAccuracies)
    
    lastModel = copy.deepcopy(model.state_dict())
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    
    # Used for debugging the running time between cpu/gpu
    #saveRunningTime(time_elapsed, (num_epochs*10), device)
    
    return model, lastModel


# Testing of the model using test data
# Returns a list of the wrongly guessed labels
def testing(testData, model, device):
    
    # Variables needed in the function
    correct = 0
    total = 0
    wrongLabels = []
    
#    Was for ROC-AUC
#    allLabels = None
#    allPreds = None
    
    # Runs the model without gradient calculations as that is not needed for testing.
    # It also saves on memory.
    with torch.no_grad():
        # Looping over the data stored in testData to be tested on the model.
        for dataTensor, labels in testData:
            # Predicting the label (labels) based on the sequence (dataTensor),
            dataTensor, labels = dataTensor.to(device), labels.to(device)
            labels.float()
            output = model(dataTensor.float())
            
            # Predict probability and prediction
            #predictionProb = (sum(output.detach().numpy()) / len(output.detach().numpy()))
            #predictionProb = sum(output.detach().numpy())
            predictionProb = (output.cpu()).detach().numpy().max()
            predicted = np.where(predictionProb>0.5,1,0)
            predictedTens = torch.from_numpy(predicted).long().to(device)
            
           # print(prediction_prob)

            labels = labels.long()
            
            # Counts the number of correct guesses
            correct += (predictedTens.data == labels.data).sum().item()
            
#            Was for ROC-AUC but it gives an error I can't fix     
#            if allLabels is None:
#                allLabels = labels.detach().numpy()
#            else:
#                allLabels = np.append(allLabels, labels.detach().numpy()) #, axis=0)
#                
#            if allPreds is None:
#                allPreds = predictionProb
#            else:
#                allPreds = np.append(allPreds, predictionProb) #, axis=0)

            
            if not predictedTens == labels:
                wrongLabels.append(labels)
                
            total += 1 
            
#    roc_curve gives: TypeError: Singleton array 0 cannot be considered a valid collection.      
#     And seeing as Pytorch doesn't really have a good ROC curve documentation, this isn't finished
#    # ROC-AUC calculation
#    fpr = dict()
#    tpr = dict()
#    rocAuc = dict()
#    
#    # i = index
#    for i in range(total):
#        fpr[i], tpr[i], _ = roc_curve(allLabels[i], allPreds[i])
#        rocAuc[i] = auc(fpr[i], tpr[i])
#        
#    print(rocAuc)
#    
    
    #print(True)
    
#    prediction2 = output.detach().numpy()
#                    
#    #print(prediction2)
#    
#    prediction_prob = prediction2[1]
#    prediction_binary = np.where(prediction_prob>0.5,1,0)
#    print(prediction_binary)
#    _, predicted = torch.max(output, 0)
#    print(predicted)
#    # #Print Area Under Curve
#    false_positive_rate, recall, thresholds = roc_curve(labels, prediction_prob)
#    roc_auc = auc(false_positive_rate, recall)
#    plt.figure()
#    plt.title('Receiver Operating Characteristic (ROC)')
#    plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
#    plt.legend(loc='lower right')
#    plt.plot([0,1], [0,1], 'r--')
#    plt.xlim([0.0,1.0])
#    plt.ylim([0.0,1.0])
#    plt.ylabel('Recall')
#    plt.xlabel('Fall-out (1-Specificity)')
#    plt.show()
#    
    
    
    print('Accuracy of the network on the test data: %1.4f %%' % (
        100 * correct / total))
    return wrongLabels


# Saves the model as file to be used elsewhere
def save_model(model, name):
      torch.save(model.state_dict(), name+".pth")  
 
    
# Initialization of the model and moves it to the specified device
# Adam and SGD are both options to be used as an optimizer,
# SGD is default
def modelInit(device, optimizerName="SGD"):
    # Creating the network/model
    model = network()
    
    # Loss function
    criterion = nn.MSELoss(reduction="sum")
    
    # Which optimizer
    if optimizerName.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)#, amsgrad=True)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
      
    # A scheduler    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    # Moving model to the device (cpu/gpu)
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        
    return criterion, optimizer, scheduler, model  


# Which device gets used to run the model on.
# cpu is default
def getDevice(cudaAllowed=False):
    if cudaAllowed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
        
    print("Running on: %s" % device)
    return device

###