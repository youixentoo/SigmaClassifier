# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 16:09:05 2019

@author: gebruiker
"""
import torch
import os
import numpy as np
from random import shuffle

# Saves the time it took to train the model
def saveRunningTime(timeElapsed, epochs, deviceType):
    line = "{} : {:.0f}".format(epochs, timeElapsed)
    if deviceType == "cpu":
        with open("cpuTimes.txt", "a+") as file:
            file.seek(0)
            file.write(line + "\n")
    else:
        with open("cudaTimes.txt", "a+") as file:
            file.seek(0)
            file.write(line + "\n")
            
    file.close()

   
# Reads the data file and returns the useful data     
def fileReading(filename):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(cwd+os.sep+filename) as file:
        return [["-".join([line.split("\t")[0], line.split("\t")[1]]), line.split("\t")[2]] for line in file.readlines()[1::]]


# This function processes the sigma data
# The type of sigma is stored as a string, this isn't very uselful however so it gets
# converted to a list with all the sigmas
# This list is sorted in reverse, that way Sigma70 is first in the list,
# if keepOtherSigs = False (default) only Sigma70 is kept as a label
# if allInUpperCase = True (default) the sequence gets converted to be in uppercase
# The processed data gets returned
def sigDataProcess(sigData, keepOtherSigs=False, allInUpperCase=True):
    processed = []
    for dataLine in sigData:
        sequence = dataLine[1]
        labelsRaw = map(str.strip, (dataLine[0].split("-")[1]).split(","))
        labels = sorted(labelsRaw, reverse=True)
        
        if not keepOtherSigs:
            labels = labels[0]
        
        if allInUpperCase:
            sequence = sequence.upper()
            
        processed.append([sequence, labels])
    return processed
    

# Splits a list into 2 lists
# The splitPercentage is the length the first returned list
# Example:
# splitPercentage = 90
# list1 (len 100) ==> list2 (len 90) + list3 (len 10)
def dataSplit(dataList, splitPercentage):
    dataListLength = len(dataList)
    splitIndex = dataListLength - (dataListLength//(100/splitPercentage))
    splitIndex = int(splitIndex)
    firstListIndexes = dataList[splitIndex:]
    secondListIndexes = dataList[:splitIndex]
    return firstListIndexes, secondListIndexes
    
 
# Converts the sequence to a list of ints as specified in the DNAToIntDict
# Converts the label to an int
# Both then get converted to a tensor, with "float" as datatype 
def convertDNAToIntsAndTorchTensors(dataList, labelsDict):
    DNAToIntDict = {"A": 0, "T": 1, "C": 2, "G": 3}
    convertedData = []
    for seq in dataList:
        # toConvert is the dna sequence
        toConvert = seq[0]
        convertedSeq = [DNAToIntDict.get(nucl) for nucl in toConvert]
        tensorSeqs = torch.from_numpy(np.array(convertedSeq, dtype="float"))
        label = labelsDict.get(seq[1])
        labelTensor = torch.from_numpy(np.array(label, dtype="float"))
        convertedData.append([tensorSeqs, labelTensor])
    
    return convertedData


# Writes the wrongly guessed labels to a file
def writeAwayWrong(wrongLabelsList, labelsDict):
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open("%s%swrongLabelsFile.txt" % (cwd, os.sep), "w") as file:
        for label in wrongLabelsList:
            file.write(labelsDict.get(label.item())+"\n")        
    file.close()
    
    
# Data intialization from the None data and the Sigma data files    
def dataInitializations(trainingSplit, validationSplit, noneDataAmount, printLengths=False):
    # Loading the data from the files
    dataNone = fileReading("genesOutput81.txt")
    dataSig = fileReading("OutputSigData.txt")
    
    # First step of processing of the None and Sigma data
    readyNone = [[data[1], data[0].split("-")[1]] for data in dataNone]
    readySig = sigDataProcess(dataSig, False, True)
    
    # Splitting the None data based on the percentage given in the config file
    noneDataSplit, unused = dataSplit(readyNone, noneDataAmount)
    
    # Splitting based on the amount of training vs testing data 
    noneDataTV, noneTesting = dataSplit(noneDataSplit, trainingSplit)
    sigDataTV, sigTesting = dataSplit(readySig, trainingSplit)
    
    # Splitting based on how much of the training data will be used as validation 
    noneValidation, noneTraining = dataSplit(noneDataTV, validationSplit)
    sigValidation, sigTraining = dataSplit(sigDataTV, validationSplit)
    
    #print(len(noneDataSplit),len(noneTesting),len(noneValidation),len(noneTraining))
    #print(len(readySig),len(sigTesting),len(sigValidation),len(sigTraining))
    
    labelsDict = {"None": 0, "Sigma70": 1}
    
    # Combining the None and Sigma lists into 1 and shuffle them.
    # Shuffling not needed on the test data
    trainingDataList = (noneTraining + sigTraining)
    shuffle(trainingDataList)
    
    validationDataList = (noneValidation + sigValidation)
    shuffle(validationDataList)
    
    testingDataList = (noneTesting + sigTesting)
    
    # Converting the data to floats and tensors to be used by the model
    trainingData = convertDNAToIntsAndTorchTensors(trainingDataList, labelsDict)
    validationData = convertDNAToIntsAndTorchTensors(validationDataList, labelsDict)
    testingData = convertDNAToIntsAndTorchTensors(testingDataList, labelsDict)
    
    # Prints the the lengths of the datasets if set to True
    if printLengths:
        print('#'*30)
        print("Train data length: ",len(trainingDataList))
        print("Val data length: ",len(validationDataList))
        print("Total train length: ",len(trainingDataList)+len(validationDataList))
        print("Test data length: ",len(testingDataList))
        print("Length sigma testdata: ",len(sigTesting), "\nLength none testdata: ",str(len(testingDataList)-len(sigTesting)))
        print('#'*30)
        print()
    
    # Puts the training and validation datasets together in a dictionary
    dataDictTV = {"train": trainingData, "val": validationData}
    
    # Returning the datasets
    return testingData, dataDictTV