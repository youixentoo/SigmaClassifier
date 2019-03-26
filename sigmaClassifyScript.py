# -*- coding: utf-8 -*-
"""
Script used to Test a network to predict Sigma70
"""
import os
import torch
import torch.backends.cudnn as cudnn
import networkUtils as NU
import scriptFunctions as SF
from sigmaConfigs import configs
import numpy as np

from sys import argv
   
def main(args):
    config = configs()
    # Getting the config settings from the config file
    trainingSplit = config.getTrainsplit()
    validationSplit = config.getValidationSplit()
    noneDataAmount = config.getNoneDataAmount()
    
    # Running on cpu or gpu, False=cpu
    device = NU.getDevice(False)
    
    # Use the best model from the training or the second best one.
    useBestModel = False

    # Get the model name
    modelName = config.getModelName(useBestModel)

    """Start of script"""
    
    # Dictionary to convert the labels to the type
    labelsDict = {0: "None", 1: "Sigma70"}
    
    # Create and load in the model created by the training script
    cwd = os.path.dirname(os.path.realpath(__file__))
    model = NU.network()
    model.load_state_dict(torch.load("%s%s%s.pth" % (cwd, os.sep, modelName)))
    model.eval()
    
    # Move model to the device (cpu/gpu)
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    
    # Data initialization
    testingData, unused = SF.dataInitializations(trainingSplit, validationSplit, noneDataAmount)
    
    # Testing the model with the test data
    # wrongLabels contains all the wrong guesses
    wrongLabels = NU.testing(testingData, model, device)
    
    # Saves the wrong guesses in an external file
    SF.writeAwayWrong(wrongLabels, labelsDict)
    
    # Counts how many times the model guesses both options wrong
    wrongArrays = np.unique(wrongLabels, return_counts=True)    
    
    #wrongLabelsDict = dict(zip(wrongArrays[0], wrongArrays[1]))
    
    # Print the outcomes
    print('#'*30)
    try:
        print("Wrong None: ",str(wrongArrays[1][0]))
    except Exception:
        print("Wrong None: 0?")
    try:
        print ("Wrong Sigma: ",str(wrongArrays[1][1])) 
    except Exception:
        print("Wrong Sigma: 0?")
    print("Wrong labels: ", len(wrongLabels))
    print("Total test data: ", len(testingData))
    print('#'*30)
    #print(wrongLabels)

    #print(model)


if __name__ == "__main__":
    main(argv)