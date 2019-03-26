# -*- coding: utf-8 -*-
"""
Script used to Train a network to predict Sigma70

Runs best in terminal
"""
import networkUtils as NU
from sigmaConfigs import configs
import scriptFunctions as SF
import torch
import os
from sys import argv

def main(args):
    config = configs()
    # Getting the config settings from the config file
    trainingSplit = config.getTrainsplit()
    validationSplit = config.getValidationSplit()
    noneDataAmount = config.getNoneDataAmount() 
    epochs = config.getEpochs()
    KFoldSeed = config.getKFoldSeed()
    optimizerName = config.getOptimizer()
    modelName = config.getModelName()
    modelName2nd = config.getModelName(False)
    
    # Running on cpu or gpu (allowCUDA), default is False
    device = NU.getDevice(False)

    # CPU did it in about 8 min and 30 sec with 50 epochs
    # GPU did it in about 15 min and 5 sec with 50 epochs

    """Start of script"""
    
    # Data inits
    testingData, dataDictTV = SF.dataInitializations(trainingSplit, validationSplit, noneDataAmount)
    
    # Model initialization
    criterion, optimizer, scheduler, model = NU.modelInit(device, optimizerName)
    
    # Training the model
    model, snd_model = NU.train_model(model, criterion, optimizer, scheduler, epochs, device, dataDictTV, KFoldSeed)
    
    #Current working dir
    cwd = os.path.dirname(os.path.realpath(__file__))
    
    # Save the model to be used for testing and such
    torch.save(model.state_dict(), "%s%s%s.pth" % (cwd, os.sep, modelName))
    torch.save(model.state_dict(), "%s%s%s.pth" % (cwd, os.sep, modelName2nd))


if __name__ == "__main__":
    main(argv)