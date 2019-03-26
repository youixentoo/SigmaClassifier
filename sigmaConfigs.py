# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:21:18 2019

@author: gebruiker
"""
import random


class configs():
    def __init__(self):
        # Configs
        ###########################################################################
        # How much training data to testing data (%)
        self.trainingSplit = 80
        # How much of the training data is validation (%)
        self.validationSplit = 5
        # How much of the total of 49401 None data (%) (35% = len(readySig) * 20)
        # Decimals are possible, though 0.05 is about the minimum
        self.noneDataAmount = 10
        # Number of epochs * 10 (training loops) 
        self.epochs = 5
        # KFold seed for randomness, needs to be an integer
        self.kfoldSeed = random.randint(1,12000)
        # Optimizer name
        # SGD or Adam, SGD is default
        self.optimizerName = "SGD"
        # Model name
        self.modelName = "sigma70ModelFin"
   
    def getTrainsplit(self):
        return self.trainingSplit
     
    def getValidationSplit(self):
        return self.validationSplit
    
    def getNoneDataAmount(self):
        return self.noneDataAmount
     
    def getEpochs(self):
        return self.epochs
    
    def getKFoldSeed(self):
        print("Random seed: %s" % (str(self.kfoldSeed)))
        return self.kfoldSeed
    
    def getOptimizer(self):
        return self.optimizerName
    
    def getModelName(self, useBestModel=True):
        if useBestModel:
            return self.modelName
        else:
            return self.modelName + "_2nd"