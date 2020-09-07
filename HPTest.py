#Emmett Wainwright
#HPTest.py
#Iterate through tests of different neural network configurations
#to find the optimal set of parameters of those given

import json
import itertools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #set tenserflow logging to error or above
import tensorflow as tf
from tensorflow.keras import models, layers, backend
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sys import exit

###############################
### VARIABLES and CALLBACKS ###
###############################

jsonFilePath = "CNNInfo.json"
outFilePath = "HPTest.out"
batchSize = 32
maxNumEpochs = 10
stepsPerEpoch = 6400
regularizer = tf.keras.regularizers.l2(0.01)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy',
    mode='max', verbose=1, save_best_only=True)

###############
### CLASSES ###
###############

#LyrParam Class: Takes name of json import file and reads in params
class LyrParam :
    #Initialize Parameter set as a created neural network
    def __init__(self, numRepsPerTest, cnnNumLayers, cnnAct, cnnChan1, cnnChan2, cnnChan3,
            cnnDrop, cnnBatch, cnnReg, denNumLayers, denNumChannels, denAct):
        self.numRepsPerTest = numRepsPerTest
        self.convNumLayers = cnnNumLayers
        self.convActFunc = cnnAct
        self.convChan1 = cnnChan1
        self.convChan2 = cnnChan2
        self.convChan3 = cnnChan3
        self.convDrop1 = cnnDrop
        self.convDrop2 = cnnDrop
        self.convDrop3 = cnnDrop
        self.convBatchNorm = cnnBatch
        self.convReg1 = None
        self.convReg2 = None
        self.denseNumLayers = denNumLayers
        self.denseNumChannels = denNumChannels
        self.denseActFunc = denAct
        self.returnData = []
        self.bestAccuracy = []
        self.bestEpoch = []
        self.lossAtEpoch = []
        self.hst = None
        
        if cnnReg == True:
            self.convReg1 = regularizer
            self.convReg2 = regularizer
        
        if self.convNumLayers > 0 :
            self.returnData.append("Channels: " + str(self.convChan1)
                + " Dropout: " + str(self.convDrop1)
                + " BatchNorm: " + str(self.convBatchNorm)
                + " Reg: " + str(self.convReg1))
        if self.convNumLayers > 1 :
            self.returnData.append("Channels: " + str(self.convChan2)
                + " Dropout: " + str(self.convDrop2)
                + " BatchNorm: " + str(self.convBatchNorm)
                + " Reg: " + str(self.convReg2))
        if self.convNumLayers > 2 :
            self.returnData.append("Channels: " + str(self.convChan3)
                + " Dropout: " + str(self.convDrop3)
                + " BatchNorm: " + str(self.convBatchNorm))
        
        #Create Neural Network Model
        #testRepetitions = [models.Sequential()] * self.numRepsPerTest
        self.nn = [None] * self.numRepsPerTest
        for x in range(self.numRepsPerTest):
            #Create Neural Network Model
            self.nn[x] = models.Sequential()
            #Build CNN Layers
            self.nn[x].add(layers.Conv2D(self.convChan1, (3, 3),
                activation=self.convActFunc, kernel_regularizer=self.convReg1,
                bias_regularizer=self.convReg1, input_shape=(28, 28, 1)))
            if self.convBatchNorm:
                self.nn[x].add(BatchNormalization())
            self.nn[x].add(layers.MaxPooling2D((2, 2)))
            self.nn[x].add(layers.Dropout(self.convDrop1))
            if self.convNumLayers > 1 :
                self.nn[x].add(layers.Conv2D(self.convChan2, (3, 3),
                    activation=self.convActFunc, kernel_regularizer=self.convReg2,
                    bias_regularizer=self.convReg2))
                if self.convBatchNorm:
                    self.nn[x].add(BatchNormalization())
                self.nn[x].add(layers.MaxPooling2D((2, 2)))
                self.nn[x].add(layers.Dropout(self.convDrop2))
            if self.convNumLayers > 2 :
                self.nn[x].add(layers.Conv2D(self.convChan3, (3, 3),
                    activation=self.convActFunc))
                if self.convBatchNorm:
                    self.nn[x].add(BatchNormalization())
                self.nn[x].add(layers.Dropout(self.convDrop3))
            self.nn[x].add(layers.Flatten())
            #Build Dense Layers
            for i in range(self.denseNumLayers):
                self.nn[x].add(layers.Dense(self.denseNumChannels,
                activation=self.denseActFunc))
            #Build Final Dense Softmax Layer
            self.nn[x].add(layers.Dense(10, activation='softmax'))
    
    #Print out the parameters
    def printParams(self):
        print(self.__dict__)
    
    #Run test of parameter set
    def test(self):
        for x in range(self.numRepsPerTest):
            print("   Testing Round " + str(x+1) + " of " + str(self.numRepsPerTest))
            # Process it all, configure parameters, and get ready to train
            self.nn[x].compile(
                optimizer="rmsprop",             # Improved backprop algorithm
                loss='categorical_crossentropy', # "Misprediction" measure
                metrics=['accuracy']             # Report CCE value as we train
            )
            #Image Augmentation
            #self.nn[x].fit(datagen.flow(train_data, train_labels, batch_size=batchSize),
            #    steps_per_epoch=len(train_data) / batchSize, epochs=maxNumEpochs)
            #Run Test
            self.hst = self.nn[x].fit(datagen.flow(train_data, train_labels,
                batch_size = batchSize),
                steps_per_epoch =(len(train_data)/batchSize), epochs = maxNumEpochs,
                validation_data = (test_data, test_labels), callbacks=[es, mc])
            print("Accuracy: " + str(self.hst.history['val_accuracy']))
            self.bestAccuracy.insert(x, 0) #initialize that array position
            for index, accuracy in enumerate(self.hst.history['val_accuracy']):
                if self.bestAccuracy[x] < accuracy:
                    self.bestAccuracy[x] = accuracy
                    self.bestEpoch.insert(x, index+1)
                    self.lossAtEpoch.insert(x, self.hst.history['val_loss'][index])
            self.returnData.append("Test " + str(x+1)
                + ": Best validatation at epoch " + str(self.bestEpoch[x])
                + " with loss of " + str(self.lossAtEpoch[x])
                + " and accuracy " + str(self.bestAccuracy[x]))
            
        return self.returnData
    
    #Get Data
    def getData(self):
        return self.returnData


#################
### Functions ###
#################

#Load Test Data into list comprehension of LyrParam instances
def setupTests():
    print("Loading from json file")
    jsonFile = open(jsonFilePath, "r") #Open Json reader
    testParams = json.load(jsonFile)
    numRepsPerTest = testParams['numRepsPerTest']
    numCnnLayers = testParams['numCnnLayers']
    numEndingDenseLayers = testParams['numEndingDenseLayers']
    CNNactFunction = testParams['CNNLayers']['actFunction']
    CNNnumChannels = testParams['CNNLayers']['numChannels']
    CNNdropout = testParams['CNNLayers']['dropout']
    CNNbatchNormalization = testParams['CNNLayers']['batchNormalization']
    CNNregularization = testParams['CNNLayers']['regularization']
    DENactFunction = testParams['DenseLayerOptions']['actFunction']
    DENnumChannels = testParams['DenseLayerOptions']['numChannels']
    jsonFile.close() #Close Json reader
    
    #List Comprehension of all parameters to get all permutations
    print("Initializing all permutations of LyrParam")
    allPermutations = [LyrParam(numReps, cnnLayers, cnnAct,
            cnnChan1, cnnChan2, cnnChan3, cnnDrop, cnnBatch,
            cnnReg1, denNumLayers, denNumChannels, denAct)
        for numReps in numRepsPerTest
        for cnnLayers in numCnnLayers
        for cnnAct in CNNactFunction
        for cnnChan1 in CNNnumChannels
        for cnnChan2 in CNNnumChannels
        for cnnChan3 in CNNnumChannels
        for cnnDrop in CNNdropout
        for cnnBatch in CNNbatchNormalization
        for cnnReg1 in CNNregularization
        for denNumLayers in numEndingDenseLayers
        for denNumChannels in DENnumChannels
        for denAct in DENactFunction]
    return allPermutations


###########################################
### Neural Network: Setup and Run Tests ###
###########################################

#Load MNIST Data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

# Reshape training and test data to add an additional dimension of 1 channel
train_data = train_data.reshape((60000, 28, 28, 1))
test_data = test_data.reshape((10000, 28, 28, 1))

# Revise pixel data to 0.0 to 1.0, 32-bit float
train_data = train_data.astype('float32') / 255  # ndarray/scalar op
test_data = test_data.astype('float32') / 255

# Turn 1-value labels to 10-value vectors
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Setup Image Generation
datagen = ImageDataGenerator(
   rotation_range=5,
   shear_range=.2,
   zoom_range=.1,
   width_shift_range=2,
   height_shift_range=2
)

#Setup Tests into LyrParam instances
allConfigurations = []
allConfigurations = setupTests()

#Output to termial number of configurations and list all configurations
print("----------------------------------")
print("Number of test combinations: " + str(len(allConfigurations)))
print("----------------------------------")
print("\nAll CNN test permutations:")
for x in allConfigurations: LyrParam.printParams(x)

#Run Test Loop
for index, eachConfiguration in enumerate(allConfigurations):
    #if index+1 > 3:
        #break
    #Printout info on current test
    print("\nTesting Configuration " + str(index+1)
        + " of " + str(len(allConfigurations)))
    
    #Run Test
    returnData = eachConfiguration.test()
    
    #Output data to file
    with open(outFilePath, "a") as outputFile:
        outputFile.write("\nConfiguration " + str(index+1)
            + " of " + str(len(allConfigurations)) + "\n")
        for line in returnData:
            outputFile.write(str(line) + "\n")

#Variables to keep track of most accurate configuration
overallBestAccuracy = 0
mostAccurateConfiguration = None
epochAccuracyReached = 0

#Search for most accurate configuration among all tests
for configuration in allConfigurations:
    for index, accuracy in enumerate(configuration.bestAccuracy):
        if accuracy > overallBestAccuracy:
            overallBestAccuracy = accuracy
            mostAccurateConfiguration = configuration
            epochAccuracyReached = index+1

#Get the Configuration data for the mosta accurate test
returnData = mostAccurateConfiguration.getData()

#Write to file the details of the most accurate configuration
with open(outFilePath, "a") as outputFile:
    outputFile.write("\n\nHighest accuracy: " + str(overallBestAccuracy))
    outputFile.write("\nHighest accuracy in this configuration: \n")
    for line in returnData:
            outputFile.write(str(line) + "\n")
