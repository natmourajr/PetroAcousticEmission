"""
    Helpful functions

"""



import numpy as np
from Functions import models

from sklearn.model_selection import train_test_split as tts
import sklearn.preprocessing as preprocessing
from sklearn.metrics import confusion_matrix


def catToSparse(target):
    """
        Converts between a categorical array to maximum sparse codification. Normally used on a target array.

    """
    cat_outputs = -np.ones([target.shape[0],len(np.unique(target))])
    for i,j in enumerate(target):
        cat_outputs[i,int(j)] = 1
    return cat_outputs


def createLSTMInput(data,target, sampleSize, step):
    """
        Takes a Samples x Dimension data input matrix and a Samples x 1 target array and converts both to an LSTM/CNN acceptable format using sampleSize and step (the latter to create overlap between samples).

    """

    if step > sampleSize:
        step = sampleSize
    indexRange = range(0,data.shape[0]-sampleSize+1,step)
    outputData = np.zeros((len(indexRange), sampleSize, data.shape[1]))
    targetTensor = np.zeros((len(indexRange), sampleSize, target.shape[1]))
    outputTarget = np.zeros((len(indexRange), 1))

    for index, k in enumerate(indexRange):
        outputData[index,:,:] = data[k:(k+sampleSize),:]
        targetTensor[index,:,:] = target[k:(k+sampleSize),:]
        classes, count = np.unique(targetTensor[index,:,:],return_counts=True)
        outputTarget[index,0] = classes[np.argmax(count)]

    return outputData, outputTarget
