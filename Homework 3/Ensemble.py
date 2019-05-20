import numpy as np
import pandas as pd
from numba import jit

@jit(nopython = True)
def Ensemble(prediction : np.array) -> np.array:
    ''' returns array[n][k] of predictions. column i uses trees 1 to i to create prediction '''
    ensemble = np.zeros(prediction.shape)
    for i in range(prediction.shape[0]):
        for k in range(i + 1):
            ensemble[i] += prediction[k]

        ensemble[i][ensemble[i] >= 0] = 1
        ensemble[i][ensemble[i] < 0] = -1
        
    return ensemble

@jit
def CalcError(prediction : np.array, data : np.array):
    results = prediction - data
    return np.count_nonzero(results) / data.shape[0]

train = pd.read_csv('hw3_train.dat', header=None, delimiter=' ')[2].values
test = pd.read_csv('hw3_test.dat', header=None, delimiter=' ')[2].values

trainPredict = np.load('trainPred.npy')
testPredict = np.load('testPred.npy')
'''
trainEnsemble = Ensemble(trainPredict)
np.save(f'trainEnsemb.npy', trainEnsemble)
testEnsemble = Ensemble(testPredict)
np.save(f'testEnsemb.npy', testEnsemble)
'''

trainEnsemble = np.load('trainEnsemb.npy')
testEnsemble = np.load('testEnsemb.npy')

treeCount = 30_000
Eins = []
for i in range(treeCount):
    Eins.append(CalcError(trainEnsemble[i], train))

np.save(f'Eins.npy', Eins)

Eouts = []
for i in range(treeCount):
    Eouts.append(CalcError(testEnsemble[i], test))

np.save(f'Eouts.npy', Eouts)