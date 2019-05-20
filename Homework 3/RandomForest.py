import pandas as pd
import numpy as np
from DecisionTree import DecisionTree
from numba import jit


def scale(data):
    data[0] = data[0].values.astype(float) * 10**6
    data[0] = data[0].values.astype(int)
    data[1] = data[1].values.astype(float) * 10**6
    data[1] = data[1].values.astype(int)

def GenTree(tree):
    unLearned = [tree]
    while len(unLearned) > 0:
        tree = unLearned.pop()
        tree.learn(unLearned)

@jit
def CalcError(prediction : np.array, data : np.array):
    results = prediction - data
    return np.count_nonzero(results) / data.shape[0]

@jit
def Prediction(tree, data) -> np.array:
    prediction = np.empty(data.shape[0])
    for i, row in data.iterrows():
        prediction[i] = tree.predict(row)

    return prediction

def CalcEGins(treeCount, trainPredict, data, id):
    egins = np.empty(treeCount)
    for i in range(treeCount):
        egins[i] = CalcError(trainPredict[i], data)

    np.save(f'egins{id}.npy', egins)

def GenerateRandomForest(treeCount, id):
    train = pd.read_csv('hw3_train.dat', header=None, delimiter=' ')
    scale(train)
    test = pd.read_csv('hw3_test.dat', header=None, delimiter=' ')
    scale(test)
    
    trainPredict = np.empty((treeCount, train.shape[0]))
    testPredict = np.empty((treeCount, test.shape[0]))
    sampleSize = int(train.shape[0] * 0.8)
    print(f"Generating Random Forest {id}")
    for i in range(0, treeCount):
        print(i + 1, end='\r')

        sample = train.sample(sampleSize)
        sampleX = sample.sort_values(by=0).reset_index()
        sampleY = sample.sort_values(by=1).reset_index()
        
        dTree = DecisionTree((sampleX, sampleY))
        GenTree(dTree)
        trainPredict[i] = Prediction(dTree, train)
        testPredict[i] = Prediction(dTree, test)

    np.save(f'trainPred{id}.npy', trainPredict)
    np.save(f'testPred{id}.npy', testPredict)

    CalcEGins(treeCount, trainPredict, train[2].values, id)