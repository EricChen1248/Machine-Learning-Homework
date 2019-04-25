#%%
import numpy as np
import pandas as pd
import random
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import zero_one_loss, make_scorer

#%%
DATA = 'hw2_lssvm_all.dat'

data = pd.read_csv(DATA, sep=' ', header=None)
data = data.drop(0, axis=1)
data.head()

LABEL = 11

#%%
trainingData = data[:400]
trainingX = trainingData[trainingData.columns[0:-1]]
trainingY = trainingData[trainingData.columns[-1]]
#%%
trainingX.head()

#%%
trainingY.head()

#%%
testingData = data[400:]
testingX = testingData[testingData.columns[0:-1]]
testingY = testingData[testingData.columns[-1]]


#%% 
def CalculatedSampleError(ridge : RidgeClassifier, xs : pd.DataFrame, ys : pd.DataFrame) -> float:
    predictionResults = ridge.predict(xs) * np.array(ys)
    incorrect = np.sum(np.array(predictionResults) < 0)
    return incorrect / len(predictionResults)

def RidgedClassification():
    # the alpha used in sklearn corresponds to the lambda used in our course
    ridge = RidgeClassifier(alpha = lam)
    ridge.fit(trainingX, trainingY)

    Ein = CalculatedSampleError(ridge,trainingX, trainingY)
    Eout = CalculatedSampleError(ridge, testingX, testingY)
    return Ein, Eout
    
def CalculatedSampleErrorAggregation(ridges : list, xs : pd.DataFrame, ys : pd.DataFrame) -> float:
    predictions = []
    for ridge in ridges:
        predictions.append(ridge.predict(xs))
    
    predictionResults = np.sum(predictions, axis=0) * np.array(ys)
    incorrect = np.sum(np.array(predictionResults) < 0)
    return incorrect / len(predictionResults)

def BaggedRidgeClassification():
    ridges = []
    for _ in range(250):
    # the alpha used in sklearn corresponds to the lambda used in our course
        baggedIndex = random.choices(range(400), k=400)
        baggedX = trainingX.iloc[baggedIndex]
        baggedY = trainingY.iloc[baggedIndex]
        ridge = RidgeClassifier(alpha = lam)
        ridge.fit(baggedX, baggedY)
        ridges.append(ridge)

    Ein = CalculatedSampleErrorAggregation(ridges,trainingX, trainingY)
    Eout = CalculatedSampleErrorAggregation(ridges, testingX, testingY)
    return Ein, Eout

#%% [markdown]
## Testing Ridge Regression
#%%
lambdas = [0.05, 0.5, 5, 50, 500]
for lam in lambdas:
    # the alpha used in sklearn corresponds to the lambda used in our course
    Ein, Eout = RidgedClassification()
    print(f"RidgeClassifier regression with lambda: {lam}\n\t\t Ein = {Ein}\tEout = {Eout}")

#%% [markdown]
## Testing Ridge Regression with Bagged Data (250 iters.)

#%%
for lam in lambdas:
    Ein, Eout = BaggedRidgeClassification()
    print(f"BaggedRidgeClassifier regression with lambda: {lam}\n\t\t Ein = {Ein}\tEout = {Eout}")



#%%
