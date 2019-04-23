#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
def CalculatedSampleError(ridge : Ridge, xs : pd.DataFrame, ys : pd.DataFrame) -> float:
    prediction = ridge.predict(xs)
    
    correct = 0
    incorrect = 0
    for z in zip(ys, prediction):
        if z[0] * z[1] > 0:
            correct += 1
        else:
            incorrect += 1

    return incorrect / (correct + incorrect)

#%%
lambdas = [0.05, 0.5, 5, 50, 500]

for lam in lambdas:
    # the alpha used in sklearn corresponds to the lambda used in our course
    ridge = Ridge(alpha = lam)
    ridge.fit(trainingX, trainingY)
    scorer = make_scorer(zero_one_loss)

    Ein = CalculatedSampleError(ridge,trainingX, trainingY)
    Eout = CalculatedSampleError(ridge, testingX, testingY)

    print(f"Ridge regression with lambda: {lam}\n\t\t Ein = {Ein}\tEout = {Eout}")

#%%
len(testingX)

#%%
