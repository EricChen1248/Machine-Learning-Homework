#%%
import numpy as np
import pandas as pd


#%%
TRAIN = 'hw2_adaboost_train.dat'
TEST = 'hw2_adaboost_test.dat'

train = pd.read_csv(TRAIN, sep=' ', header=None)
train = train.astype(float)
trainX = train[train.columns[:-1]]
trainY = train[train.columns[-1]]
#trainY = [float(y.replace('\n')) for y in trainY]

test = pd.read_csv(TEST, sep=' ', header=None)
testX = test[test.columns[:-1]]
testY = test[test.columns[-1]]

#%%
def GenerateStump(xs, feature):
    stumps = []
    for r in range(len(xs) - 1):
        stumps.append((xs[feature][r] + xs[feature][r + 1]) / 2)
    
    return stumps

def CalculateError(xs, ys, thresh, feature):
    wrong = 0
    for i in range(len(xs)):
        x = xs[feature][i]
        y = ys[i]

        if x < thresh:
            pred = -1
        else:
            pred = +1
        
        # if stump is wrong, add u
        if y * pred < 0:
            wrong += abs(y)

    rate = wrong / sum([abs(y) for y in ys])
    if rate > 0.5:
        return (+1, rate)
    else:
        return (-1, 1 - rate)

def BestStump(xs : pd.DataFrame, ys : pd.DataFrame, features : list, stumps : list) -> tuple:
    BestError = float('inf')
    BestS = None
    for feature in features:
        for stump in stumps[feature]:
            s, error = CalculateError(xs, ys, stump, feature)

            if error < BestError:
                BestS = (s, feature, stump)
                BestError = error

    return (BestS, BestError)

def UpdateY(xs, ys, g, d):
    s, feature, thresh = g
    for i in range(len(xs)):
        # incorrect
        if xs[feature][i] * s < thresh:
            ys[i] = ys[i] * d
        # correct
        else:
            ys[i] = ys[i] / d
            
def GenerateG(iterations, xs, ys, features, stumps):
    Gs = []
    for _ in range(iterations):
        g, e = BestStump(xs, ys, features, stumps)
        d = np.sqrt((1-e) / e)
        UpdateY(xs, ys, g, d)
        Gs.append((g, np.log(d)))
    
    return Gs

#%%  
features = [0, 1]
stumps = [GenerateStump(trainX, feature) for feature in features]
iterations = 10

G = GenerateG(iterations, trainX, trainY, features, stumps)

#%%
print(G)


def FindBestThresh(xs: pd.DataFrame, ys: pd.DataFrame, feature: int) -> tuple:
    BestErrorPos = abs(ys.query(ys.columns[0] <= 0).sum())
    BestErrorNeg = ys.query(ys.columns[0] > 0).sum()
    BestThreshPos = None
    BestThreshNeg = None
    curErrorPos = BestErrorPos
    curErrorNeg = BestErrorNeg
    for i in range(len(xs) - 1):
        y = ys[i]

        # if y on pos side of NEG, NEG has more error if y < 0 (- <0 == increase)
        # # else less error if y > 0 (- >0 == decrease)
        curErrorNeg -= y


        # if y on pos side of POS, POS has less error if y < 0 (+ <0 == decrease)
        # # else more error if y > 0 (+ >0 == increase)
        curErrorPos += y

        if curErrorPos < BestErrorPos:
            BestErrorPos = curErrorPos
            BestThreshPos = i

        if curErrorNeg < BestErrorNeg:
            BestErrorNeg = curErrorNeg
            BestThreshNeg = i

    # Convert to midpoints
    if BestThreshPos < len(xs) - 1:
        BestThreshPos = (xs[feature][i] + xs[feature][i + 1]) / 2
    else:
        BestThreshPos = -float('inf')
    
    if BestThreshNeg < len(xs) - 1:
        BestThreshNeg = (xs[feature][i] + xs[feature][i + 1]) / 2
    else:
        BestThreshNeg = -float('inf')

    # Return better of NEG or POS
    if BestErrorPos < BestErrorNeg:
        return (1, BestThreshPos, BestErrorPos)
    else:
        return (-1, BestThreshNeg, BestErrorNeg)
