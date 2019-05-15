import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue


def scale(data):
    data[0] = data[0].values.astype(float) * 10**6
    data[0] = data[0].values.astype(int)
    data[1] = data[1].values.astype(float) * 10**6
    data[1] = data[1].values.astype(int)
    
train = pd.read_csv('hw3_train.dat', header=None, delimiter=' ')
scale(train)
test = pd.read_csv('hw3_test.dat', header=None, delimiter=' ')
scale(test)

def Gini(group : pd.DataFrame) -> float:
    classes = [1, -1]

    size = group.shape[0]
    if size == 0:
        return 1

    score = 0.0
    for c in classes:
        p = group[group[2] == c].shape[0] / size
        score += p * p

    return 1 - score

def GiniGroup(groups : list) -> float:
    count = sum([group.shape[0] for group in groups])

    gini = 0.0
    for group in groups:
        size = group.shape[0]
        if size == 0:
            continue

        gini += Gini(group) * (size / count)

    return gini

def Slice(data : pd.DataFrame, value : int):
    return (data.iloc[:value], data.iloc[value:])

def FindBestGini(datas : list) -> tuple:
    bestGini = float('inf')
    bestVal = 0
    bestJ = 0
    dir = None
    for i in range(2):
        rowI = 0
        data = datas[i]
        for _, row in data.iterrows():
            cut = Slice(data, rowI)
            
            gini = GiniGroup(cut)

            if bestGini > gini:
                bestGini = gini
                bestVal = row[i]
                bestJ = rowI
                dir = i

            rowI += 1
            if bestGini == 0:
                break
                    
        if bestGini == 0:
            break

    return (bestVal, dir, bestJ)

def PredictAndCalculate(data, tree):
    wrong = 0
    for _, row in data.iterrows():
        if tree.predict(row) != row[2]:
            wrong += 1
        
    return wrong / data.shape[0]

class DecisionTree:
    def __init__(self, datas, height = 1, parent = None):
        self.datas = datas
        self.parent = parent
        self.__height = height

    def __cut(self, ref, data):
        index = set(ref[0]['index'].tolist())
        return (data[data['index'].isin(index)], data[~data['index'].isin(index)])

    def learn(self, unLearned):
        self.G = FindBestGini(self.datas)
        G = self.G
        cuts = [None, None]
        chosen = G[1]
        cut = Slice(self.datas[chosen], G[2])
        cuts[chosen] = cut

        other = (chosen + 1) % 2
        cuts[other] = self.__cut(cut, self.datas[other])
 
        if Gini(cuts[chosen][0]) == 0:
            # left side, value column, first item
            self.leftTree = int(cuts[chosen][0][2].iloc[0])
        else:
            self.leftTree = DecisionTree((cuts[0][0], cuts[1][0]), self.__height + 1, self)
            unLearned.append(self.leftTree)


        if Gini(cuts[chosen][1]) == 0:
            self.rightTree = int(cuts[chosen][1][2].iloc[0])
        else:
            self.rightTree = DecisionTree((cuts[0][1], cuts[1][1]), self.__height + 1, self)
            unLearned.append(self.rightTree)


    def predict(self, data):
        if data[self.G[1]] < self.G[0]:
            return self.__predict(self.leftTree, data)

        return self.__predict(self.rightTree, data)

    def __predict(self, branch, data):
        if isinstance(branch, int):
            return branch
        
        return branch.predict(data)

    def getTreeHeight(self):
        lHeight = self.__height if isinstance(self.leftTree, int) else self.leftTree.getTreeHeight()
        rHeight = self.__height if isinstance(self.rightTree, int) else self.rightTree.getTreeHeight()
        return max(lHeight, rHeight)    

    def __getMost(self):
        d = self.datas[0]
        count1 = d[d[2] == 1].shape[0]
        if count1 > d.shape[0] - count1:
            return 1
        else:
            return -1

    def prune(self, height):
        if height == self.__height:
            if isinstance(self.leftTree, DecisionTree):
                self.leftTree = self.leftTree.__getMost()
            if isinstance(self.rightTree, DecisionTree):
                self.rightTree = self.rightTree.__getMost()
        else:
            if isinstance(self.leftTree, DecisionTree):
                self.leftTree.prune(height)
            if isinstance(self.rightTree, DecisionTree):
                self.rightTree.prune(height)

# Single Tree
trainX = train.sort_values(by=0).reset_index()
trainY = train.sort_values(by=1).reset_index()

dTree = DecisionTree((trainX, trainY))
unLearned = [dTree]

ins = []
outs= []
iter = 0
while len(unLearned) > 0:
    tree = unLearned.pop()
    tree.learn(unLearned)
    print(iter, end='\r')
    iter += 1

height = dTree.getTreeHeight()

Ein = PredictAndCalculate(train, dTree)
ins.append((height, Ein))
Eout = PredictAndCalculate(test, dTree)
outs.append((height, Eout))

print(f"Height: {height}")
print(f"Ein:\t{Ein}")
print(f"Eout:\t{Eout}")

# Pruning
for h in reversed(range(1, height)):
    dTree.prune(h)
    print(f"Height: {h}")

    Ein = PredictAndCalculate(train, dTree)
    ins.append((h, Ein))
    Eout = PredictAndCalculate(test, dTree)
    outs.append((h, Eout))

    print(f"Ein:\t{Ein}")
    print(f"Eout:\t{Eout}")

def plotTuple(tup, name):
    x, y = zip(*tup)
    plt.plot(x, y, label = name)

plotTuple(ins, "Ein")
plotTuple(outs, "Eout")
plt.legend()
plt.show()


def GenTree(q):
    pass

treeCount = 100 #30_000
# Random Forest
sampleSize = int(train.shape[0] * 0.8)
trees = []
print("Generating Random Forest")
for i in range(treeCount):
    print(i + 1, end='\r')
    sample = train.sample(sampleSize)
    sampleX = sample.sort_values(by=0).reset_index()
    sampleY = sample.sort_values(by=1).reset_index()
    
    dTree = DecisionTree((sampleX, sampleY))

    trees.append(dTree)
    unLearned = [dTree]

    while len(unLearned) > 0:
        tree = unLearned.pop()
        tree.learn(unLearned)


def ForestPredict(data : pd.DataFrame, trees : list) -> np.array:
    predictions = np.empty((data.shape[0], treeCount))
    for i, row in data.iterrows():
        for j in range(treeCount):
            predictions[i][j] = trees[j].predict(row)

    return predictions

def Ensemble(prediction):
    ensemble = np.empty((prediction.shape[0], prediction.shape[1]))
    for k in range(prediction.shape[1]):
        for i in range(prediction.shape[0]):
            predict = 0
            for j in range(k + 1):
                predict += prediction[i][j]
        
        ensemble[i][k] = -1 if predict < 0 else 1
    
    return ensemble

trainPredict = ForestPredict(train, trees)
testPredict = ForestPredict(test, trees)

trainEnsemble = Ensemble(trainPredict)
testEnsemble = Ensemble(testPredict)

def CalcError(prediction, data):
    wrong = 0
    for i, row in data.iterrows():
        if prediction != row[2]:
            wrong += 1
        
    return wrong / data.shape[0]

Eins = []
for i in range(treeCount):
    Eins.append(CalcError(trainEnsemble[:, i], train))
