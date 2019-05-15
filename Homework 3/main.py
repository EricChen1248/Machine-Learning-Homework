import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('hw3_train.dat', header=None, delimiter=' ')
test = pd.read_csv('hw3_test.dat', header=None, delimiter=' ')

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

def Slice(data : pd.DataFrame, index: int, value : int):
    return (data[data[index] <= value], data[data[index] > value])

def FindBestGini(data : pd.DataFrame) -> tuple:
    bestGini = float('inf')
    bestVal = 0
    dir = None
    
    for _, row in data.iterrows():
        cut = Slice(data, 0, row[0])

        gini = GiniGroup(cut)

        if bestGini > gini:
            bestGini = gini
            bestVal = row[0]
            dir = 0

        cut = Slice(data, 1, row[1])

        gini = GiniGroup(cut)

        if bestGini > gini:
            bestGini = gini
            bestVal = row[1]
            dir = 1
        
        if bestGini == 0:
            break

    return (bestVal, dir)

def PredictAndCalculate(data, tree):
    wrong = 0
    for _, row in data.iterrows():
        if tree.predict(row) != row[2]:
            wrong += 1
        
    return wrong / data.shape[0]

class DecisionTree:
    def __init__(self, data, height = 1, parent = None):
        self.data = data
        self.parent = parent
        self.__height = height

    def learn(self, unLearned):
        self.G = FindBestGini(self.data)
        cut = Slice(self.data, self.G[1], self.G[0])

        if Gini(cut[0]) == 0:
            # left side, value column, first item
            self.leftTree = int(cut[0][2].iloc[0])
        else:
            self.leftTree = DecisionTree(cut[0], self.__height + 1, self)
            unLearned.append(self.leftTree)


        if Gini(cut[1]) == 0:
            self.rightTree = int(cut[1][2].iloc[0])
        else:
            self.rightTree = DecisionTree(cut[1], self.__height + 1, self)
            unLearned.append(self.rightTree)


    def predict(self, data):
        if data[self.G[1]] <= self.G[0]:
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
        count1 = self.data[self.data[2] == 1].shape[0]
        if count1 > self.data.shape[0] - count1:
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
dTree = DecisionTree(train)
unLearned = [dTree]

ins = []
outs= []
while len(unLearned) > 0:
    tree = unLearned.pop()
    tree.learn(unLearned)

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

treeCount = 100 #30_000
# Random Forest
sampleSize = int(train.shape[0] * 0.8)
trees = []
print("Generating Random Forest")
for i in range(treeCount):
    print(i + 1, end='\r')
    sample = train.sample(sampleSize)
    dTree = DecisionTree(sample)

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
