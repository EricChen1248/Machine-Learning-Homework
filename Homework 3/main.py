from DecisionTree import DecisionTree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import threading

def scale(data):
    data[0] = data[0].values.astype(float) * 10**6
    data[0] = data[0].values.astype(int)
    data[1] = data[1].values.astype(float) * 10**6
    data[1] = data[1].values.astype(int)

def PredictAndCalculate(data, tree):
    wrong = 0
    for _, row in data.iterrows():
        if tree.predict(row) != row[2]:
            wrong += 1
        
    return wrong / data.shape[0]

def plotTuple(tup, name):
    x, y = zip(*tup)
    plt.plot(x, y, label = name)

def main():
    train = pd.read_csv('hw3_train.dat', header=None, delimiter=' ')
    scale(train)
    test = pd.read_csv('hw3_test.dat', header=None, delimiter=' ')
    scale(test)

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

    plotTuple(ins, 'Ein')
    plotTuple(outs, 'Eout')
    plt.legend()
    plt.show()

    return

    from multiprocessing import Process
    from RandomForest import GenerateRandomForest
    treeCount = 30_000
    batchSize = 6
    trees = 500

    for j in range(treeCount // batchSize // trees):
        processes = []
        for i in range(batchSize):
            p = Process(target=GenerateRandomForest, args = (trees, i + j * batchSize))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print("==Completed==")



if __name__ == "__main__":
    main()