import pandas as pd

def Slice(data : pd.DataFrame, value : int):
    return (data.iloc[:value], data.iloc[value:])

def Gini(left, right = None):
    if right == None:
        classes = [1, -1]

        size = left.shape[0]
        if size == 0:
            return 1

        score = 0.0
        for c in classes:
            p = left[left[2] == c].shape[0] / size
            score += p * p

        return 1 - score

    lSize = left[-1] + left[1]
    rSize = right[-1] + right[1]
    tSize = rSize + lSize
    l = (left[-1] ** 2 + left[1] ** 2) / lSize / tSize if lSize != 0 else 0
    r = (right[-1] ** 2 + right[1] ** 2) / rSize / tSize if rSize != 0 else 0
        
    return (1 - l - r)

def FindBestGini(datas : list) -> tuple:
    bestGini = float('inf')
    bestVal = 0
    bestJ = 0
    dir = None
    for i in range(2):
        rowI = 0
        data = datas[i]
        left = {-1 : 0, 1 : 0}
        right = {-1 : data[data[2] == -1].shape[0], 1 : data[data[2] == 1].shape[0]}

        for _, row in data.iterrows():
            gini = Gini(left, right)

            if bestGini > gini:
                bestGini = gini
                bestVal = row[i]
                bestJ = rowI
                dir = i

            rowI += 1

            left[row[2]] += 1
            right[row[2]] -= 1

            if bestGini < 0.01:
                break
                    
        if bestGini < 0.01:
            break

    return (bestVal, dir, bestJ)


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
