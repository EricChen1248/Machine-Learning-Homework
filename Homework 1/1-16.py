import sys
import os
import math
import lib.libsvm.python.svm as svm
import lib.libsvm.python.svmutil as svmutil
import numpy as np
import random

def loadData(filePath : str, xAsDict = True, dataLimit = float('inf')):
	from ast import literal_eval
	ys = []
	xs = []
	count = 0
	with open(filePath, 'r') as f:
		for row in f:
			count += 1
			if count > dataLimit:
				break

			r = list(filter(None, row[:-1].split(' ')))
			y = int(float(r[0]))
			if xAsDict:
				x = literal_eval("{" + ','.join(r[1:]) + "}")
			else:
				x = np.array(list(map(float, r[1:])))
			ys.append(y)
			xs.append(x)

	return ys, np.array(xs)

CS = [-2, -1, 0, 1, 2]
DIGIT = 0
widths = []

def kernel(x1, x2):
	x = x1 - x2
	return math.exp(-80 * x.dot(x))


if __name__ == "__main__":
	print("Learning to recognize digit", DIGIT)
	trainFile = sys.argv[1]

	# Loading Data
	y, x = loadData(trainFile, False)
	size = len(y)
	for i in range(size):
		y[i] = -1 if y[i] == DIGIT else 1

	# End Load Data
	bestGammas = []
	for i in range(100):
		C = 0.1
		print("Iteration:", i)

		tx = []
		ty = []

		nx = []
		ny = []
		chosen = set(random.sample(range(size), 1000))
		for j in range(size):
			if j in chosen:
				tx.append(x[j])
				ty.append(y[j])
			else:
				nx.append(x[j])
				ny.append(y[j])

		bestE = float('inf')
		bestGamma = 0
		for g in [-2, -1, 0, 1, 2]:
			gamma = 10**g
			train = svmutil.svm_train(ny, nx,'-s {0} -t {1} -c {2} -d 2 -g {3} -q'.format(0, 2, C, gamma))
			print("Testing:", g, end = ' ')
			pLab, pAcc, pVal = svmutil.svm_predict(ty, tx, train)
			if 100 - pAcc[0] < bestE:
				bestE = 100 - pAcc[0]
				bestGamma = g

		print("Best gamma:", bestGamma)
		bestGammas.append(bestGamma)


	import matplotlib.pyplot as plt
	
	plt.hist(bestGammas, bins = [-2, -1, 0, 1, 2, 3], align='left', edgecolor='black')
	plt.xticks([-2, -1, 0, 1, 2])
	plt.show()
	