import sys
import os
import math
import lib.libsvm.python.svm as svm
import lib.libsvm.python.svmutil as svmutil
import numpy as np

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

weights = []

CS = [-5,-3,-1,1,3]
DIGIT = 0
if __name__ == "__main__":
	print("Learning to recognize digit", DIGIT)
	trainFile = sys.argv[1]
	testFile = sys.argv[2]

	# Loading Data
	y, x = loadData(trainFile, False)
	size = len(y)

	for i in range(size):
		y[i] = -1 if y[i] == DIGIT else 1

	testY, testX = loadData(testFile, False)
	testSize = len(testY)
	for i in range(testSize):
		testY[i] = -1 if testY[i] == DIGIT else 1

	# End Load Data

	for c in CS:
		C = 10**c
		print("Using C with value", C)
		print("Training model on " + trainFile)
		train = svmutil.svm_train(y, x,'-s {0} -t {1} -c {2} -d 2 -g 80 -q'.format(0, 0, C))

		rho = train.rho[0]
		svs = train.get_SV()
		coeff = train.get_sv_coef()
		w = [0, 0]
		for (sv, co) in zip(svs, coeff):
			w[0] += sv[1] * co[0]
			w[1] += sv[2] * co[0]
			
		weights.append(w)
		print("Weights:", w[0], w[1])
		print("Testing model on " + testFile)
		pLab, pAcc, pVal = svmutil.svm_predict(testY, testX, train)
		print()
	
	
	for i in range(len(weights)):
		weights[i] = weights[i][0] ** 2 + weights[i][1] ** 2

	import matplotlib.pyplot as plt
		
	plt.plot(CS, weights)
	plt.show()