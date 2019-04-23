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

CS = [-5,-3,-1,1,3]
DIGIT = 4
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
	accs = []
	for c in CS:
		C = 10**c
		print("Using C with value", C)
		print("Training model on " + trainFile)
		train = svmutil.svm_train(y, x,'-s {0} -t {1} -c {2} -d 2 -g 80 -q'.format(0, 1, C))

		print("Testing model on " + trainFile)
		pLab, pAcc, pVal = svmutil.svm_predict(y, x, train)
		accs.append(100 - pAcc[0])

		print("\n")
		
	
	import matplotlib.pyplot as plt
		
	plt.plot(CS, accs)
	plt.show()