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
	for c in CS:
		C = 10**c
		print("Using C with value", C)
		print("Training model on " + trainFile)
		train = svmutil.svm_train(y, x,'-s {0} -t {1} -c {2} -d 2 -g 80 -q'.format(0, 2, C))

		svs = train.get_SV()
		coef = train.get_sv_coef()
		w = 0
		fsv = None
		fc = None
		for (sv, c) in zip(svs, coef):
			if c[0] > 0 and c[0] < C:
				fsv = np.array([sv[1], sv[2]])
				fc = c
				break
		
		for (sv, c) in zip(svs, coef):
			w += c[0] * kernel(np.array([sv[1], sv[2]]), fsv)

		widths.append(w)
		print(w)
		print("\n")
		
	
	import matplotlib.pyplot as plt
		
	plt.plot(CS, widths)
	plt.show()