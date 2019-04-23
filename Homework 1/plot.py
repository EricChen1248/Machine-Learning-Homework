import sys
import numpy as np
import matplotlib.pyplot as plt
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



trainFile = sys.argv[1]

y, x = loadData(trainFile, False)
size = len(y)

for i in range(size):
    y[i] = 1 if y[i] == 4 else -1


x_val = [z[0] for z in x]
y_val = [z[1] for z in x]

plt.scatter(x_val, y_val, c=y, alpha=0.9  )
plt.show()

