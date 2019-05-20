import numpy as np
import os

egins = np.load(f'egins0.npy')
trainPred = np.load(f'trainPred0.npy')
testPred = np.load(f'testPred0.npy')

for i in range(1, 60):
    egins = np.concatenate([egins, np.load(f'egins{i}.npy')])
    trainPred = np.concatenate([trainPred, np.load(f'trainPred{i}.npy')])
    testPred = np.concatenate([testPred, np.load(f'testPred{i}.npy')])
    os.remove(f'egins{i}.npy')
    os.remove(f'trainPred{i}.npy')
    os.remove(f'testPred{i}.npy')

np.save('egins.npy', egins)
np.save('trainPred.npy', trainPred)
np.save('testPred.npy', testPred)