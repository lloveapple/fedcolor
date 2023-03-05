import numpy as np

data = np.load('./data/office/{}_train.pkl'.format("amazon"), allow_pickle=True)

print(data)