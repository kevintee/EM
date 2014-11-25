import numpy as np

data = []
with open('./gaussmix.csv') as f:
    for line in f:
        data.append([float(x) for x in line.strip().split(',')])

data = np.asarray(data)
means = [np.mean(x) for x in data.T]
variance = sum([(x-means)**2 for x in data])/200
