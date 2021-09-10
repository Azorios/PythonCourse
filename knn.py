import numpy as np
from sklearn import datasets

dataset = datasets.load_iris()
data, category = dataset.data, dataset.target

print(data)
print(category)


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k