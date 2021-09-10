import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
data, category = dataset.data, dataset.target

print(data)
print(category)

data_train, data_test, category_train, category_test = train_test_split(data, category, test_size=0.2, random_state=1234)


def euclidian_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:

    def __init__(self, k=3):
        self.k = k