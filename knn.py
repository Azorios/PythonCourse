import numpy as np
import matplotlib as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
data, category = dataset.data, dataset.target

#print(data)
#print(category)

data_train, data_test, category_train, category_test = train_test_split(data, category, test_size=0.2, random_state=1234)

class KNN:
    def __init__(self, k, data_train, data_test, category_train,
                 category_test):
        self.k = k
        self.data_train = data_train
        self.data_test = data_test
        self.category_train = category_train
        self.category_test = category_test
    def euclidian_distance(self, x1, x2):
        """
        returns the euclidian distance of two given points x1,x2
        :param x1: point 1
        :param x2: point 2
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    def predict(self, data, data_category, train_point):
        """
        predicts the categorie of the training point based on the k nearest neighbors to it
        :param data: data points used to predict the training points category
        :param data_category: categories of the data points
        :param train_point: point the categories will be predicted for
        :return: the category prediction
        """
        distances = []
        # calculate all euclidian distances to the train point
        for index in range(len(data)):
            dist = self.euclidian_distance(data[index],train_point)
            distances.append([dist,index])
        # convert to numpy array
        distances = np.array(distances)
        #print(distances)
        sort_index = np.argsort(distances, axis=0)
        # get only the k nearest values
        sort_index = np.delete(sort_index[:self.k],obj=1, axis=1)
        # get corresponding data categories
        data_category = np.take(data_category, sort_index)
        # look for most common category
        prediction = np.bincount(data_category[:,0]).argmax()
        return prediction
    def knn_algorithm(self):
        self.predict(self.data_train, self.category_train, self.data_test[0])
    def plot_histogram(self):
        pass

knn_one = KNN(5, data_train, data_test, category_train, category_test)
knn_one.knn_algorithm()