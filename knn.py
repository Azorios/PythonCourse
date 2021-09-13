import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
data, category = dataset.data, dataset.target

# print(data)
# print(category)

data_train, data_test, category_train, category_test = train_test_split(data, category, test_size=0.2,
                                                                        random_state=1234)


def euclidean_distance(x1, x2):
    """
    returns the euclidian distance of two given points x1,x2
    :param x1: point 1
    :param x2: point 2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k, d_train, d_test, c_train, c_test):
        self.k = k
        self.data_train = d_train
        self.data_test = d_test
        self.category_train = c_train
        self.category_test = c_test

    def predict(self, train_point):
        """
        predicts the category of the training point based on the k nearest neighbors to it
        :param data_train: data points used to predict the training points category
        :param category_train: categories of the data points
        :param train_point: point the categories will be predicted for
        :return: the category prediction
        """

        distances = []
        # calculate all euclidean distances to the train point
        for index in range(len(self.data_train)):
            dist = euclidean_distance(data[index], train_point)
            distances.append([dist, index])

        # convert to numpy array and sort from lowest to highest distance
        sort_index = np.argsort(np.array(distances), axis=0)
        #sort_index = np.sort(np.array(distances))
        print(sort_index)

        # get only the k nearest values
        sort_index_k = sort_index[0:self.k]

        # get corresponding data categories
        data_category = np.take(self.category_train, sort_index_k)

        # look for most common category
        prediction = np.bincount(data_category[:, 0]).argmax()

        return prediction

    def knn(self):
        predictions = []
        for point in data_test:
            prediction = self.predict(point)
            predictions.append(prediction)
        return predictions

    def accuracy(self, predictions):
        """

        :param predictions:
        :return:
        """
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == self.category_test[i]:
                correct += 1
        acc = correct / len(self.data_test) * 100
        return acc

    def plot_histogram(self):
        pass





knn_one = KNN(21, data_train, data_test, category_train, category_test)
pred1 = knn_one.knn()
print(pred1)
print(knn_one.category_test)
accuracy = knn_one.accuracy(pred1)
print(accuracy)

fig, ax = plt.subplots()

p1 = ax.scatter(data[:, 0], data[:, 1], c=category, s=25)
plt.title('Data split into three categories')
ax.legend([p1], ['Categories'], scatterpoints=3)
plt.show()

fig2, ax = plt.subplots()
p2 = ax.scatter(data_train[:, 0], data_train[:, 1], c=category_train, s=25)
p2_ = ax.scatter(data_test[:, 0], data_test[:, 1], c='red', s=25)
plt.title('Data split into training and test samples')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.legend([p2, p2_], ['Training Data', 'Test Data'], scatterpoints=3)
plt.show()

#TODO
# was passiert, wenn k gerade ist und die Kategorie nicht eindeutig bestimmt/predicted werden kann
# min k = 3