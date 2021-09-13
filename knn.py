import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


def euclidean_distance(x1, x2):
    """
    returns the euclidean distance of two given points x1,x2
    :param x1: point 1
    :param x2: point 2
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, k):
        self.k = k
        self.data = self.category = self.data_train = self.data_test = self.category_train = self.category_test = None

    def fit(self):
        dataset = datasets.load_wine()
        self.data, self.category = dataset.data, dataset.target
        self.data_train, self.data_test, self.category_train, self.category_test = train_test_split(self.data,
                                                                                                    self.category,
                                                                                                    test_size=0.2,
                                                                                                    random_state=1234)

    def knn(self):
        self.fit()
        predictions = []

        for test_point in self.data_test:
            prediction = self.predict(test_point)
            predictions.append(prediction)

        self.accuracy(predictions)
        self.plot_data()

    def predict(self, test_point):
        """
        predicts the category of the training point based on the k nearest neighbors to it
        :param data_train: data points used to predict the training points category
        :param category_train: categories of the data points
        :param test_point: point the categories will be predicted for
        :return: the category prediction
        """

        distances = []
        # calculate all euclidean distances to the train point
        for index in range(len(self.data_train)):
            dist = euclidean_distance(self.data_train[index], test_point)
            distances.append([dist, index])

        # convert to numpy array and sort from lowest to highest distance
        sort_index = np.argsort(np.array(distances), axis=0)
        print(sort_index)

        # get only the k nearest values
        sort_index_k = sort_index[0:self.k]

        # get corresponding data categories
        data_category = np.take(self.category_train, sort_index_k)

        # look for most common category
        prediction = np.bincount(data_category[:, 0]).argmax()

        return prediction

    def accuracy(self, predictions):
        """
        calculates accuracy of predictions by checking if the predictions equals the categories of the test samples
        :param predictions:
        :return:
        """
        #####################
        correct = 0
        for i in range(len(predictions)):
            if predictions[i] == self.category_test[i]:
                correct += 1
        acc = correct / len(self.data_test) * 100
        print(acc)

    def plot_data(self):
        fig, ax = plt.subplots()
        ax.scatter(self.data[:, 0], self.data[:, 1], c=self.category, edgecolor='black', s=35)
        plt.title('Data split into three categories')
        plt.show()

        fig2, ax = plt.subplots()
        ax.scatter(self.data_train[:, 0], self.data_train[:, 1], c=self.category_train, edgecolor='black', s=35)
        p = ax.scatter(self.data_test[:, 0], self.data_test[:, 1], c='red', edgecolor='black', s=35)
        plt.title('Data split into training and test samples')
        ax.legend([p], ['Test Data'])
        plt.show()

    def plot_histogram(self):
        pass


if __name__ == '__main__':
    knn = KNN(20)
    knn.knn()


#TODO
# was passiert, wenn k gerade ist und die Kategorie nicht eindeutig bestimmt/predicted werden kann
