from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from sklearn.metrics import accuracy_score


def euc(a, b):  # gives distance between two points
    return distance.euclidean(a,b)


class MyClassifier:
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        predictions = []
        for row in x_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.x_train[0])
        best_index = 0
        for i in range(1, len(self.x_train)):
            dist = euc(row, self.x_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


iris = datasets.load_iris()
x = iris.data
y = iris.target

# split total data into train and test data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

#call our own classifier class
my_classifier = MyClassifier()
my_classifier.fit(x_train, y_train)  # provide training data to the classifier
predictions = my_classifier.predict(x_test)

print(accuracy_score(y_test, predictions))
