from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
x = iris.data
y = iris.target

# split total data into train and test data sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)

# using KneighbourClssifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(x_train, y_train)  # provide training data to the classifier
predictions = my_classifier.predict(x_test)

# find accuracy
print(accuracy_score(y_test, predictions))

