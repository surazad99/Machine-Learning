from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
iris = load_iris()
test_idx = [0, 50, 100]
# print(iris.target_names)

# training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# making a classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print(clf.predict(test_data))
print(test_target)



