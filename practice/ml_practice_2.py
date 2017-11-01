import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
#features are sepal and petal measurements
print(iris.feature_names)
#targets are what we want classified, setosa, versicolor, virginica
print(iris.target_names)
test_idx = [0, 50, 100]

#Training data
train_target = np.delete(iris.target, test_idx) 
train_data= np.delete(iris.data, test_idx, axis=0)

#Testing Data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

print("Size of training data is",str(len(train_data)) ," and size of testing data is: ", str(len(test_data)))
#create a decision tree classifier and train it
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print("Test target data: ",test_target)
print("Prediction: ", clf.predict(test_data))
