import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import sklearn
import sklearn.linear_model
import sklearn.svm
import sklearn.ensemble
import sklearn.naive_bayes
import sklearn.svm
import sklearn.neighbors


arr = [1, 2, 3, 4, 6, 7]

np.random.shuffle(arr)

clf = sklearn.ensemble.RandomForestClassifier(n_estimators=2, n_jobs=-1)

print([1, 2] + [4, 5])

# help(sklearn.neighbors.KNeighborsClassifier)