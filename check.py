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


def s(a=0, b=1):
    return a+b

o = {'a': 2, 'b': 3}

print(s(**o))

# help(sklearn.neighbors.KNeighborsClassifier)