import random
from math import exp
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# currentIris = 'Iris-setosa'
#
# def get_data(address):
#     samples = []
#     labels = []
#     with open(address) as file:
#         for line in file:
#             parts = line.split(",")
#             last = parts[-1]
#             samples.append([float(x) for x in parts[0:-1]])
#             if last.strip() == currentIris:
#                 labels.append(1)
#             else:
#                 labels.append(0)
#     permutation = np.random.permutation(len(samples))
#     # return samples[permutation], labels[permutation]
#     return samples, labels
#
#
# def split_data(samples, labels, waterline=60000):
#     train_samples, train_labels = samples[:waterline], labels[:waterline]
#     test_samples, test_labels = samples[waterline:], labels[waterline:]
#     return train_samples, train_labels, test_samples, test_labels
#
#
# samples, labels = get_data("./data/iris.data.txt")
# print(samples[0])

iris = datasets.load_iris()
target_names = iris.target_names

X = iris.data
y = iris.target

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# help(pca.fit(X).transform)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['red', 'blue', 'green']
lw = 2

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')

plt.show()
