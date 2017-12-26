import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])

# r = np.multiply(-1, [1, 1])
r = np.apply_along_axis(lambda w: w ** 2, 0, a)
print(r)
