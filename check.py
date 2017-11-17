import numpy as np

a = [(1.0, 3.157530680769389), (2.0, 1.6278820596099703), (2.0, 0.6082762530298217), (3.0, 2.095232683975696), (1.0, 3.295451410656816)]
sort_order = lambda x: x[-1]
a.sort(key=sort_order)
# a = np.sort(a)
# help(np.sort)
print(a)

