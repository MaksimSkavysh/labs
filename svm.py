from svmutil import *
from math import ceil
import matplotlib.pyplot as plt


def cross_validation_split(dataset, n_folds):
    folds_data = list()
    fold_length = ceil(len(dataset)/n_folds)
    for i in range(0, n_folds):
        folds_data.append(dataset[fold_length*i: fold_length*(i + 1)])
    return folds_data


def run_cross_validation(x, y, params_str, n_folds):
    folds_data_x = cross_validation_split(x, n_folds)
    folds_data_y = cross_validation_split(y, n_folds)
    accuracy = 0
    m = None

    for i in range(0, len(folds_data_x)):
        test_fold_x = folds_data_x[i]
        test_fold_y = folds_data_y[i]
        train_x = []
        train_y = []
        for j in range(0, len(folds_data_x)):
            if i != j:
                train_x = train_x + folds_data_x[j]
                train_y = train_y + folds_data_y[j]

        m = svm_train(train_y, train_x, params_str)
        p_label, p_acc, p_val = svm_predict(test_fold_y, test_fold_x, m)
        accuracy = accuracy + p_acc[0]

    accuracy = accuracy/n_folds
    return accuracy, m


y, x = svm_read_problem('./prepared/train_scaled.txt')
y_test, x_test = svm_read_problem('./prepared/test_scaled.txt')


n_folds = 10
k = 10
d_list = [1, 2, 3, 4]
accuracies = []
k_list = []
extr = []

for d in d_list:
    for i in range(-k, k+1, 1):
        c = 2 ** i
        params_str = '-t 1 -d ' + str(d) + ' -q -r ' + str(c)
        print('k: ', 0, ' c: ', c, ' prarms: ', params_str)
        acc, m = run_cross_validation(x, y, params_str, n_folds)
        accuracies.append(acc)
        k_list.append(i)

    plt.figure(d)
    plt.plot(accuracies)
    plt.ylabel('accuracy (d='+ str(d) + ')')
    plt.xlabel('k = [' + str(-k) + ', ' + str(k) + ']')

    max_acc = max(accuracies)
    max_index = accuracies.index(max_acc)
    extr.append((d, accuracies[max_index], k_list[max_index]))
    accuracies = []
    k_list = []


print(extr)
max_val = (0, 0, 0)
for ex in extr:
    if max_val[1] <= ex[1]:
        max_val = ex

print('best parameters (d, acc, k): ', max_val)


# n_folds = 10
# C = 2 ** 5
# d_list = range(1, 6)
# emp_acc = []
# accuracies = []
# for d in d_list:
#     params_str = '-t 1 -d ' + str(d) + ' -q -r ' + str(C)
#     acc, m = run_cross_validation(x, y, params_str, n_folds)
#     accuracies.append(acc)
#     p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
#     emp_acc.append(p_acc[0])


#
# plt.figure(5)
# plt.plot(accuracies)
# plt.ylabel('validation accuracy (C='+ str(C) + ')')
# plt.xlabel('kernel function degree')
#
# plt.figure(6)
# plt.plot(emp_acc)
# plt.ylabel('kernel function degree')


# n_folds = 10
# C = 2 ** 5
# d_list = range(1, 8)
# emp_acc = []
# accuracies = []
# for d in d_list:
#     params_str = '-t 1 -d ' + str(d) + ' -r ' + str(C)
#     m = svm_train(y, x, params_str)
    # p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
    # emp_acc.append(p_acc[0])


plt.show()

