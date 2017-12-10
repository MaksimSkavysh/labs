import matplotlib.pyplot as plt
loss = list()
acc = list()
# with open("./out3.txt") as file:
with open("./m3out.txt") as file:
    for line in file:
        if line.__contains__("Momentum"):
            loss_ = "loss: "
            acc_ = "- acc: "
            loss_start_index = line.find(loss_)
            acc_start_index = line.find(acc_)
            acc_before_value_index = acc_start_index + len(acc_)
            acc_end_index = acc_before_value_index + line[acc_before_value_index:].find(" ")
            loss.append(float(line[loss_start_index+len(loss_):acc_start_index].strip()))
            acc.append(float(line[acc_before_value_index:acc_end_index].strip()))

start = 500
# plt.plot(acc[500:])
plt.figure(0)
plt.plot(acc[1:])
plt.ylabel('accuracy')


plt.figure(1)
plt.plot(loss[1:])
plt.ylabel('loss')
plt.show()

