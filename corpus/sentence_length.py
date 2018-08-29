import matplotlib.pyplot as plt
'''length = dict()
with open('data/train', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split(' ')
        flag = length.get(len(line))
        if flag is None:
            length[len(line)] = 1
        else:
            length[len(line)] += 1
# 散点图
x_values = length.keys()
y_values = length.values()
plt.scatter(x_values, y_values, s=10)'''

'''data = dict()
for k, v in length.items():
    key = k//5
    flag = data.get(key)
    if flag is None:
        data[key] = v
    else:
        data[key] += v
data = sorted(data.items(), key=lambda x: x[0], reverse=False)
y = []
for i in data:
    y.append(i[1])
# 柱状图
plt.bar(range(len(y)), y)'''

import numpy as np
import math
data = []
with open('data/train', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split(' ')
        data.append(len(line))
mean = sum(data)/len(data)
var = math.sqrt(np.var(data))
print(round(mean+var))

'''plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.title('sentcece_length', fontsize=12)
plt.xlabel('length', fontsize=10)
plt.ylabel('num', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.show()'''
