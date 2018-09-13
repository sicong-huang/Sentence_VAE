# Sentence length histogram

import matplotlib.pyplot as plt
data = []
with open('../raw_data/ptb/train', encoding='utf-8') as f:
    for line in f:
        line = line.strip('\n').split(' ')
        data.append(len(line))
plt.hist(data, bins=40, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.title('sentcece_length', fontsize=12)
plt.xlabel('length', fontsize=10)
plt.ylabel('num', fontsize=10)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.show()
