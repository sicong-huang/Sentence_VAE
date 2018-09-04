import math
import numpy as np
from corpus.data_utils import get_datasets


def vocabulary(datasets, n):
    v = {}
    for line in datasets:
        for i in range(len(line) - n + 1):  # 逐个单词进行统计
            word = line[i]
            for j in range(n - 1):
                word = word + ' ' + line[i + j + 1]  # 单词连接
            flag = v.get(word)  # 更新字典
            if word == '':
                continue
            if flag is None:
                v[word] = 1
            else:
                v[word] += 1
    return v


'''def Good_Turing(vocab):
    nr = {}
    r = {}
    pr = {}
    num = 0
    # nr
    for key, value in vocab.items():
        num += value
        flag = nr.get(value)
        if flag:
            nr[value] += 1
        else:
            nr[value] = 1
    # r
    for key, value in nr.items():
        flag = nr.get(key+1)
        if flag:
            temp = (key+1)*nr[key+1]/(nr[key]*num)
        else:
            temp = key/num
        r[key] = temp
    # pr
    r0 = nr[1] / num
    # print(nr)
    # print(r)
    # print(pr)
    return r, r0'''


def pp(datasets, test, n):
    for line in datasets:
        line.insert(0, '<bos>')
        line.append('<eos>')
    v1 = vocabulary(datasets, n-1)
    v2 = vocabulary(datasets, n)
    pp = []
    # Good_Turing smoothing
    # pr1, r0_1 = Good_Turing(v1)
    # pr2, r0_2 = Good_Turing(v2)
    for line in test:
        line.insert(0, '<bos>')
        line.append('<eos>')
        num = len(line) - n + 1
        p = 0
        for i in range(len(line)-n+1):
            word1 = ''
            word2 = line[i]
            for j in range(n-1):
                if j == n-2:
                    word1 = word2
                word2 = word2 + ' ' + line[i+j+1]
            flag1 = v1.get(word1)
            if flag1:
                n1 = v1[word1]
            else:
                n1 = 0
            flag2 = v2.get(word2)
            if flag2:
                n2 = v2[word2]
            else:
                n2 = 0
            # add one smoothing
            p += math.log((n2+1)/(n1+len(v1)-2), 2)
            # print('word1 %s' % word1)
            # print('n1 %d' % n1)
            # print('word2 %s' % word2)
            # print('n2 %d' % n2)
            # Good_Turing smoothing
            '''flag1 = pr1.get(n1)
            flag2 = pr2.get(n2)
            if flag1 and flag2:
                p += math.log(pr2[n2] / pr1[n1])
            elif flag1:
                p += math.log(r0_2 / pr1[n1])
            elif flag2:
                p += math.log(pr2[n2] / r0_1)
            else:
                p += math.log(r0_2 / r0_1)'''
            # print(p)
        pp.append(pow(2, -p/num))
        # print(pp)
    result = 1
    for i in pp:
        result *= i
    return pow(result, 1/len(pp))


if __name__ == '__main__':
    datasets = get_datasets('../raw_data/ptb/test', lowercase=True)
    test = get_datasets('../results/test.txt', lowercase=True)
    # test = get_datasets('C:\\F\\python\\keras\\test.txt', lowercase=True)

    result = pp(datasets, test, 2)
    print(result)