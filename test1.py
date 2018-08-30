import numpy as np
import pickle
f = np.load('data/test.npz')
a = f['index']
for i in range(10):
    print(a[i])

# with open('embedding/glove.6B/glove.6B.300d.txt') as f:
#     print(f.readline())
with open('data/word2index.pkl', 'rb') as f:
    a = pickle.load(f)
    print(a)
'''with open('data/test','r',encoding='utf-8') as f:
    i = 1
    for line in f:
        line = line.strip().split(' ')
        if len(line) == 31:
            print(i)
        i += 1'''

'''a = [1, 2, 3]
pickle_file = open('a.pkl', 'wb')
pickle.dump(a, pickle_file)
with open('a.pkl', 'rb') as f:
    a = pickle.load(f)
print(a[1])'''

'''import nltk
datasets = []
lowercase = True
filename = 'raw_data/wikitext-2/123.txt'
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        if lowercase:
            line = line.lower()
        line = line.strip().split(' ')
        if line[0] == '=' or line[0] == '':
            continue
        else:
            for word in line:
                flag = word.find('@')
                if flag != -1:
                    line.pop(line.index(word))
        datasets.append(line)
print(datasets)'''


'''class CoNLLDataset(object):
    datasets = []
    def __init__(self, filename):
        with open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip('\n')
                line = line.strip(' ').split(' ')
                self.datasets.append(line)'''

'''def add_glove(filename, dim_word): 
    a = ['<start>']
    for i in range(dim_word):
        a.append('0.0')
    with open(filename, 'a+', encoding='utf-8') as f:
        f.write(' '.join(a))
        f.close()'''


'''def get_vocab(datasets):
    vocab = dict()
    vocab['<pad>'] = 0
    vocab['<start>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    i = 4
    for line in datasets:
        for word in line:
            flag = vocab.get(word)
            if flag is None:
                vocab[word] = i
                i += 1
    return vocab'''

'''def load_vocab(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[word] = i
            i += 1
    return d'''

'''def index2words(filename):
    d = dict()
    with open(filename, encoding='utf-8') as f:
        i = 0
        for word in f:
            word = word.strip('\n')
            d[i] = word
            i += 1
    return d'''

'''def write_vocab(filename, vocab):
    result = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
    with open(filename, 'w', encoding='utf-8') as f:
        for word in result:
            f.write(word[0]+'\n')'''


'''def get_trimmed_datasets(filename, datasets, vocab, max_length):
    embeddings = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        sen = np.zeros(max_length)
        sen[0] = vocab['<start>']
        for i in range(max_length-1):
            if i == max_length-2:
                sen[max_length-1] = vocab['eos']
                break
            if i == len(line):
                sen[i+1] = vocab['eos']
                break
            else:
                flag = vocab.get(line[i])
                if flag is None:
                    sen[i+1] = vocab['unk']
                else:
                    sen[i+1] = vocab[line[i]]
        embeddings[k] = sen
        k += 1
    np.savez_compressed(filename, index=embeddings)'''


'''# -*- coding: utf-8 -*-
import argparse
import os
filename_train = '1'
filename_valid = '2'
filename_test = '3'
filename_glove = '4'
parser = argparse.ArgumentParser()
parser.add_argument('--sign', action='store_true', help='sign')
parser.add_argument('--filename', metavar='filename', type=str, nargs='+',
                    help='display filename[filename_train, filename_valid, filename_test, filename_glove](default:*)')
parser.add_argument('--max_length', metavar='NUM', type=int, help='display max_length')
parser.add_argument('--dim_word', metavar='NUM', type=int, help='display dim_word')
args = parser.parse_args()
if args.filename:
    name = ['filename_train', 'filename_valid', 'filename_test', 'filename_glove']
    for i in range(len(args.filename)):
        if args.filename == '*':
            continue
        else:
            if os.path.isfile(args.filename[i]):
                a = name[i] + '=' + '\'' + args.filename[i] + '\''
                exec(a)
            else:
                raise FileNotFoundError('No such file or directory: {}'.format(args.filename[i]))
if args.max_length:
    max_length = args.max_length
if args.dim_word:
    dim_word = args.dim_word
print(args.sign)'''

'''import string
a = string.punctuation
with open('embedding/glove.6B/glove.6B.50d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip().split(' ')
        # if line[0] in a:
            # print(line[0])
        if line[0] == 'A':
            print(line[0])'''
