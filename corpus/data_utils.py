import numpy as np
import pickle
import nltk
import os
from scipy.stats import truncnorm

# check whether the given 'filename' exists
# raise a FileNotFoundError when file not found
def file_check(filename):
    for name in filename:
        if os.path.isfile(name) is False:
            raise FileNotFoundError('No such file or directory: {}'.format(name))


def dim_check(filename, dim_word):
    with open(filename, 'r', encoding='utf-8') as f:
        line = f.readline()
        line = line.strip().split(' ')
        num = len(line)
        if num != dim_word + 1:
            raise ValueError('dimension of input file "filename" must agree with dim_word. Found filename dimension to be {} and \
                             dim_word to be {}'.format(num - 1, dim_word))

'''def get_datasets_1(filename, lowercase):
    datasets = []
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
        if lowercase:
            text = text.lower()
        sents = nltk.sent_tokenize(text)
        for sent in sents:
            words = nltk.word_tokenize(sent)
            datasets.append(words)
    return datasets'''


def get_datasets(filename, lowercase):
    datasets = []
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
    return datasets


'''def get_datasets_ptb(filename):
    datasets = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            line = line.strip('\n')
            line = line.strip(' ').split(' ')
            datasets.append(line)
    return datasets'''


def seq_len(datasets):
    data = []
    for line in datasets:
        data.append(len(line))
    mean = sum(data) / len(data)
    v = np.std(data)
    result = mean + v
    result = int(round(result))
    return result


def get_train_vocab(dataset):
    vocab = set()
    for line in dataset:
        for word in line:
            vocab.add(word)
    return vocab


def get_glove_vocab(filename):
    vocab = set()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split(' ')
            vocab.add(line[0])
    return vocab


def word2index(train_words, glove_vocab):
    words = train_words & glove_vocab
    words = list(words)
    vocab = dict()
    vocab['<pad>'] = 0
    vocab['<bos>'] = 1
    vocab['<eos>'] = 2
    vocab['<unk>'] = 3
    i = 4
    for word in words:
        flag = vocab.get(word)
        if flag is None:
            vocab[word] = i
            i += 1
    return vocab


def index2word(vocab):
    index = []
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=False)
    for i in vocab:
        index.append(i[0])
    return index


def glove_embedding(filename_glove, filename_trimmed_glove, dim_word, vocab):
    embeddings = dict()
    with open(filename_glove, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            if word in vocab.keys():
                embedding = [float(x) for x in line[1:]]
                embeddings[vocab[word]] = embedding
        for i in range(4):
            np.random.seed(i)
            embedding = truncnorm.rvs(-2, 2, size=dim_word)
            embeddings[i] = embedding
    embeddings = sorted(embeddings.items(), key=lambda x: x[0], reverse=False)
    embeddings_array = np.zeros((embeddings[-1][0]+1, dim_word))
    for i in embeddings:
        embeddings_array[i[0]] = i[1]
    np.savez_compressed(filename_trimmed_glove, embeddings=embeddings_array)


def write_vocab(filename, vocab):
    with open(filename, 'wb') as f:
        pickle.dump(vocab, f)


# cut word
def sentence2index(sentence, vocab, max_length):
    sentence = sentence.lower()
    words = nltk.word_tokenize(sentence)
    result = sent2index(words, vocab, max_length)
    return result


# with open('data/word2index.pkl', 'rb') as f:
#     vocab = pickle.load(f)
# eg.sentence = ['no', 'it', 'was', 'n't', 'black', 'monday']
def sent2index(sentence, vocab, max_length):
    result = np.zeros(max_length)
    for i in range(max_length):
        if i == len(sentence):
            result[i] = vocab['<eos>']
            break
        else:
            flag = vocab.get(sentence[i])
            if flag is None:
                result[i] = vocab['<unk>']
            else:
                result[i] = vocab[sentence[i]]
    return result


def get_trimmed_datasets(filename, datasets, vocab, max_length):
    embeddings = np.zeros([len(datasets), max_length])
    k = 0
    for line in datasets:
        sen = sent2index(line, vocab, max_length)
        embeddings[k] = sen
        k += 1
    np.savez_compressed(filename, index=embeddings)


# with open('data/index2word.pkl', 'rb') as f:
#     vocab = pickle.load(f)
def index2sentence(index, vocab):
    result = []
    for i in index:
        if i == 0:
            break
        else:
            result.append(vocab[i])
    return ' '.join(result)

