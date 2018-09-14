'''
1. Encoder the paragraph.
2. Clustering(k-means, GMM) sentences and finding the core.
3. Decoder the core.
'''
import nltk
import keras
import sys
sys.path.append('../')
from corpus.data_utils import sentence2index
import utils
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from nltk.tokenize.treebank import TreebankWordDetokenizer

# encode a sentence string to latent representation
def encode(sentence, encoder, word2idx, seq_len):
    sequence = sentence2index(sentence, word2idx, seq_len)
    return encoder.predict(np.array([sequence]))


# decode a code into string sentence
def decode(code, decoder, idx2word, seq_len, bos_idx, eos_idx, detok):
    out = np.array(bos_idx).reshape(1, -1)
    state = code
    predicted = []
    for _ in range(seq_len):
        out, state = decoder.predict([out, state])
        out = np.argmax(out, axis=-1)
        if eos_idx == np.asscalar(out):
            break
        predicted.append(np.asscalar(out))
    predicted = [idx2word[index] for index in predicted]
    return detok.detokenize(predicted)


def k_means(x, num):
    kmeans = KMeans(n_clusters=num, random_state=0).fit(x)
    centers = kmeans.cluster_centers_
    return centers


def GMM(x, num):
    gmmModel = GaussianMixture(n_components=num, random_state=0).fit(x)
    y = np.zeros([num, x.shape[1]])
    for i in range(num):
        for j in range(x.shape[1]):
            y[i][j] = gmmModel.means_[i][j]
    return y



if __name__ == '__main__':
    seq_len = 32
    # load pretrained encoder and decoder
    encoder = keras.models.load_model('../saved_models/encoder.h5')
    print('encoder loaded from: saved_models/encoder.h5')
    decoder = keras.models.load_model('../saved_models/decoder.h5')
    print('decoder loaded from: saved_models/decoder.h5')

    ####---- generation ----####
    # load word2idx and idx2word
    idx2word = utils.load_object('../data/index2word.pkl')
    word2idx = utils.load_object('../data/word2index.pkl')

    # input paragraph
    f = open('text.txt', 'r', encoding='utf-8')
    text = f.read()

    # Cut the text into sentences and put them in 'sentences'
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(text)
    print('text')
    print(sentences)

    # encoder
    result = np.zeros([len(sentences), 512])
    for i in range(len(sentences)):
        result[i] = encode(sentences[i], encoder, word2idx, seq_len)

    # K-Means
    # centers = k_means(result, 1)

    # GMM
    centers = GMM(result, 1)

    # decoder
    bos_idx = word2idx['<bos>']
    eos_idx = word2idx['<eos>']
    detok = TreebankWordDetokenizer()

    print('decode')
    for code in centers:
        print(decode(code.reshape(1, -1), decoder, idx2word, seq_len, bos_idx, eos_idx, detok))
