'''
this test program is split into 2 parts
1. reads in sentences from the file 'sentences.txt' and encode then decode them
2. interpolate between 2 sentences
'''

import numpy as np
import nltk
import keras
import argparse
from nltk.tokenize.treebank import TreebankWordDetokenizer

from corpus.data_utils import sentence2index
import utils

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

# a np.linspace function on vectors
def linspace(start, end, N):
    step = (end - start) / (N - 1)
    return step * np.arange(N)[:, None] + start

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='a program to display the functionality of VAE')
    parser.add_argument('--seqlen', '-s', type=int, default=31,
                        help='the sequence length used for training the VAE (default 31)')
    args = parser.parse_args()

    # load pretrained encoder and decoder
    encoder = keras.models.load_model('saved_models/encoder.h5')
    print('encoder loaded from: saved_models/encoder.h5')
    decoder = keras.models.load_model('saved_models/decoder.h5')
    print('decoder loaded from: saved_models/decoder.h5')

    ####---- generation ----####
    # load word2idx and idx2word
    idx2word = utils.load_object('data/index2word.pkl')
    word2idx = utils.load_object('data/word2index.pkl')

    # generate
    bos_idx = word2idx['<bos>']
    eos_idx = word2idx['<eos>']
    detok = TreebankWordDetokenizer()
    print('===== encode and decode =====')
    with open('sentences.txt', 'r') as f:
        for sent in f:
            orig_sent = sent.strip('\n')
            code = encode(orig_sent, encoder, word2idx, args.seqlen)
            dec_sent = decode(code, decoder, idx2word, args.seqlen + 10, bos_idx, eos_idx, detok)
            print(orig_sent)
            print(dec_sent)
            print()

    print('===== interpolation =====')
    start_sent = 'how is this possible'
    end_sent = 'what i cannot create i do not understand'
    start_code = encode(start_sent, encoder, word2idx, args.seqlen).reshape(-1)
    end_code = encode(end_sent, encoder, word2idx, args.seqlen).reshape(-1)
    all_codes = linspace(start_code, end_code, 10)
    for cod in all_codes:
        print(decode(cod.reshape(1, -1), decoder, idx2word, args.seqlen, bos_idx, eos_idx, detok))
