import numpy as np
import nltk
import keras
from nltk.tokenize.treebank import TreebankWordDetokenizer

from corpus.data_utils import sentence2index
import utils

# encode a sentence string to latent representation

####==== proprocess input sentences the same way =====####
def encode(sentence, encoder, word2idx, seq_len):
    # tokens = nltk.word_tokenize(sentence)
    # tokens.append('<eos>')
    # sequence = []
    # for word in tokens:
    #     try:
    #         sequence.append(word2idx[word])
    #     except KeyError:
    #         sequence.append(word2idx['<unk>'])
    # if len(sequence) < seq_len:
    #     sequence = sequence + [word2idx['<pad>']] * (seq_len - len(sequence))
    # elif len(sequence) > seq_len:
    #     sequence = sequence[:seq_len]
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
    seq_len = 32

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
    with open('sentences.txt', 'r') as f:
        for sent in f:
            orig_sent = sent.strip('\n')
            code = encode(orig_sent, encoder, word2idx, seq_len)
            dec_sent = decode(code, decoder, idx2word, seq_len + 10, bos_idx, eos_idx, detok)
            print(orig_sent)
            print(dec_sent)
            print()

    print('interpolation:')
    start_sent = 'i want to be like this man'
    end_sent = 'there is no way that you can make it'
    start_code = encode(start_sent, encoder, word2idx, seq_len).reshape(-1)
    end_code = encode(end_sent, encoder, word2idx, seq_len).reshape(-1)
    all_codes = linspace(start_code, end_code, 10)
    for cod in all_codes:
        print(decode(cod.reshape(1, -1), decoder, idx2word, seq_len, bos_idx, eos_idx, detok))
