import numpy as np
import nltk
import keras
from nltk.tokenize.treebank import TreebankWordDetokenizer

import data_utils

# encode a sentence string to latent representation
def encode(sentence, encoder, word2idx, seq_len):
    tokens = nltk.word_tokenize(sentence)
    tokens.append('eos')
    sequence = []
    for word in tokens:
        try:
            sequence.append(word2idx[word])
        except KeyError:
            sequence.append(word2idx['unk'])
    if len(sequence) < seq_len:
        sequence = sequence + [word2idx['<pad>']] * (seq_len - len(sequence))
    elif len(sequence) > seq_len:
        sequence = sequence[:seq_len]
    return encoder.predict(np.array([sequence]))

# decode a code into string sentence
def decode(code, decoder, idx2word, seq_len, eos_idx, detok):
    out = np.array(1).reshape(1, -1)  # initial "<start>" token
    state = code
    predicted = []
    for _ in range(seq_len):
        out, state = decoder.predict([out, state])
        out = np.asscalar(np.argmax(out, axis=-1))
        if out == eos_idx:
            break
        predicted.append(out)
    predicted = [idx2word[index] for index in predicted]
    return detok.detokenize(generated_sent)

if __name__ == '__main__':
    seq_len = 32

    # load pretrained encoder and decoder
    encoder = keras.models.load_model('saved_models/encoder.h5')
    print('encoder loaded from: saved_models/encoder.h5')
    decoder = keras.models.load_model('saved_models/decoder.h5')
    print('decoder loaded from: saved_models/decoder.h5')

    ####---- generation ----####
    # load word2idx and idx2word
    idx2word = data_utils.load_object('data/index2word.pkl')
    word2idx = data_utils.load_object('data/word2index.pkl')

    # generate
    eos_idx = word2idx['eos']
    detok = TreebankWordDetokenizer()
    with open('test_sentences.txt', 'r') as f:
        for sent in f:
            orig_sent = sent.strip('\n')
            code = encode(orig_sent, encoder, word2idx, seq_len)
            dec_sent = decode(code, decoder, idx2word, seq_len + 10, eos_idx, detok)
            print(orig_sent)
            print(dec_sent)
            print()
