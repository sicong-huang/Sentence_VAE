import numpy as np
import keras

import utils

# decode a code into string sentence
def decode(code, decoder, batch_size, seq_len, bos_idx):
    out = np.ones((batch_size, 1), dtype=np.int32) * bos_idx
    state = code
    predicted = np.empty((batch_size, seq_len), dtype=np.int32)
    for i in range(seq_len):
        out, state = decoder.predict([out, state])
        out = np.argmax(out, axis=-1)
        predicted[:, [i]] = out
    return predicted

if __name__ == '__main__':
    batch_size = 128

    test = utils.load_data('test.npz', 'index', np.int32)

    seq_len = test.shape[1]

    # load pretrained encoder and decoder
    encoder = keras.models.load_model('saved_models/encoder.h5')
    print('encoder loaded from: saved_models/encoder.h5')
    decoder = keras.models.load_model('saved_models/decoder.h5')
    print('decoder loaded from: saved_models/decoder.h5')

    ####---- generation ----####
    # load word2idx and idx2word
    idx2word = utils.load_object('data/index2word.pkl')
    word2idx = utils.load_object('data/word2index.pkl')

    bos_idx = word2idx['<bos>']
    eos_idx = word2idx['<eos>']

    f = open('results/test.txt', 'w')
    for i in range(0, test.shape[0] - batch_size, batch_size):
        code = encoder.predict(test[i: i + batch_size])
        predicted = decode(code, decoder, batch_size, seq_len, bos_idx)
        for seq in predicted:
            result = []
            for index in seq:
                if index == eos_idx:
                    result.append('\n')
                    break
                result.append(idx2word[index])
            result = ' '.join(result)
            f.write(result)
    f.close()
