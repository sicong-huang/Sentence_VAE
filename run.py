import numpy as np
# import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from gru_model_class import ModelStruct
import data_utils

# load data and fit it to batch size
def load_all_data(batch_size):
    train = data_utils.load_data('train.npz', 'index', np.int32)
    valid = data_utils.load_data('valid.npz', 'index', np.int32)
    test = data_utils.load_data('test.npz', 'index', np.int32)
    train = data_utils.fit_batch(train, batch_size)
    valid = data_utils.fit_batch(valid, batch_size)
    test = data_utils.fit_batch(test, batch_size)
    return train, valid, test

def plot_models(vae, encoder, decoder):
    plot_model(vae, to_file='vae.png', show_shapes=True, show_layer_names=True)
    plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
    plot_model(decoder, to_file='decoder.png', show_shapes=True, show_layer_names=True)

def summarize_models(vae, encoder, decoder):
    print('end-to-end VAE')
    vae.summary()
    print('encoder')
    encoder.summary()
    print('decoder')
    decoder.summary()

# encode a sentence string to latent representation
def encode(sentence, encoder, word2idx):
    state = encoder.predict(test[0].reshape(1, -1))
    out = np.array(1).reshape(1, -1)  # initial "<start>" token
    predicted = []
    for _ in range(seq_len):
        out, state = decoder.predict([out, state])
        out = np.argmax(out, axis=-1)
        predicted.append(np.asscalar(out))
    predicted = np.array(predicted)

if __name__ == '__main__':
    batch_size = 128

    train, valid, test = load_all_data(batch_size)  # load training data

    # define model hyper-parameters
    seq_len = train.shape[1]
    embedding_matrix = data_utils.load_data('trimmed_glove.npz', 'embeddings', np.float32)
    latent_size = 64
    batch_shape = (batch_size, seq_len)
    plot = False
    summary = False

    # construct models
    model_struct = ModelStruct(batch_shape, embedding_matrix, latent_size)
    vae = model_struct.assemble_vae_train()
    encoder = model_struct.assemble_encoder_infer()
    decoder = model_struct.assemble_decoder_infer()

    if plot:  # plot when needed
        plot_models(vae, encoder, decoder)
    if summary:  # display model summary when needed
        summarize_models(vae, encoder, decoder)

    # vae.fit(train, train, batch_size=batch_size, epochs=1, shuffle=True, validation_data=(valid, valid))
    # loss, acc = vae.evaluate(test, test, batch_size=batch_size)
    # print('evaluation result')
    # print('loss =', loss, 'accuracy =', acc)
    #####----- 1 problem: accuracy -----######

    ####---- generation ----####
    # load word2idx and idx2word
    idx2word = data_utils.load_object('index2word.pkl')
    # word2idx = data_utils.load_object('word2index.pkl')

    # generate
    test_sent = test[1]

    state = encoder.predict(test_sent.reshape(1, -1))  # encode to latent representation
    out = np.array(1).reshape(1, -1)  # initial "<start>" token
    predicted = []
    for _ in range(seq_len):
        out, state = decoder.predict([out, state])
        out = np.argmax(out, axis=-1)
        predicted.append(np.asscalar(out))
    predicted = np.array(predicted)

    orig_sent = [idx2word[i] for i in test_sent]
    generated_sent = [idx2word[i] for i in predicted]
    print('original test sentence:')
    print(orig_sent)
    print('generated sentence:')
    print(generated_sent)
