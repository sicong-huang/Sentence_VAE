import numpy as np
import argparse
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt

import model
import utils

def parse():
    parser = argparse.ArgumentParser(description='program to train a sequence VAE')
    parser.add_argument('--batch', '-b', type=int, default=256,
                        help='batch size for training (default 256)')
    parser.add_argument('--epochs', '-e', type=int, default=20,
                        help='number of training epochs to perform (default 20)')
    parser.add_argument('--latent', '-l', type=int, default=256,
                        help='number of dimensions in latent representation (default 256)')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='plot models')
    parser.add_argument('--summary', '-s', action='store_true',
                        help='display model summary')
    parser.add_argument('--history', '-hi', action='store_true',
                        help='plot training history')
    return parser.parse_args()

# load data and fit it to batch size
def load_all_data(batch_size):
    train = utils.load_data('train.npz', 'index', np.int32)
    valid = utils.load_data('valid.npz', 'index', np.int32)
    test = utils.load_data('test.npz', 'index', np.int32)
    train = utils.fit_batch(train, batch_size)
    valid = utils.fit_batch(valid, batch_size)
    test = utils.fit_batch(test, batch_size)
    return train, valid, test

def plot_models(vae, encoder, decoder):
    plot_model(vae, to_file='images/vae.png', show_shapes=True, show_layer_names=True)
    plot_model(encoder, to_file='images/encoder.png', show_shapes=True, show_layer_names=True)
    plot_model(decoder, to_file='images/decoder.png', show_shapes=True, show_layer_names=True)

def summarize_models(vae, encoder, decoder):
    print('end-to-end VAE')
    vae.summary()
    print('encoder')
    encoder.summary()
    print('decoder')
    decoder.summary()

def plot_history(history):
    history['epochs'] = [i for i in range(1, len(history['loss']) + 1)]  # add epoch indexing

    fig = plt.figure(figsize=(10, 4))
    loss_ax = fig.add_subplot(1, 2, 1)
    loss_ax.plot('epochs', 'loss', data=history)
    loss_ax.plot('epochs', 'val_loss', data=history)
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.set_title('Loss')
    loss_ax.legend(['train', 'validation'])

    acc_ax = fig.add_subplot(1, 2, 2)
    acc_ax.plot('epochs', 'accuracy', data=history)
    acc_ax.plot('epochs', 'val_accuracy', data=history)
    acc_ax.set_xlabel('epoch')
    acc_ax.set_ylabel('loss')
    acc_ax.set_title('Accuracy')
    acc_ax.legend(['train', 'validation'])
    fig.savefig('images/train_hist.png')

if __name__ == '__main__':
    args = parse()

    train, valid, test = load_all_data(args.batch)  # load training data

    bos_idx = 1  # '<bos>'' has index 1
    seq_len = train.shape[1]
    batch_shape = (args.batch, seq_len)
    embedding_matrix = utils.load_data('trimmed_glove.npz', 'embeddings', np.float32)

    # construct models
    model_struct = model.ModelStruct(batch_shape, embedding_matrix, args.latent, bos_idx)
    vae = model_struct.assemble_vae_train()
    encoder = model_struct.assemble_encoder_infer()
    decoder = model_struct.assemble_decoder_infer()

    if args.plot:  # plot when needed
        plot_models(vae, encoder, decoder)
    if args.summary:  # display model summary when needed
        summarize_models(vae, encoder, decoder)

    # train
    hist = vae.fit(train, train, batch_size=args.batch, epochs=args.epochs,
                   shuffle=True, validation_data=(valid, valid))
    loss, acc = vae.evaluate(test, test, batch_size=args.batch)
    print('evaluation result')
    print('loss =', loss, 'accuracy =', acc)

    if args.history:  # plot training history if needed
        plot_history(hist.history)

    # save
    encoder.save('saved_models/encoder.h5')
    print('encoder saved at: saved_models/encoder.h5')
    decoder.save('saved_models/decoder.h5')
    print('decoder saved at: saved_models/decoder.h5')

    #####----- 1 problem: accuracy -----######
