import tensorflow as tf
import keras

from keras.layers import Input, Dense, Lambda, GRU, TimeDistributed, Embedding
from keras.models import Model
import keras.backend as K

class ModelStruct:
    def __init__(self, batch_shape, embedding_matrix, latent_size, bos_idx):
        # self.__check_inputs(batch_shape, embedding_dim, latent_size, vocab_size)  # check inputs are valid

        self.batch_shape = batch_shape  # shape of input data
        self.latent_size = latent_size
        vocab_size, self.embedding_dim = embedding_matrix.shape
        self.batch_size, self.seq_len = batch_shape
        self.embedded_shape = (self.batch_size, self.seq_len, self.embedding_dim)  # shape of embedded data
        self.bos = K.constant(bos_idx, dtype='int32')

        ### embedding layer ###
        self.embedding_layer = Embedding(vocab_size, self.embedding_dim,\
                weights=[embedding_matrix], trainable=False)

        ### encoder components ###
        self.encode_gru = GRU(self.latent_size, name='encoder_gru')
        self.mean_output = Dense(self.latent_size, name='mean_dense')  # output layer for mean
        self.var_output = Dense(self.latent_size, name='variance_dense')  # output layer for variance

        ### sampling layer component ###
        self.sampling_layer = Lambda(self.__sampling, name='sampling_layer')

        ### decoder components ###
        self.convert_layer = Lambda(self.__convert, name='convert_layer')
        self.decode_gru = GRU(self.latent_size, return_sequences=True, return_state=True, name='decoder_gru')
        self.output_dense = Dense(vocab_size, activation='softmax', name='decoder_output')

    # sampling function used by sampling layer
    def __sampling(self, args):
        sample_mean, sample_log_std = args
        epsilon = K.random_normal(shape=(self.batch_size, self.latent_size))
        return sample_mean + K.exp(sample_log_std) * epsilon

    # a function used in decoder input
    # to convert training data into RNN input form
    def __convert(self, data):
        bos_tensor = self.embedding_layer(K.ones(shape=(self.batch_size, 1), dtype='int32') * self.bos)
        data = K.concatenate([bos_tensor, data], axis=1)
        data = K.slice(data, start=(0, 0, 0), size=self.embedded_shape)
        return data

    # return an end-to-end VAE for training
    def assemble_vae_train(self):
        encode_in = Input(batch_shape=self.batch_shape, dtype='int32', name='vae_in')
        embedded = self.embedding_layer(encode_in)
        hidden_state = self.encode_gru(embedded)
        mean = self.mean_output(hidden_state)
        log_std = self.var_output(hidden_state)
        z = self.sampling_layer([mean, log_std])
        decode_in = self.convert_layer(embedded)
        decode_states, _ = self.decode_gru(decode_in, initial_state=z)
        decode_out = TimeDistributed(self.output_dense)(decode_states)
        vae = Model(encode_in, decode_out, name='VAE')

        vae_loss_fn = self.__vae_loss_helper(encode_in, decode_out, mean, log_std)
        accuracy_fn = self.__accuracy_helper(encode_in, decode_out)
        vae.compile(optimizer='adam', loss=vae_loss_fn, metrics=[accuracy_fn])

        # add VAE loss
        # reconstruction_loss = K.mean(K.sum(K.sparse_categorical_crossentropy(encode_in, decode_out), axis=1))
        # kl_loss = -0.5 * K.mean(1 + log_std - K.square(mean) - K.exp(log_std))
        # vae_loss = reconstruction_loss + kl_loss
        # vae.add_loss(vae_loss)

        return vae

    # a helper function that returns a loss function used to compute gradient
    def __vae_loss_helper(self, encode_in, decode_out, mean, log_std):
        def vae_loss(y_true, y_pred):
            reconstruction_loss = K.mean(K.sum(K.sparse_categorical_crossentropy(encode_in, decode_out), axis=1))
            kl_loss = -0.5 * K.mean(1 + log_std - K.square(mean) - K.exp(log_std))
            return reconstruction_loss + kl_loss
        return vae_loss

    # a helper function that returns a metric function to monitor accuracy
    def __accuracy_helper(self, encode_in, decode_out):
        def accuracy(y_true, y_pred):
            pred = tf.argmax(decode_out, axis=-1, output_type=tf.int32)
            correct_count = tf.count_nonzero(tf.equal(pred, encode_in), axis=-1, dtype=tf.float32)
            return tf.div(correct_count, self.seq_len)
        return accuracy

    '''
    Returns an encoder for inference, the components of this encoder is the same
    as the ones that made up the vae.

    The encoder, when called on .predict() method, returns the mean of code that
    represents input data.
    '''
    def assemble_encoder_infer(self):
        encode_in = Input(shape=(self.seq_len,), name='encoder_in')
        embedded = self.embedding_layer(encode_in)
        hidden_state = self.encode_gru(embedded)
        mean = self.mean_output(hidden_state)
        return Model(encode_in, mean)

    '''
    Returns a decoder for inference, the components of this decoder is the same
    as the ones that made up the vae.

    The decoder's .predict() method takes 2 parameters 'decoder input' and
    'initial state' respectively.

    The decoder, when called on .predict() method, returns a probability distribution
    of the next word over all words in vocabulary.

    In this version, embedding matrix has to be predefined, and it must contain
    <start> token with its embedding being all zeros.
    '''
    def assemble_decoder_infer(self):
        init_state = Input(shape=(self.latent_size,), name='decoder_initial_state')
        decode_in = Input(shape=(1,), dtype='int32', name='decoder_in')
        embedded = self.embedding_layer(decode_in)
        decode_states, hidden_state = self.decode_gru(embedded, initial_state=init_state)
        # decode_out = Reshape()
        decode_out = TimeDistributed(self.output_dense)(decode_states)
        return Model([decode_in, init_state], [decode_out, hidden_state], name='decoder')

    # helper method used to check inputs are valid
    # throws corresponding exceptions when expectations are not met
    def __check_inputs(self, batch_shape, embedding_dim, latent_size, vocab_size):
        batch_shape_type = type(batch_shape)
        batch_shape_len = len(batch_shape)
        embedding_dim_type = type(embedding_dim)
        latent_size_type = type(latent_size)
        vocab_size_type = type(vocab_size)

        if batch_shape_type != tuple:
            raise TypeError('expect "batch_shape" to be type tuple, instead got {}'.format(batch_shape_type))
        elif batch_shape_len != 2:
            raise ValueError('expect "batch_shape" to have length == 2, instead got {}'.format(batch_shape_len))
        elif not all(i > 0 for i in batch_shape):
            raise ValueError('all elements in batch_shape must be greater than 0, instead got {}'.format(batch_shape))
        elif embedding_dim_type != int:
            raise TypeError('expect "embedding_dim" to be type int, instead got {}'.format(embedding_dim_type))
        elif embedding_dim <= 0:
            raise ValueError('expect "embedding_dim" to be greater than 0, instead got {}'.format(embedding_dim))
        elif latent_size_type != int:
            raise TypeError('expect "latent_size" to be type int, instead got {}'.format(latent_size_type))
        elif latent_size <= 0:
            raise ValueError('expect "latent_size" to be greater than 0, instead got {}'.format(latent_size))
        elif vocab_size_type != int:
            raise TypeError('expect "vocab_size" to be type int, instead got {}'.format(vocab_size_type))
        elif vocab_size <= 0:
            raise ValueError('expect "vocab_size" to be greater than 0, instead got {}'.format(vocab_size))
