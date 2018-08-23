import numpy as np
import keras
# import matplotlib.pyplot as plt
# from keras.utils.vis_utils import plot_model
from gru_model_class import ModelStruct
import data_utils

batch_size = 32

# load data and fit it to batch size
train = data_utils.load_data('train')
valid = data_utils.load_data('valid')
test = data_utils.load_data('test')
train = data_utils.fit_batch(train, batch_size)
valid = data_utils.fit_batch(valid, batch_size)
test = data_utils.fit_batch(test, batch_size)

# define model hyper-parameters
seq_len = train.shape[1]
embedding_dim = 128
latent_size = 64
batch_shape = (batch_size, seq_len)
vocab_size = int(np.max(train) + 1)

# construct models
model_struct = ModelStruct(batch_shape, embedding_dim, latent_size, vocab_size)
vae = model_struct.assemble_vae_train()
# encoder = model_struct.assemble_encoder_infer()
# decoder = model_struct.assemble_decoder_infer()

# plot_model(vae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
'''
def accuracy(epoch, logs):
    pred = vae.predict(valid, batch_size=batch_size)
    pred_argmax = np.argmax(pred, axis=-1)
    correct_count = np.sum(np.equal(valid, pred_argmax))
    acc = correct_count / (seq_len * valid.shape[0])
    print('After epoch', epoch, 'accuracy =', acc)
'''
# pred_before = vae.predict(test)
# acc_before = accuracy(test, pred_before)
#####----- 2 problems: accuracy and generation -----######

# acc_callback = keras.callbacks.LambdaCallback(on_epoch_end=accuracy)

# display and fit model
vae.summary()
vae.fit(train, train, batch_size=batch_size, epochs=1, shuffle=True, validation_data=(valid, None))

loss = vae.evaluate(test, test, batch_size=batch_size)
print('loss', loss)

# reconstructed = vae.predict(x_test, batch_size=batch_size)
'''
state = encoder.predict(x_test[0].reshape(1, 28, 28))
out = np.zeros((1, 1, 28), dtype=np.float32)  # initial "start" vector
predicted = []
for _ in range(28):
    out, state = decoder.predict([out, state])
    predicted.append(out.reshape(-1,))
predicted = np.stack(predicted, axis=0)

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(x_test[0], cmap='gray')
ax1.set_axis_off()

ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(predicted, cmap='gray')
ax2.set_axis_off()

plt.show()
'''
# show results
# n = 5

# encoded_means, _ = encoder.predict(test_imgs)
# decoded_imgs_means = decoder.predict(encoded_means).reshape(-1, 28, 28)
# decoded_imgs_noise = vae.predict(test_imgs).reshape(-1, 28, 28)
# test_imgs = x_test[0: n]
# recon_imgs = reconstructed[0: n]
# fig = plt.figure()
# for i in range(1, n + 1):
#     # display original
#     ax = fig.add_subplot(2, n, i)
#     ax.imshow(test_imgs[i - 1], cmap='gray')
#     ax.set_axis_off()
#
#     # display mean reconstruction
#     ax = fig.add_subplot(2, n, i + n)
#     ax.imshow(recon_imgs[i - 1], cmap='gray')
#     ax.set_axis_off()
#
#     # display noisy reconstruction
#     # ax = fig.add_subplot(3, n, i + 2 * n)
#     # plt.imshow(decoded_imgs_noise[i - 1], cmap='gray')
#     # ax.set_axis_off()
#
# plt.show()
