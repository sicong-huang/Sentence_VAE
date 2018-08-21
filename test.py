import numpy as np
from gru_model_class import ModelStruct
import data_utils
'''
train = data_utils.load_data('train')
valid = data_utils.load_data('valid')
test = data_utils.load_data('test')

print('train shape:', train.shape)
print('valid shape:', valid.shape)
print('test shape:', test.shape)

print('train max', np.max(train))
print('valid max', np.max(valid))
print('test max', np.max(test))

print('train type', train.dtype)
print('valid type', valid.dtype)
print('test type', test.dtype)
'''

batch_size = 64
seq_len = 32
embedding_dim = 100
latent_size = 64
vocab_size = 1000
batch_shape = (batch_size, seq_len)

# data = np.random.rand(batch_size, seq_len, input_size)

ms = ModelStruct(batch_shape, embedding_dim, latent_size, vocab_size)
vae = ms.assemble_vae_train()
vae.summary()

# output = encoder.predict(data, batch_size=batch_size)
# print(output)
