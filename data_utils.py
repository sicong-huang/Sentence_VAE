import numpy as np

# load indexed sentence data
def load_data(which_data):
    with np.load('data/' + which_data + '.npz') as f:
        data = f[which_data].astype(np.int32)
    return data

# a function to trim the first axis of data to fit the given batch size
def fit_batch(data, batch):
    data_points = data.shape[0]
    remainder = data_points % batch
    return data[:data_points - remainder]
