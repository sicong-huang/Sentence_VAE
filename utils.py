import numpy as np
import pickle

# load indexed sentence data
def load_data(file, name, type):
    with np.load('data/' + file) as f:
        data = f[name].astype(type)
    return data


# a function to trim the first axis of data to fit the given batch size
def fit_batch(data, batch):
    data_points = data.shape[0]
    remainder = data_points % batch
    return data[:data_points - remainder]


def save_object(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
