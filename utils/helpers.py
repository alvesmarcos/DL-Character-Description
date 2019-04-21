import pickle as pkl
import numpy as np

def save(obj, name):
    with open(name, 'wb') as f:
        pkl.dump(obj, f, pkl.HIGHEST_PROTOCOL)

def load(name):
    with open(name, 'rb') as f:
        return pkl.load(f)

def unwrapper_data(data):
    images = []
    labels = []
    for f in data:
        data = load(f)
        for X, y in zip(data['X'], data['y']):
            images.append(X)
            labels.append(y)
    return np.array(images, dtype=np.float32),  np.array(labels, dtype=np.uint8)
    