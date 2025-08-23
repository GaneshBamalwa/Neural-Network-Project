import numpy as np
import pickle

def get_batches(X, Y, batch_size):
    for start_idx in range(0, len(X), batch_size):
        end_idx = start_idx + batch_size
        yield X[start_idx:end_idx], Y[start_idx:end_idx]

def save_model(filename, layers):
    params = [{'weights': layer.weights, 'biases': layer.biases} for layer in layers]
    with open(filename, 'wb') as f:
        pickle.dump(params, f)

def load_model(filename, layers):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    for layer, param in zip(layers, params):
        layer.weights = param['weights']
        layer.biases = param['biases']
