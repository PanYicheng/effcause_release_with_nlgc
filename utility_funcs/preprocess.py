import numpy as np


def fill_zero(data):
    """Fill 0s in the data with recent value.

    data is N x M matrix, N is node num, M is data point num.
    Each row is a variable.
    """
    filled_data = data.copy()
    for i in range(filled_data.shape[0]):
        for j in range(filled_data.shape[1]):
            if filled_data[i, j] == 0 and j >= 1:
                filled_data[i, j] = filled_data[i, j-1]
    for i in range(filled_data.shape[0]-1, -1, -1):
        for j in range(filled_data.shape[1]-1, -1, -1):
            if filled_data[i, j] == 0 and j <= filled_data.shape[1]-2:
                filled_data[i, j] = filled_data[i, j+1]
    data = filled_data
    data_mean = np.mean(data, axis=1, keepdims=True)
    data_std = np.std(data, axis=1, keepdims=True)
    data = (data - data_mean)/data_std
    return data
