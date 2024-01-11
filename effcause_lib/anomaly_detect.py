import os

import matplotlib.pyplot as plt
import numpy as np


def anomaly_detect(
    data,
    weight=1,
    mean_interval=60,
    anomaly_proportion=0.3,
    verbose=True,
    path_output=None,
):
    """Detect the time when anomaly first appears.

    Params:
        data: multi column data where each column represents a variable.
        weight: weight assigned to every variable when calculating anomaly
            scores.
        mean_interval: the size of sliding window to calculate standard deviation.
            Anomaly score within the first (mean_interval - 1) timestamps are 0.
        anomaly_proportion: proportion of anomaly variables considered to be
            anomaly, must relates to weight.
        verbose: the debugging print level: 0 (Nothing), 1 (Method info), 2 (Phase info), 3(Algorithm info)
    Returns:
        start_index: index in data when anomaly starts.
    """
    data_ma = []
    data_std = []

    def moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n

    for col in range(data.shape[1]):
        data_ma.append(
            np.concatenate(
                [
                    np.zeros([mean_interval - 1]),
                    moving_average(data[:, col], n=mean_interval),
                ],
                axis=0,
            )
        )
    data_ma = np.array(data_ma).T
    for col in range(data.shape[1]):
        one_std = [0 for i in range(mean_interval - 1)]
        for row in range(data.shape[0] - mean_interval):
            one_std.append(np.std(data[row : row + mean_interval, col]))
        data_std.append(one_std)
    data_std = np.array(data_std).T

    # Sum over nodes to get dither level of time
    # Here apply a weight of 1 to every node
    if weight == 1:
        dither_proportion = np.sum(
            (data_std > 1.0) * np.ones([data_std.shape[1]]), axis=1
        )

    start_time_list = [
        i
        for i in np.argsort(dither_proportion)[::-1]
        if dither_proportion[i] >= data.shape[1] * anomaly_proportion
    ]
    start_time = start_time_list[0]

    return start_time, dither_proportion[start_time]
