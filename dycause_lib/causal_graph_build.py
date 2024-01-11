import numpy as np


def normalize_by_column(transition_matrix):
    for col_index in range(transition_matrix.shape[1]):
        if np.sum(transition_matrix[:, col_index]) == 0:
            continue
        transition_matrix[:, col_index] = transition_matrix[:, col_index] / np.sum(
            transition_matrix[:, col_index]
        )
    return transition_matrix


def normalize_by_row(transition_matrix):
    for row_index in range(transition_matrix.shape[0]):
        if np.sum(transition_matrix[row_index, :]) == 0:
            continue
        transition_matrix[row_index, :] = transition_matrix[row_index, :] / np.sum(
            transition_matrix[row_index, :]
        )
    return transition_matrix


def get_overlay_count(n_sample, ordered_intervals):
    overlay_counts = np.zeros([n_sample, 1], dtype=np.int)
    # Reconstruct the intervals to tuple format (None, (s1, e1)).
    if len(ordered_intervals) > 0 and isinstance(ordered_intervals[0][0], int):
        ordered_intervals = [(None, _) for _ in ordered_intervals]
    for interval in ordered_intervals:
        overlay_counts[interval[1][0] : interval[1][1], 0] += 1
    return overlay_counts


def get_ordered_intervals(matrics, significant_thres, list_segment_split):
    array_results_YX, array_results_XY = matrics
    array_results_YX = np.abs(array_results_YX)
    array_results_XY = np.abs(array_results_XY)
    nrows, ncols = array_results_YX.shape
    intervals = []
    pvalues = []
    for i in range(nrows):
        for j in range(i + 1, ncols):
            if (abs(array_results_YX[i, j]) < significant_thres) and ( 
                array_results_XY[i, j] >= significant_thres
                or array_results_XY[i, j] == -1
            ):
                intervals.append((list_segment_split[i], list_segment_split[j]))
                pvalues.append((array_results_YX[i, j], array_results_XY[i, j]))
    ordered_intervals = list(zip(pvalues, intervals))
    ordered_intervals.sort(key=lambda x: (x[0][0], -x[0][1]))
    return ordered_intervals


def get_segment_split(n_sample, step):
    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)
    return list_segment_split


# Following are not used functions


def get_intervals_over_overlaythres(counts, overlay_thres):
    mask = counts > overlay_thres
    if not np.any(mask):
        return []
    indices = np.where(mask)[0]
    starts = [indices[0]]
    ends = []
    for i in np.where(np.diff(indices, axis=0) > 1)[0]:
        ends.append(indices[i] + 1)
        starts.append(indices[i + 1])
    ends.append(indices[-1] + 1)
    return list(zip(starts, ends))


def get_max_overlay_intervals(counts):
    if np.max(counts) == 0:
        return []
    sample_indices_max = np.where(np.max(counts) == counts)[0]
    starts = [sample_indices_max[0]]
    ends = []
    for i in np.where(np.diff(sample_indices_max, axis=0) > 1)[0]:
        ends.append(sample_indices_max[i] + 1)
        starts.append(sample_indices_max[i + 1])
    ends.append(sample_indices_max[-1] + 1)
    return list(zip(starts, ends))


def get_cover_proportion(n_sample, ordered_intervals):
    x = np.zeros([n_sample])
    for interval in ordered_intervals:
        x[interval[1][0] : interval[1][1]] = 1
    return np.sum(x) / (0.0 + n_sample)


def adaptive_thresholding(DCC, threshold, N, normal_axis='col'):
    """Build the dependency graph using adaptive thresholding algorithm.

    Args:
        DCC : the dynamic causality curves of edges.
        threshold : the adaptive threshold.
        N : num of time series.
        normal_axis: the axis along which to apply normalization. Can be col or row.
    """
    edge = []
    edge_weight = {}
    for x_i in range(N):
        bar_data = []
        for y_i in range(N):
            if x_i != y_i:
                key = "{0}->{1}".format(x_i, y_i)
                arr = DCC[key]
                if isinstance(arr, np.ndarray):
                    bar_data.append(float(arr.sum()))
                else:
                    bar_data.append(float(np.sum(arr)))
            else:
                bar_data.append(0.0)
        bar_data_thres = np.max(bar_data) * threshold
        for y_i in range(N):
            if bar_data[y_i] >= bar_data_thres and bar_data_thres > 0:
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = bar_data[y_i]
    # Make the transition matrix with edge weight estimation
    transition_matrix = np.zeros([N, N])
    for key, val in edge_weight.items():
        x, y = key
        transition_matrix[x, y] = val
    if normal_axis == 'col':
        transition_matrix = normalize_by_column(transition_matrix)
    elif normal_axis == 'row':
        transition_matrix = normalize_by_row(transition_matrix)
    else:
        raise NotImplementedError("No such normal_axis. Can only be col or row.")
    return transition_matrix