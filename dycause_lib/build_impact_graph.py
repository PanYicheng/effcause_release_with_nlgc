"""Build correlation graph using granger causality.

Contains normal granger causality method and 
granger causality interval method.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
import pickle
import threading
from tqdm import tqdm

import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests as granger_std

# Granger causal interval 作者提供的代码
from dycause_lib.Granger_all_code import loop_granger
from dycause_lib.causal_graph_build import get_segment_split
from dycause_lib.causal_graph_build import get_ordered_intervals
from dycause_lib.causal_graph_build import get_overlay_count
from dycause_lib.causal_graph_build import normalize_by_row
from dycause_lib.draw_graph import *



def build_graph_normalgranger(
    data, data_head, significant_thres, dir_output=".", verbose=False
):
    if verbose:
        print("{:-^80}".format("Running Normal Granger Causality"))
    lag = 5
    granger_results_file_path = os.path.join(
        dir_output, "granger_results_lag{}.pkl".format(lag)
    )
    if os.path.exists(granger_results_file_path):
        with open(granger_results_file_path, "rb") as f:
            granger_results = pickle.load(f)
    else:
        granger_results = defaultdict(dict)
        for x_i in range(len(data_head)):
            for y_i in range(len(data_head)):
                if x_i == y_i:
                    continue
                if verbose:
                    print("{:^10}Calculating {}->{}".format("", x_i, y_i), end="\r")
                array_YX = np.concatenate(
                    [data[y_i, :].T.reshape(-1, 1), data[x_i, :].T.reshape(-1, 1)],
                    axis=1,
                )
                array_XY = np.concatenate(
                    [data[x_i, :].T.reshape(-1, 1), data[y_i, :].T.reshape(-1, 1)],
                    axis=1,
                )
                result = granger_std(array_YX, lag, addconst=True, verbose=False)
                p_value_YX = result[lag][0]["ssr_ftest"][1]
                if p_value_YX < significant_thres:
                    result = granger_std(array_XY, lag, addconst=True, verbose=False)
                    p_value_XY = result[5][0]["ssr_ftest"][1]
                else:
                    p_value_XY = -1
                granger_results["{}->{}".format(x_i, y_i)]["p_value"] = (
                    p_value_YX,
                    p_value_XY,
                )
        if verbose:
            print("")
        os.makedirs(os.path.dirname(granger_results_file_path), exist_ok=True)
        with open(granger_results_file_path, "wb") as f:
            pickle.dump(granger_results, f)
    transition_matrix = np.zeros([data.shape[0], data.shape[0]])
    for x_i in range(data.shape[0]):
        for y_i in range(data.shape[0]):
            if x_i == y_i:
                continue
            p_value = granger_results["{}->{}".format(x_i, y_i)]["p_value"]
            if p_value[0] < significant_thres and p_value[1] >= significant_thres:
                transition_matrix[x_i, y_i] = 1
    return transition_matrix


def build_graph_grangerinterval(
    data, data_head, significant_thres, dir_output=".", verbose=False
):
    # region Granger causal interval algorithm
    if verbose:
        print("{:-^80}".format("Running Granger Intervals"))
    local_length = data.shape[1]
    local_data = data.T
    step = 50
    lag = 5
    max_segment_len = min(local_length, 300)
    min_segment_len = step
    list_segment_split = get_segment_split(local_length, step)

    local_results_file_path = os.path.join(
        dir_output,
        "local-results",
        "local_results"
        "_len{len}_lag{lag}_sig{sig}_step{step}.pkl".format(
            len=local_length, lag=lag, sig=significant_thres, step=step
        ),
    )
    if os.path.exists(local_results_file_path):
        print(
            "{:^10}".format("") + "Loading previous granger interval results:",
            os.path.basename(local_results_file_path),
        )
        with open(local_results_file_path, "rb") as f:
            local_results = pickle.load(f)
    else:
        print(
            "{space:^10}{name}:\n"
            "{space:^15}len          :{len}\n"
            "{space:^15}lag          :{lag}\n"
            "{space:^15}significant  :{sig}\n"
            "{space:^15}step         :{step}\n"
            "{space:^15}min len      :{min}\n"
            "{space:^15}max len      :{max}\n"
            "{space:^15}segment split:".format(
                space="",
                name="Calculating granger intervals",
                len=local_length,
                lag=lag,
                sig=significant_thres,
                step=step,
                min=min_segment_len,
                max=max_segment_len,
            ),
            list_segment_split,
        )
        local_results = defaultdict(dict)

        def granger_process(x, y):
            try:
                ret = loop_granger(
                    local_data,
                    data_head,
                    dir_output,
                    data_head[x],
                    data_head[y],
                    significant_thres,
                    "fast_version_3",
                    -1,
                    lag,
                    step,
                    "simu",
                    max_segment_len,
                    min_segment_len,
                    verbose=False,
                    return_result=True,
                )
            except Exception as e:
                print("Exception occurred at {} -> {}!".format(x, y), e)
                ret = (None, None, None, None, None)
            return ret

        lock = threading.Lock()
        total_thread_num = [len(data_head) * (len(data_head) - 1)]
        thread_results = [0 for i in range(total_thread_num[0])]

        pbar = tqdm(total=total_thread_num[0], ascii=True)

        def thread_func(i, x, y):
            # print('Thread {} started'.format(i))
            thread_results[i] = granger_process(x, y)
            lock.acquire()
            pbar.update(1)
            lock.release()
            return

        executor = ThreadPoolExecutor(max_workers=6)
        i = 0
        for x_i in range(len(data_head)):
            for y_i in range(len(data_head)):
                if x_i == y_i:
                    continue
                executor.submit(thread_func, i, x_i, y_i)
                i = i + 1
        executor.shutdown(wait=True)
        pbar.close()
        i = 0
        for x_i in range(len(data_head)):
            for y_i in range(len(data_head)):
                if x_i == y_i:
                    continue
                (
                    total_time,
                    time_granger,
                    time_adf,
                    array_results_YX,
                    array_results_XY,
                ) = thread_results[i]
                matrics = [array_results_YX, array_results_XY]
                ordered_intervals = get_ordered_intervals(
                    matrics, significant_thres, list_segment_split
                )
                local_results["%s->%s" % (x_i, y_i)]["intervals"] = ordered_intervals
                local_results["%s->%s" % (x_i, y_i)]["result_YX"] = array_results_YX
                local_results["%s->%s" % (x_i, y_i)]["result_XY"] = array_results_XY
                i = i + 1
        os.makedirs(os.path.dirname(local_results_file_path), exist_ok=True)
        with open(local_results_file_path, "wb") as f:
            pickle.dump(local_results, f)
    # endregion

    # region Build impact graph using generated intervals
    plot_temp_figure = 1
    # Overlay intervals to get histogram estimates of edges
    histogram_sum = defaultdict(int)
    edge = []
    edge_weight = dict()
    for x_i in range(len(data_head)):
        for y_i in range(len(data_head)):
            if y_i == x_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = local_results[key]["intervals"]
            overlay_counts = get_overlay_count(local_length, intervals)
            # whether plot temporaray figure pair wise
            if plot_temp_figure >= 2:
                os.makedirs(os.path.join(dir_output, "pair-imgs"), exist_ok=True)
                print(
                    "{:^10}Ploting {:2d}->{:2d}".format("", x_i + 1, y_i + 1), end="\r"
                )
                draw_overlay_histogram(
                    overlay_counts,
                    "{}->{}".format(x_i + 1, y_i + 1),
                    os.path.join(
                        dir_output, "pair-imgs", "{0}->{1}.png".format(x_i + 1, y_i + 1)
                    ),
                )
            histogram_sum[key] = sum(overlay_counts)
    # skip the \r print line
    if plot_temp_figure:
        print("")
    # Make edges from 1 node using comparison and auto-threshold
    auto_threshold_ratio = 0.8
    for x_i in range(len(data_head[:])):
        bar_data = []
        for y_i in range(len(data_head)):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histogram_sum[key])
        # whether plot temporary figure from one node
        if plot_temp_figure:
            if not os.path.exists(os.path.join(dir_output, "aggre-imgs")):
                os.makedirs(os.path.join(dir_output, "aggre-imgs"))
            print("{:^10}Ploting aggre imgs {:2d}".format("", x_i + 1), end="\r")
            draw_bar_histogram(
                bar_data,
                "From {0}".format(x_i + 1),
                os.path.join(dir_output, "aggre-imgs", "{0}.png".format(x_i + 1)),
            )
        bar_data_thres = np.max(bar_data) * auto_threshold_ratio
        for y_i in range(len(data_head)):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = bar_data[y_i]
    # skip the \r print line
    if plot_temp_figure:
        print("")
    # Make the transition matrix with edge weight estimation
    transition_matrix = np.zeros([data.shape[0], data.shape[0]])
    for key, val in edge_weight.items():
        x, y = key
        transition_matrix[x, y] = val
    # transmission_matrix = normalize_by_column(transmission_matrix)
    transition_matrix = normalize_by_row(transition_matrix)
    # endregion

    return transition_matrix


