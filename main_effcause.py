import argparse
import datetime
import threading
import os
import sys
import time
import pickle
import logging
from collections import defaultdict
import random
import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager, Pool, RLock, freeze_support

import numpy as np
import networkx as nx
from tqdm.auto import tqdm

from effcause_lib.anomaly_detect import anomaly_detect
from effcause_lib.granger import bidirect_granger
from effcause_lib.causal_graph_build import get_segment_split
from effcause_lib.causal_graph_build import get_bidirect_intervals
from effcause_lib.causal_graph_build import get_count
from effcause_lib.causal_graph_build import normalize_by_row, normalize_by_column
from effcause_lib.randwalk import randwalk
from effcause_lib.ranknode import ranknode, analyze_root
from utility_funcs.proc_data import load_data, safe_dump_obj
from utility_funcs.evaluation_function import prCal, my_acc, pr_stat, print_prk_acc
from utility_funcs.format_ouput import format_to_excel
from utility_funcs.excel_utils import saveToExcel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

def sub_process_discover(
        hyper_params,
        i,
        shared_result_dict):
    try: 
        reuse_invdirection = hyper_params["reuse_invdirection"]
        j = 0
        if hyper_params["share_data_by_pickle"]:
            data_path = hyper_params["local_data"]
            with open(data_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = hyper_params['local_data']
        length = data.shape[1]
        data_head = [str(i) for i in range(length)]
        sub_dict = defaultdict(dict)
        
        if hyper_params['verbose'] and i == 0:
            if reuse_invdirection:
                pbar = tqdm(total=length * (length -1 )/ 2)
            else:
                pbar = tqdm(total=length * (length -1 ))
        for x_i in range(length):
            for y_i in range(length):
                # Skip self to self causality tests because Granger causality assumes an autoregression
                # in default.
                if x_i == y_i:
                    continue
                # If reuse_invdirection is True, we only test for the pair (x_i -> y_i, x_i < y_i) and reuse the internal
                # data to discover relations in the inverse pair (y_i -> x_i).
                if x_i > y_i and reuse_invdirection:
                    continue
                if (j % hyper_params['max_workers'] == i):
                    sub_dict['{}->{}'.format(x_i, y_i)] = bidirect_granger(
                        data,
                        data_head,
                        hyper_params['dir_output'],
                        str(x_i),
                        str(y_i),
                        hyper_params['significant_thres'],
                        hyper_params['method'],
                        hyper_params['trip'],
                        hyper_params['lag'],
                        hyper_params['step'],
                        hyper_params['max_segment_len'],
                        hyper_params['min_segment_len'],
                        rolling_method=hyper_params['rolling_method'],
                        min_nobs=hyper_params["min_nobs"],
                        adftest=hyper_params['adftest'],
                        verbose=hyper_params['verbose'],
                        return_result=True)
                j = j + 1
                if hyper_params['verbose'] and i == 0:
                    pbar.update(1)
        if hyper_params['verbose'] and i == 0:
            pbar.close()
        
        for key, value in sub_dict.items():
            shared_result_dict[key]=value

        return
    except Exception as e:
        print("Exception occurred!", e)
        logging.error("Exception occurred!")


def effcause_causal_discover(
    # Data params
    data,
    # Granger interval based graph construction params
    step=60,
    significant_thres=0.1,
    lag=10,  # must satisfy: step > 3 * lag + 1
    adaptive_threshold=0.7,
    use_multiprocess=True,
    max_workers=3,
    reuse_invdirection=True,
    rolling_method="pyc",
    min_nobs=25,
    max_segment_len=None,
    adftest=True,
    share_data_by_pickle=False,
    # Debug_params
    verbose=False,
    runtime_debug=False,
    *args,
    **kws,
):
    """EffCause Causal Discover Algorithm

    Args:
        data (numpy array): The input time series of shape [N, T]
        step (int, basic window size): The basic window size. Defaults to 60.
        significant_thres (float, optional): Granger significance value. Defaults to 0.1.
        lag (int, optional): How many past values from time t. Defaults to 10.
        use_multiprocess (bool, optional): Whether use multiprocess library. 
                                           If False, use multithread library. Defaults to True.
        max_workers (int, optional): Maximum process or thread number. Defaults to 3.
        reuse_invdirection (bool, optional): Whether reuse the test results from inverse direction. Default to True.
                                             If True, for any pair (x, y), only one direction will be tested (x <= y).
                                             The test results from the inverse direction is obtained from internal pruning data.
                                             This could lose some information but accelerate the algorithm.
                                             If False, use the the normal pairwise test scheme. The results from the inverse direction
                                             is not used and wasted. TODO: could be improved.
        rolling_method: RollingOLS实现的方式，pyc为正确的，但是很慢；zyf为采取张伊凡的方式，尝试修复错误。
        min_nobs: the min_nobs parameter in rolling regression, only used in zyf mode. This parameter has
            constrains: `min_nobs must be larger than the number of regressors in the model and less than window`
        max_segment_len: the maximum tested sliding window. Default is None, which means the maximum is all the data.
        adftest: whether to conduct the Adf stationarity test. If the data is not stationary, we will
                 reject the any granger test results.
        share_data_by_pickle: whether use disk file to share data across processes.
            Default is False, which will transfer data using multiprocess default mechanism.
        verbose (bool, optional): Whether print runtime info. Defaults to False.
        runtime_debug (bool, optional): Whether enable run time test mode. 
                                        This is used to measure run time. Defaults to False.
    """
    data_source = "effcause_temp"
    np.random.seed(42)
    random.seed(42)
    time_stat_dict = {}
    tic = time.time()
    if verbose:
        # verbose level >= 1: print method name
        print("{:-^80}".format("EffCause"))
    dir_output = "temp_results/effcause/" + data_source
    os.makedirs(dir_output, exist_ok=True)
    # Use the timezone in my location.
    local_tz = datetime.timezone(datetime.timedelta(hours=8))
    time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    if verbose:
        print("{:-^80}".format("EffCause impact graph construction phase"))
        print("{:<10}".format("") + f"Starting at {time_str:^80}")
    local_length = data.shape[0]
    local_data = data
    
    # method = "v2" # Always use online update algoritm
    if (step * lag < local_length):
        method = "v2" # online update algorithm
    else:
        method = "v1" # original prune algorithm

    trip = -1
    if max_segment_len is None:
        max_segment_len = local_length
    min_segment_len = step
    list_segment_split = get_segment_split(local_length, step)


    if verbose:
        print(
            "{space:^10}{name}:\n"
            "{space:^15}data shape   :{shape}\n"
            "{space:^15}lag          :{lag}\n"
            "{space:^15}significant  :{sig}\n"
            "{space:^15}step         :{step}\n"
            "{space:^15}min len      :{min}\n"
            "{space:^15}max len      :{max}\n"
            "{space:^15}method       :{method}\n"
            "{space:^15}rolling meth :{roll_method}\n"
            "{space:^15}min_nobs     :{min_nobs}\n"
            "{space:^15}adf test     :{adftest}\n"
            "{space:^15}share_data_by_pickle:{share_data_by_pickle}\n"
            "{space:^15}segment split:".format(
                space="",
                name="Calculating granger intervals",
                shape=local_data.shape,
                lag=lag,
                sig=significant_thres,
                step=step,
                min=min_segment_len,
                max=max_segment_len,
                method=method,
                roll_method=rolling_method,
                min_nobs=min_nobs,
                adftest=adftest,
                share_data_by_pickle=share_data_by_pickle
            ),
            list_segment_split,
        )
    local_results = defaultdict(dict)
    result_dict = defaultdict(dict)
    
    if (max_segment_len > 100) and use_multiprocess and max_workers > 1:
        if verbose:
            print("Using ProcessPoolExecutor.")
        hyper_params = {
            'local_data': local_data,
            'dir_output': dir_output,
            'significant_thres': significant_thres,
            'method': method,
            'trip': trip,
            'lag': lag,
            'step': step,
            'max_segment_len': max_segment_len,
            'min_segment_len': min_segment_len,
            'max_workers': max_workers,
            "reuse_invdirection": reuse_invdirection,
            "rolling_method": rolling_method,
            "min_nobs": min_nobs,
            "adftest": adftest,
            "share_data_by_pickle": share_data_by_pickle,
            'verbose': verbose
        }
        if share_data_by_pickle:
            hyper_params["local_data"] = os.path.join(dir_output, 
                f"temp_data_{time_str}.pkl")
            safe_dump_obj(local_data, hyper_params["local_data"])
        manager = Manager()
        shared_result_dict = manager.dict()
        executor = ProcessPoolExecutor(max_workers=max_workers, 
                initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),))
        futures = []
        for i in range(max_workers):
            futures.append(executor.submit(
                sub_process_discover,
                hyper_params,
                i,
                shared_result_dict))
        executor.shutdown(wait=True)
        result_dict = shared_result_dict
    else:
        # In single process mode, no need to share data by pickle file.
        share_data_by_pickle = False
        hyper_params = {
            'local_data': local_data,
            'dir_output': dir_output,
            'significant_thres': significant_thres,
            'method': method,
            'trip': trip,
            'lag': lag,
            'step': step,
            'max_segment_len': max_segment_len,
            'min_segment_len': min_segment_len,
            'max_workers': 1,
            "reuse_invdirection": reuse_invdirection,
            "rolling_method": rolling_method,
            "min_nobs": min_nobs,
            "adftest": adftest,
            "share_data_by_pickle": share_data_by_pickle,
            'verbose': verbose
        }
        # Single process function call, just use the normal dict.
        sub_process_discover(hyper_params, 0, result_dict)
    if runtime_debug:
        time_stat_dict["time_total"] = 0
        time_stat_dict["time_OLS"] = []
        time_stat_dict["time_window"] = []
        time_stat_dict["time_granger"] = 0
        time_stat_dict["time_adf"] = []
    i = 0
    for x_i in range(data.shape[1]):
        for y_i in range(data.shape[1]):
            if x_i == y_i:
                continue
            if x_i > y_i and reuse_invdirection:
                continue
            (
                time_dict,
                array_results_YX,
                array_results_XY,
            ) = result_dict['{}->{}'.format(x_i, y_i)]
            total_time = time_dict['total_time']
            time_OLS = time_dict['time_OLS']
            time_granger = time_dict['time_granger']
            time_adf = time_dict['time_adf']
            time_window = time_dict['time_window']
            if runtime_debug:
                time_stat_dict["time_total"] += total_time
                time_stat_dict["time_OLS"].append(time_OLS)
                time_stat_dict["time_window"].append(time_window)
                time_stat_dict["time_granger"] += time_granger
                time_stat_dict["time_adf"].append(time_adf)
            matrics = [array_results_YX, array_results_XY]
            intervals_XY, intervals_YX = get_bidirect_intervals(
                matrics, significant_thres, list_segment_split
            )
            local_results["%s->%s" %
                            (x_i, y_i)]["intervals"] = intervals_XY
            if reuse_invdirection:
                local_results["%s->%s" %
                                (y_i, x_i)]["intervals"] = intervals_YX

            intervals_XY, intervals_YX = get_bidirect_intervals(
                matrics, significant_thres, list_segment_split, with_inv_judge=False
            )
            local_results["%s->%s" %
                            (x_i, y_i)]["intervals_noinv"] = intervals_XY
            if reuse_invdirection:
                local_results["%s->%s" %
                                (y_i, x_i)]["intervals_noinv"] = intervals_YX
            # result_YX is the Granger test results of: Y <-- X (target <-- feature).
            # result_XY is the Granger test results of: X <-- Y (feature <-- target).
            local_results["%s->%s" % (x_i, y_i)]["result_YX"] = array_results_YX
            local_results["%s->%s" % (x_i, y_i)]["result_XY"] = array_results_XY
            i = i + 1

    # region Construction impact graph using generated intervals
    # Generate dynamic causal curve between two services by overlaying intervals
    histogram_sum = defaultdict(int)
    dcc = {}
    edge = []
    edge_weight = dict()
    for x_i in range(data.shape[1]):
        for y_i in range(data.shape[1]):
            if x_i == y_i:
                continue
            key = "{0}->{1}".format(x_i, y_i)
            intervals = local_results[key]["intervals"]
#             intervals = local_results.get(key, defaultdict(dict)).get("intervals", defaultdict(dict))
            overlay_counts = get_count(local_length, intervals)
            dcc[key] = overlay_counts
            histogram_sum[key] = sum(overlay_counts)
    # Make edges from 1 node using comparison and auto-threshold
    for x_i in range(data.shape[1]):
        bar_data = []
        for y_i in range(data.shape[1]):
            key = "{0}->{1}".format(x_i, y_i)
            bar_data.append(histogram_sum[key])
            
        bar_data_thres = np.max(bar_data) * adaptive_threshold
        for y_i in range(data.shape[1]):
            if bar_data[y_i] >= bar_data_thres:
                edge.append((x_i, y_i))
                edge_weight[(x_i, y_i)] = bar_data[y_i]
    # Make the transition matrix with edge weight estimation
    transition_matrix = np.zeros([data.shape[1], data.shape[1]])
    for key, val in edge_weight.items():
        x, y = key
        transition_matrix[x, y] = val
    transition_matrix = normalize_by_column(transition_matrix)
    # endregion
    toc = time.time()
    time_stat_dict["Construct-Impact-Graph-Phase"] = toc - tic
    tic = toc
    if runtime_debug:
        time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
        if verbose:
            print("{:<10}".format("") + "Saving runtime data to " + f"time_stat_dict_{time_str}.pkl")
        safe_dump_obj(time_stat_dict, os.path.join(dir_output,"runtime-data",f"time_stat_dict_{time_str}.pkl"))

    # endregion

    return local_results, dcc, transition_matrix, time_stat_dict


if __name__ == '__main__':
    from utility_funcs.proc_data import load_tcdf_data
    datasets, _ = load_tcdf_data("finance")

    _, _, _, time_stat_dict_3 = effcause_causal_discover(
        # Data params
        datasets[0].to_numpy()[:, 0:3],
        # Granger interval based graph construction params
        step=4000,
        significant_thres=0.1,
        lag=3,  # must satisfy: step > 3 * lag + 1
        adaptive_threshold=0.7,
        use_multiprocess=True,
        max_workers=3,
        # Debug_params
        verbose=True,
        runtime_debug=True,
    )
    print(time_stat_dict_3["Construct-Impact-Graph-Phase"])
