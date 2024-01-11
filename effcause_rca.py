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
from main_effcause import effcause_causal_discover
from utility_funcs.proc_data import load_data, safe_dump_obj
from utility_funcs.evaluation_function import prCal, my_acc, pr_stat, print_prk_acc
from utility_funcs.format_ouput import format_to_excel
from utility_funcs.excel_utils import saveToExcel

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def effcause_rca(
    # Data params
    data_source="ibm_micro_service",
    aggre_delta=1,
    start_time=None,
    before_length=300,
    after_length=300,
    # Granger interval based graph construction params
    step=50,
    significant_thres=0.05,
    lag=5,  # must satisfy: step > 3 * lag + 1
    auto_threshold_ratio=0.8,
    runtime_debug=False,
    # Root cause analysis params
    testrun_round=1,
    frontend=14,
    max_path_length=None,
    mean_method="harmonic",
    true_root_cause=[6, 28, 30, 31],
    topk_path=50,
    num_sel_node=3,
    # Debug params
    use_multiprocess=True,
    verbose=True,
    max_workers=3,
    **kws,
):
    if runtime_debug:
        time_stat_dict = {}
        tic = time.time()
    if verbose:
        print("{:#^80}".format("EffCause"))

    dir_output = "temp_results/effcause_lib/" + data_source
    os.makedirs(dir_output, exist_ok=True)
    if verbose:
        print("{:-^80}".format("Data load phase"))
    # region Load and preprocess data
    if data_source == "external":
        data = kws["data"]
        data_head = kws["data_head"]
    else:
        data, data_head = load_data(
            os.path.join("data", data_source, "rawdata.xlsx"),
            normalize=True,
            zero_fill_method='prevlatter',
            aggre_delta=aggre_delta,
            verbose=verbose,
        )
    if start_time is None:
        start_time = 0
    data = data[start_time-before_length:start_time+after_length, :]
    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['Load phase'] = toc-tic
        tic = toc

    local_results, dcc, transition_matrix, time_stat_dict_3 = effcause_causal_discover(
        # Data params
        data,
        # Granger interval based graph construction params
        step=step,
        significant_thres=significant_thres,
        lag=lag,  # must satisfy: step > 3 * lag + 1
        adaptive_threshold=auto_threshold_ratio,
        use_multiprocess=use_multiprocess,
        max_workers=max_workers,
        rolling_method="zyf",
        # Debug_params
        verbose=verbose,
        runtime_debug=runtime_debug,
    )

    # region backtrace root cause analysis
    if verbose:
        print("{:-^80}".format("Back trace root cause analysis phase"))
    topk_list = range(1, 6)
    prkS = [0] * len(topk_list)
    if not isinstance(frontend, list):
        frontend = [frontend]
    for entry_point in frontend:
        if verbose:
            print("{:*^40}".format(" Entry: {:2d} ".format(entry_point)))
        prkS_list = []
        acc_list = []
        for i in range(testrun_round):
            ranked_nodes, new_matrix = analyze_root(
                transition_matrix,
                entry_point,
                data,
                mean_method=mean_method,
                max_path_length=max_path_length,
                topk_path=topk_path,
                prob_thres=0.2,
                num_sel_node=num_sel_node,
                use_new_matrix=False,
                verbose=verbose,
            )
            if verbose:
                print("{:^0}|{:>8}|{:>12}|".format("", "Node", "Score"))
                for j in range(len(ranked_nodes)):
                    print(
                        "{:^0}|{:>8d}|{:>12.7f}|".format(
                            "", ranked_nodes[j][0], ranked_nodes[j][1]
                        )
                    )
            prkS = pr_stat(ranked_nodes, true_root_cause)
            acc = my_acc(ranked_nodes, true_root_cause, len(data_head))
            prkS_list.append(prkS)
            acc_list.append(acc)
        prkS = np.mean(np.array(prkS_list), axis=0).tolist()
        acc = float(np.mean(np.array(acc_list)))
        if verbose:
            print_prk_acc(prkS, acc)

    # endregion
    if runtime_debug:
        toc = time.time()
        time_stat_dict['backtrace rca'] = toc - tic
        tic = toc
        print(time_stat_dict)

    return prkS, acc



if __name__ == '__main__':
    effcause_rca(
        # Data params
        data_source="ibm_micro_service",
        aggre_delta=1,
        start_time=4653,
        before_length=300,
        after_length=100,
        # Granger interval based graph construction params
        step=60,
        significant_thres=0.01,
        lag=5,  # must satisfy: step > 3 * lag + 1
        auto_threshold_ratio=0.7,
        runtime_debug=True,
        # Root cause analysis params
        testrun_round=1,
        frontend=14,
        max_path_length=None,
        mean_method="harmonic",
        true_root_cause=[6, 28, 30, 31],
        topk_path=150,
        num_sel_node=3,
        # Debug params
        use_multiprocess=True,
        verbose=True,
        max_workers=3,
    )

    # effcause_rca(
    #     # Data params
    #     data_source="pymicro",
    #     aggre_delta=1,
    #     start_time=1200,
    #     before_length=100,
    #     after_length=0,
    #     # Granger interval based graph construction params
    #     step=30,
    #     # step=100,
    #     significant_thres=0.1,
    #     lag=9,  # must satisfy: step > 3 * lag + 1
    #     auto_threshold_ratio=0.8,
    #     runtime_debug=True,
    #     # Root cause analysis params
    #     testrun_round=1,
    #     frontend=16,
    #     max_path_length=None,
    #     mean_method="harmonic",
    #     true_root_cause=[1],
    #     topk_path=60,
    #     num_sel_node=1,
    #     # Debug params
    #     use_multiprocess=True,
    #     verbose=True,
    #     max_workers=3,
    # )


