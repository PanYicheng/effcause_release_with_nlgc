#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the EffCause experiments on the TCDF finance datasets.
"""
from main_effcause import effcause_causal_discover

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import time
import datetime
import pandas as pd
import os
import numpy as np
import random
import networkx as nx
from sklearn import metrics
import pickle
from utility_funcs.proc_data import load_tcdf_data
if __name__ == "__main__":
    # TCDF datasets
    # datasets, gt_graphs = load_tcdf_data(dir="fmri")
    # datasets_, gt_graphs_ = [], []
    # for i in range(len(datasets)):
    #     if datasets[i].shape[0]>1000:
    #         datasets_.append(datasets[i])
    #         gt_graphs_.append(gt_graphs[i])
    # datasets, gt_graphs = datasets_, gt_graphs_
    # print("Num of datasets:", len(datasets))

    # TIMINO house temporature dataset
    df = pd.read_csv(os.path.join("data", "timino_house_data", "2011-03-07bis2012-01-04.csv"))
    print(df.columns)
    # Filter only temporature
    df = df.iloc[:, [0, 2, 4, 6, 8, 10]]
    # Rename to English
    df.columns = ["Living Room", "Outside", "Kitchen Boiler", "Shed", "WC", "Bathroom"]
    df = df[["Shed", "Outside", "Kitchen Boiler", "Living Room", "WC", "Bathroom"]]
    # Normalize
    data_mean = np.mean(df, axis=0)
    data_std = np.std(df, axis=0)
    df = (df - data_mean) / data_std

    # Search through parameters and save results.
    exp_rets = []
    pbar = tqdm(total=1)
    try:
        # for i, data in enumerate(datasets):
        #     for step in [100, 200, 300]:
        #         for sign in [0.005, 0.01, 0.05]:
        #             for lag in [3]:
        step=200
        sign=1e-2
        lag=50
        max_segment_len = 1000
        local_results, dcc, mat, time_stat_dict = effcause_causal_discover(
            # Data params
            df.to_numpy()[:, :],
            # Granger interval based graph construction params
            step=step,
            significant_thres=sign,
            lag=lag,  # must satisfy: step > 3 * lag + 1
            adaptive_threshold=0.7,
            use_multiprocess=True,
            max_workers=3,
            reuse_invdirection=True,
            rolling_method="zyf",
            min_nobs=110,
            max_segment_len=max_segment_len,
            adftest=True,
            # Debug_params
            verbose=True,
            runtime_debug=False,
        )
        exp_rets.append(({
            # "dataset_id": i, 
            # "T": T,
            "step": step,
            "sign": sign,
            "lag": lag,
            "max_segment_len": max_segment_len,
            "local_results": local_results,
            "dcc": dcc,
            "time_stat_dict": time_stat_dict
        }))
        pbar.set_description("{:.2f} s".format(time_stat_dict['Construct-Impact-Graph-Phase']))
        pbar.update(1)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt! ", e)
    pbar.close()


    # Use the timezone in my location.
    local_tz = datetime.timezone(datetime.timedelta(hours=8))
    time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    fname = os.path.join("temp_results", "effcause", "timino", f"exp_rets_{time_str}.pkl")
    print("Saving to", fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(exp_rets, f)