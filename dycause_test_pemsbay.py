#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the DyCause experiments on the TCDF finance datasets.
"""
from main_dycause_mp_new import dycause_causal_discover

import matplotlib.pyplot as plt
from tqdm import tqdm
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


def get_weekly_pems_df():
    df = pd.read_hdf("/workspace/DCRNN/data/pems-bay.h5")
    print("Orignial dataframe shape:", df.shape)
    weekly_df = []
    s = df.index[0]
    while s <= df.index[-1]:
        e = s + pd.offsets.Week()
        # pandas in default includes the end point, so
        # we skip that data by using a second slide [::-1].
        # However, the last week data is not complete (not 7 days full),
        # so we don't need to skip the last.
        if e>df.index[-1]:
            weekly_df.append(df[s:e])
        else:
            weekly_df.append(df[s:e][:-1])
    #     print(df[s:e].shape)
    #     break
        s = e
    for i in weekly_df:
        print(i.shape, end=', ')
    print("Total data samples of weekly sliced dataframes:", sum([i.shape[0] for i in weekly_df]))
    return weekly_df




if __name__ == "__main__":
    # datasets, gt_graphs = load_tcdf_data(dir="finance")
    weekly_df = get_weekly_pems_df()
    # Search through parameters and save results.
    exp_rets = []
    # pbar = tqdm(total=len(datasets) * 1 * 3 * 3 * 1)
    # for i, data in enumerate(datasets):
    #     for T in range(4000, 5000, 1000):
    #         for step in [100, 200, 300]:
    #             for sign in [0.005, 0.01, 0.05]:
    #                 for lag in [3]:
    try:
        step=200
        sign=1e-2
        lag=20
        local_results, dcc, mat, time_stat_dict = dycause_causal_discover(
            # Data params
            weekly_df[0].to_numpy()[:, :],
            # Granger interval based graph construction params
            step=step,
            significant_thres=sign,
            lag=lag,  # must satisfy: step > 3 * lag + 1
            adaptive_threshold=0.7,
            use_multiprocess=True,
            max_workers=3,
            # Debug_params
            verbose=2,
            runtime_debug=True,
        )
        exp_rets.append(({
            "week_id": 0, 
            "step": step,
            "sign": sign,
            "lag": lag,
            "local_results": local_results,
            "dcc": dcc,
            "time_stat_dict": time_stat_dict
        }))
        # pbar.set_description("{:.2f} s".format(time_stat_dict['Construct-Impact-Graph-Phase']))
        # pbar.update(1)
    except KeyboardInterrupt as e:
        print("Keyboard interrupt! ", e)
    # pbar.close()


    # Use the timezone in my location.
    local_tz = datetime.timezone(datetime.timedelta(hours=8))
    time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
    fname = os.path.join("temp_results", "dycause", "pems-bay", f"exp_rets_{time_str}.pkl")
    print("Saving to", fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(exp_rets, f)