#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the EffCause experiments on the PEMS-Bay & METR-La traffic datasets.
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

def get_weekly_pems_df(data_name):
    """
    Params:
        data_name: "pems-bay" or "metr-la"
    """
    df = pd.read_hdf("/workspace/DCRNN/data/{}.h5".format(data_name))
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
    # df = pd.read_hdf("/workspace/DCRNN/data/{}.h5".format("metr-la"))
    # df_list = [df]
    # data = np.load("data/pems-bay/pems-bay.npy")
    # df = pd.DataFrame(data)
    # print("Loaded dataframe shape: ", df.shape)
    # selected_cols = [248, 122, 247, 224, 80, 93]
    # selected_cols = [  1,   2,   8,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
    #     28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  78,  79,  80,
    #     81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
    #     94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106,
    #    107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    #    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
    #    133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
    #    146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
    #    159, 161, 324] # for pems-bay
    selected_cols = [ 51,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,
        99, 100, 206] # for metr-la
    # selected_cols = [1, 2, 3]

    weekly_df = get_weekly_pems_df("metr-la")
    df_list = weekly_df
    # df_list = [df]

    # Search through parameters and save results.
    exp_rets = []
    pbar = tqdm(total=len(df_list))
    try:
        for i, df in enumerate(df_list):
        #     for step in [100, 200, 300]:
        #         for sign in [0.005, 0.01, 0.05]:
        #             for lag in [3]:
            # if i<3:
            #     continue
            step=200
            sign=1e-2
            lag=20
            local_results, dcc, mat, time_stat_dict = effcause_causal_discover(
                # Data params
                df.to_numpy()[:, selected_cols],
                # Granger interval based graph construction params
                step=step,
                significant_thres=sign,
                lag=lag,  # must satisfy: step > 3 * lag + 1
                adaptive_threshold=0.7,
                use_multiprocess=True,
                max_workers=3,
                reuse_invdirection=True,
                rolling_method="zyf",
                min_nobs=50,
                max_segment_len=None,
                adftest=True,
                share_data_by_pickle=False,
                # Debug_params
                verbose=True,
                runtime_debug=False,
            )
            exp_rets.append(({
                "week_id": i, 
                "step": step,
                "sign": sign,
                "lag": lag,
                "local_results": local_results,
                # "dcc": dcc,
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
    fname = os.path.join("temp_results", "effcause", "metr-la", f"exp_rets_{time_str}.pkl")
    print("Saving to", fname)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        pickle.dump(exp_rets, f)
