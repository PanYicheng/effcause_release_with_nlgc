#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the DyCause experiments on the TCDF datasets.
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

datasets, gt_graphs = load_tcdf_data(dir="fmri")
datasets_, gt_graphs_ = [], []
for i in range(len(datasets)):
    if datasets[i].shape[0]>1000:
        datasets_.append(datasets[i])
        gt_graphs_.append(gt_graphs[i])
datasets, gt_graphs = datasets_, gt_graphs_
print("Num of datasets:", len(datasets))
# Search through parameters and save results.
exp_rets = []
pbar = tqdm(total=len(datasets) * 1 * 1 * 3 * 1)
for i, data in enumerate(datasets):
    # for T in range(4000, 5000, 1000):
    # for step in [100, 200, 300]:
    for sign in [0.005, 0.01, 0.05]:
        for lag in [3]:
            local_results, dcc, mat, time_stat_dict = dycause_causal_discover(
                # Data params
                data.to_numpy(),
                # Granger interval based graph construction params
                step=data.shape[0],
                significant_thres=sign,
                lag=lag,  # must satisfy: step > 3 * lag + 1
                adaptive_threshold=0.7,
                use_multiprocess=True,
                max_workers=3,
                # Debug_params
                verbose=0,
                runtime_debug=True,
            )
            exp_rets.append(({
                "dataset_id": i, 
                # "T": T,
                "step": data.shape[0],
                "sign": sign,
                "lag": lag,
                "local_results": local_results,
                "dcc": dcc,
                "time_stat_dict": time_stat_dict
            }))
            pbar.set_description("{:.2f} s".format(time_stat_dict['Construct-Impact-Graph-Phase']))
            pbar.update(1)
pbar.close()


# Use the timezone in my location.
local_tz = datetime.timezone(datetime.timedelta(hours=8))
time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
fname = os.path.join("temp_results", "basicgranger", "tcdf_fmri", f"exp_rets_{time_str}.pkl")
print("Saving to", fname)
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname, "wb") as f:
    pickle.dump(exp_rets, f)