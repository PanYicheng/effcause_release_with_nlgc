#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the Basic Granger experiments on the TCDF finance datasets.
We reuse the code of DyCause by setting the step to the total data length.
"""
from main_dycause_mp_new import dycause_causal_discover

from tqdm import tqdm
import datetime
import os
import pickle
from utility_funcs.proc_data import load_tcdf_data

datasets, gt_graphs = load_tcdf_data(dir="finance")
# Search through parameters and save results.
exp_rets = []
pbar = tqdm(total=len(datasets) * 4)
for i, data in enumerate(datasets):
    for T in range(4000, 5000, 1000):
        for sign in [0.005, 0.01, 0.05, 0.1]:
            for lag in [3]:
                local_results, dcc, mat, time_stat_dict = dycause_causal_discover(
                    # Data params
                    data.to_numpy()[0:T, :],
                    # Granger interval based graph construction params
                    step=T,
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
                    "dataset_id": i, 
                    "T": T,
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
fname = os.path.join("temp_results", "basicgranger", "tcdf_finance", f"exp_rets_{time_str}.pkl")
print("Saving to", fname)
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname, "wb") as f:
    pickle.dump(exp_rets, f)