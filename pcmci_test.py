#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This test code runs the PCMCI experiments on the TCDF finance datasets.
"""
import time
import datetime
from tqdm.auto import tqdm
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
import os
import pickle
from tigramite.independence_tests import ParCorr
from utility_funcs.proc_data import load_tcdf_data

datasets, gt_graphs = load_tcdf_data(dir="finance")

def run_pcmci(data, tau_max=3, pc_alpha=0.1, verbosity=0):
    """Run PCMCI algorithm and get intermidiate results (p value matrix ...).
    
    Params:
        data: numpy array of shape [T, N]. T is the number of samples and N is the number of variables.
        tau_max:
        pc_alpha: the PC1 algorithm alpha value. If set to None, PCMCI will automatically choose a p value.
    """
    dataframe = pp.DataFrame(data)
    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=verbosity)
    pcmci_res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    # pcmci.print_significant_links(p_matrix=results['p_matrix'],
    #                                      val_matrix=results['val_matrix'],
    #                                      alpha_level=0.1)
    return pcmci_res


# Search through parameters and save results.
exp_rets = []
pbar = tqdm(total=len(datasets) * 2)
for i, data in enumerate(datasets):
    for pc_alpha in [None, 0.1]:
        tic = time.time()
        pcmci_res = run_pcmci(data.to_numpy(), pc_alpha=pc_alpha)
        toc = time.time()
        exp_rets.append({
            "dataset_id": i, 
            "pc_alpha": pc_alpha, 
            "pcmci_res": pcmci_res,
            "time": toc-tic
        })
        pbar.desc = "{:.2f} s".format(toc-tic)
        pbar.update(1)
pbar.close()

# Use the timezone in my location.
local_tz = datetime.timezone(datetime.timedelta(hours=8))
time_str = datetime.datetime.now(local_tz).strftime("%Y%m%d_%H%M%S")
fname = os.path.join("temp_results", "pcmci", "tcdf_finance", f"exp_rets_{time_str}.pkl")
print("Saving to", fname)
os.makedirs(os.path.dirname(fname), exist_ok=True)
with open(fname, "wb") as f:
    pickle.dump(exp_rets, f)