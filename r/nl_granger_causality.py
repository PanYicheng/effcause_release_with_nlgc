import os

from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter


robjects.r(f"source(\"{os.path.join(os.path.dirname(__file__), 'nl_gc.R')}\")")


def boot_nonlinear_granger(x1, x2, pwanted=4, px1=4, px2=4, n999=9):
    fun = robjects.r["nlgc"]
    with localconverter(default_converter + numpy2ri.converter):
        ret = fun(x1, x2, pwanted=pwanted, px1=px1, px2=px2, n999=n999)
    # print(ret, type(ret))
    return ret


if __name__ == "__main__":
    # import pandas as pd
    # df = pd.read_csv("/workspace/ChickEgg.csv", sep=" ")
    # x1 = df.iloc[:, 0].to_numpy()
    # x2 = df.iloc[:, 1].to_numpy()
    # ret = boot_nonlinear_granger(x1, x2, pwanted=3, px1=3, px2=3, n999=9)
    # print(ret)
    # import pdb; pdb.set_trace()
    import sys
    sys.path.append("../")
    from utility_funcs.proc_data import load_tcdf_data

    datasets, gt_graphs = load_tcdf_data(dir="finance")

    datasets_, gt_graphs_ = [], []
    for i in range(len(datasets)):
        if datasets[i].shape[0] > 1000:
            datasets_.append(datasets[i])
            gt_graphs_.append(gt_graphs[i])
    datasets, gt_graphs = datasets_, gt_graphs_
    lag = 3
    for data in datasets:
        x1 = data.iloc[:, 0].to_numpy()
        x2 = data.iloc[:, 1].to_numpy()
        ret = boot_nonlinear_granger(x1, x2, pwanted=lag, px1=lag, px2=lag, n999=9)
        print(ret)