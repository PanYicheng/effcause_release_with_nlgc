import numpy as np

import time
import datetime
import timeit
import sys

from scipy import stats
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import grangercausalitytests as granger_std
from statsmodels.tsa.stattools import adfuller as adfuller
from statsmodels.tsa.stattools import lagmat2ds as lagmat2ds
from statsmodels.tools.tools import add_constant as add_constant
from statsmodels.regression.linear_model import OLS as OLS

# from src.rolling import RollingOLS
from effcause_lib.rolling_step import RollingOLS


class cnts_prune:
    def __init__(self, cnt_promising, cnt_promising_not, cnt_not_sure, cnt_initial):
        self.cnt_promising = cnt_promising
        self.cnt_promising_not = cnt_promising_not
        self.cnt_not_sure = cnt_not_sure
        self.cnt_initial = cnt_initial

    def __str__(self):
        return "Promising: %05d, PromisingNot: %05d, NotSure: %05d, Initial: %05d" % (
            self.cnt_promising,
            self.cnt_promising_not,
            self.cnt_not_sure,
            self.cnt_initial,
        )


def bidirect_granger(
    array_data,
    array_data_head,
    path_to_output,
    feature,
    target,
    significant_thres,
    test_mode,
    trip,
    lag,
    step,
    max_segment_len,
    min_segment_len,
    rolling_method='pyc',
    min_nobs=25,
    adftest=True,
    verbose=True,
    return_result=False,
):
    """
    Granger causal intervals论文的优化算法
    参数：
        array_data: 时间序列数据，每列为一个变量
        array_data_head: 变量的名称
        path_to_output:
        feature: Granger causality的源变量名称 (feature -> target)
        target: Granger causality的目标变量名称 (feature -> target)
        significant_thres: 假设检验的显著性水平
        test_mode: 不同优化实现方式，这里默认使用最好的优化方式，即fast_version_3
        trip: 选择时间序列的哪个时间段，只在simu_real为real时有效
        lag: Granger causality test的最大历史间隔
        step: 划分区间的最小步长
        max_segment_len:进行因果检验的最大区间长度
        min_segment_len:进行因果检验的最小区间长度
        rolling_method: RollingOLS实现的方式，pyc为正确的，但是很慢；zyf为采取张伊凡的方式，尝试修复错误。
        min_nobs: the min_nobs parameter in rolling regression, only used in zyf mode. This parameter has
            constrains: `min_nobs must be larger than the number of regressors in the model and less than window`
        adftest: whether to conduct the Adf stationarity test. If the data is not stationary, we will
                 reject the any granger test results.
        verbose: whether print detailed info
        return_result: whether return result p value matrix
    """

    addconst = True
    n_sample = len(array_data)

    if int(verbose) == 2:
        print("Sample size: " + str(n_sample))

    fea_cnt = 0

    cnt_prune_YX = cnts_prune(0, 0, 0, 0)
    cnt_prune_XY = cnts_prune(0, 0, 0, 0)

    if int(verbose) == 2:
        print("Feature->Target: {} -> {}".format(feature, target))

    time1 = timeit.default_timer()

    time_granger = 0
    time_OLS = []
    time_adf = []
    time_window = []

    index_target = array_data_head.index(target)
    array_target = array_data[:, index_target : index_target + 1].astype(float)

    index_feature = array_data_head.index(feature)
    array_feature = array_data[:, index_feature : index_feature + 1].astype(float)

    array_YX = np.concatenate((array_target, array_feature), axis=1)
    array_XY = np.concatenate((array_feature, array_target), axis=1)

    # Iteration
    n_step = int(n_sample / step)
    list_segment_split = [step * i for i in range(n_step)]
    if n_sample > step * (n_step):
        list_segment_split.append(n_sample)
    else:
        list_segment_split.append(step * n_step)

    start = 0
    end = 0

    total_cnt_segment_YX = 0
    total_cnt_segment_XY = 0
    total_cnt_segment_adf = 0
    total_cnt_segment_cal_adf = 0
    total_cnt_segment_examine_adf_Y = 0

    array_results_YX = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_results_XY = np.full((n_step + 1, n_step + 1), -2, dtype=float)

    array_adf_results_X = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_adf_results_Y = np.full((n_step + 1, n_step + 1), -2, dtype=float)

    array_res2down_ssr_YX = np.full((n_step + 1, n_step + 1), -2, dtype=float)
    array_res2djoint_ssr_YX = np.full((n_step + 1, n_step + 1), -2, dtype=float)

    for i in range(n_step):
        start = list_segment_split[i]

        reset_cnt_YX = -1
        res2down_YX = None
        res2djoint_YX = None
        res2down_ssr_upper_YX = 0
        res2down_ssr_lower_YX = 0
        res2djoint_ssr_upper_YX = 0
        res2djoint_ssr_lower_YX = 0
        res2djoint_df_resid_YX = 0

        reset_cnt_XY = -1
        res2down_XY = None
        res2djoint_XY = None
        res2down_ssr_upper_XY = 0
        res2down_ssr_lower_XY = 0
        res2djoint_ssr_upper_XY = 0
        res2djoint_ssr_lower_XY = 0
        res2djoint_df_resid_XY = 0

        if test_mode == "v2":
            # YX
            dta, dtaown, dtajoint = get_lagged_data(
                array_YX[start:, :], lag, addconst=True, verbose=False
            )
            rolling_nonnan_inds = {}
            # current_used_inds = {}
            # for _ in ["YX_res2down", "YX_res2down", "XY_res2down", "XY_res2djoint"]:
            #     current_used_inds[_] = 1
            YX_res2down_all, YX_res2djoint_all = fit_regression_rolling(
                dta, dtaown, dtajoint, lag, step=step, rolling_method=rolling_method, min_nobs=min_nobs
            )
            rolling_nonnan_inds["YX_res2down"] = np.logical_not(np.isnan(YX_res2down_all.ssr)).nonzero()[0]
            # assert len(rolling_nonnan_inds["YX_res2down"]) == n_step - i + 1, \
            #     "rolling_nonnan_inds for YX_res2down is not consistent with sliding windows"
            rolling_nonnan_inds["YX_res2djoint"] = np.logical_not(np.isnan(YX_res2djoint_all.ssr)).nonzero()[0]
            # assert len(rolling_nonnan_inds["YX_res2djoint"]) == n_step - i + 1, \
            #     "rolling_nonnan_inds for YX_res2djoint is not consistent with sliding windows"
            # XY
            dta, dtaown, dtajoint = get_lagged_data(
                array_XY[start:, :], lag, addconst=True, verbose=False
            )
            XY_res2down_all, XY_res2djoint_all = fit_regression_rolling(
                dta, dtaown, dtajoint, lag, step=step, rolling_method=rolling_method, min_nobs=min_nobs
            )
            rolling_nonnan_inds["XY_res2down"] = np.logical_not(np.isnan(XY_res2down_all.ssr)).nonzero()[0]
            rolling_nonnan_inds["XY_res2djoint"] = np.logical_not(np.isnan(XY_res2djoint_all.ssr)).nonzero()[0]
            # assert len(rolling_nonnan_inds["XY_res2down"]) == n_step - i + 1, \
            #     "rolling_nonnan_inds for XY_res2down is not consistent with sliding windows"
            # assert len(rolling_nonnan_inds["XY_res2djoint"]) == n_step - i + 1, \
            #     "rolling_nonnan_inds for XY_res2djoint is not consistent with sliding windows"
            if int(verbose) == 2:
                print("rolling_method:", rolling_method)
                print("dta.shape:", dta.shape)
                for _ in ["YX_res2down", "YX_res2down", "XY_res2down", "XY_res2djoint"]:
                    print(_, ":", rolling_nonnan_inds[_])
                print()
            # from IPython.core.debugger import set_trace
            # set_trace() # 断点位置

        for j in range(i + 1, n_step + 1):
            end = list_segment_split[j]
            if int(verbose) == 2:
                print ('Interval: [%d, %d]' % (start, end), end='\n')
            # 如果分段长度过小或者过大都跳过因果检验
            if (
                len(array_YX[start:end, :]) < min_segment_len
                or len(array_YX[start:end, :]) > max_segment_len
            ):
                array_results_YX[i, j] = -2
                array_results_XY[i, j] = -2
                array_adf_results_X[i, j] = -2
                array_adf_results_Y[i, j] = -2
                array_res2down_ssr_YX[i, j] = -2
                array_res2djoint_ssr_YX[i, j] = -2
                continue
            time3 = timeit.default_timer()
            # granger test (standard)
            if test_mode == "standard":
                p_value_YX, res2down_YX, res2djoint_YX = grangercausalitytests(
                    array_YX[start:end, :], lag, addconst=True, verbose=False
                )
                p_value_XY, res2down_XY, res2djoint_XY = grangercausalitytests(
                    array_XY[start:end, :], lag, addconst=True, verbose=False
                )
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

            elif test_mode == "v1":
                total_cnt_segment_YX += 1
                (
                    p_value_YX,
                    res2down_YX,
                    res2djoint_YX,
                    res2down_ssr_upper_YX,
                    res2down_ssr_lower_YX,
                    res2djoint_ssr_upper_YX,
                    res2djoint_ssr_lower_YX,
                    res2djoint_df_resid_YX,
                    reset_cnt_YX,
                    cnt_prune_YX,
                ) = grangercausalitytests_check_F_upper_lower(
                    array_YX[start:end, :],
                    lag,
                    res2down_YX,
                    res2djoint_YX,
                    res2down_ssr_upper_YX,
                    res2down_ssr_lower_YX,
                    res2djoint_ssr_upper_YX,
                    res2djoint_ssr_lower_YX,
                    res2djoint_df_resid_YX,
                    significant_thres,
                    step,
                    reset_cnt_YX,
                    cnt_prune_YX,
                    addconst=True,
                    verbose=False,
                )

                total_cnt_segment_XY += 1
                (
                    p_value_XY,
                    res2down_XY,
                    res2djoint_XY,
                    res2down_ssr_upper_XY,
                    res2down_ssr_lower_XY,
                    res2djoint_ssr_upper_XY,
                    res2djoint_ssr_lower_XY,
                    res2djoint_df_resid_XY,
                    reset_cnt_XY,
                    cnt_prune_XY,
                ) = grangercausalitytests_check_F_upper_lower(
                    array_XY[start:end, :],
                    lag,
                    res2down_XY,
                    res2djoint_XY,
                    res2down_ssr_upper_XY,
                    res2down_ssr_lower_XY,
                    res2djoint_ssr_upper_XY,
                    res2djoint_ssr_lower_XY,
                    res2djoint_df_resid_XY,
                    significant_thres,
                    step,
                    reset_cnt_XY,
                    cnt_prune_XY,
                    addconst=True,
                    verbose=False,
                )

                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

                array_res2down_ssr_YX[i, j] = res2down_ssr_upper_YX
                array_res2djoint_ssr_YX[i, j] = res2djoint_ssr_lower_YX

            elif test_mode == "v2":
                # print("Calculating p_value_YX")
                p_value_YX = grangercausalitytests_with_model(
                    YX_res2down_all,
                    YX_res2djoint_all,
                    lag,
                    end - n_sample - 1,
                    addconst=True,
                    verbose=verbose,
                )
                # print("Calculating p_value_XY")
                p_value_XY = grangercausalitytests_with_model(
                    XY_res2down_all,
                    XY_res2djoint_all,
                    lag,
                    end - n_sample - 1,
                    addconst=True,
                    verbose=verbose,
                )
                array_results_YX[i, j] = p_value_YX
                array_results_XY[i, j] = p_value_XY

            time4 = timeit.default_timer()

            time_granger += time4 - time3

            # stationary test
            time5 = timeit.default_timer()

            total_cnt_segment_adf += 1

            if (
                p_value_YX < significant_thres
                and p_value_YX != -1
                and p_value_XY > significant_thres
                and adftest
            ) or (
                p_value_XY < significant_thres
                and p_value_XY != -1
                and p_value_YX > significant_thres
                and adftest
            ):

                total_cnt_segment_examine_adf_Y += 1

                (
                    adfstat_Y,
                    pvalue_Y,
                    usedlag_Y,
                    nobs_Y,
                    critvalues_Y,
                    icbest_Y,
                ) = adfuller(array_XY[start:end, 1], lag)
                array_adf_results_Y[i, j] = pvalue_Y

                if pvalue_Y < significant_thres:
                    (
                        adfstat_X,
                        pvalue_X,
                        usedlag_X,
                        nobs_X,
                        critvalues_X,
                        icbest_X,
                    ) = adfuller(array_XY[start:end, 0], lag)
                    array_adf_results_X[i, j] = pvalue_X
                    total_cnt_segment_cal_adf += 1
                else:
                    pvalue_X = -1
                    array_adf_results_X[i, j] = pvalue_X
            else:
                pvalue_Y = -1
                pvalue_X = -1
                array_adf_results_Y[i, j] = pvalue_Y
                array_adf_results_X[i, j] = pvalue_X

            # reject the granger result

            if pvalue_Y > significant_thres or pvalue_X > significant_thres:
                array_results_YX[i, j] = -1
                array_results_XY[i, j] = -1

            time6 = timeit.default_timer()

            time_adf.append((time6 - time5))
            time_window.append(time6 - time3)

    time2 = timeit.default_timer()
    total_time = time2 - time1

    if not return_result:
        np.save(
            path_to_output + (target + "_caused_by_" + feature).replace("/", "-"),
            array_results_YX,
        )
        np.save(
            path_to_output + (feature + "_caused_by_" + target).replace("/", "-"),
            array_results_XY,
        )

    fea_cnt += 1
    if int(verbose) == 2:
        print("Prune for Y <-- X:\n", " " * 5, cnt_prune_YX)
        print("Prune for X <-- Y:\n", " " * 5, cnt_prune_XY)
    if return_result:
        return (
            {
                "total_time": total_time,
                "time_OLS": time_OLS,
                "time_granger": time_granger,
                "time_adf": time_adf,
                "time_window": time_window,
                "cnt_prune_YX": cnt_prune_YX,
                "cnt_prune_XY": cnt_prune_XY,
            },
            array_results_YX,
            array_results_XY,
        )
    return {
        "total_time": total_time,
        "time_OLS": time_OLS,
        "time_granger": time_granger,
        "time_adf": time_adf,
        "time_window": time_window,
        "cnt_prune_YX": cnt_prune_YX,
        "cnt_prune_XY": cnt_prune_XY,
    }


def get_lagged_data(x, lag, addconst, verbose):
    """
    生成lag矩阵

    对于x=[Y X], 生成每一行包含 [Y_t Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}] 的数据

    Returns:
        dta: 整个lag矩阵
        dtaown: 只包含自己过去lag的矩阵, 即[Y_{t-1} Y_{t-2} ... Y_{t-lag}]
        dtajoint: 包含自己和其他变量过去lag时刻的数据，即 [Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}]
     """
    x = np.asarray(x)
    if x.shape[0] <= 3 * lag + int(addconst):
        raise ValueError(
            "Insufficient observations. Maximum allowable "
            "lag is {0}".format(int((x.shape[0] - int(addconst)) / 3) - 1)
        )
    # create lagmat of both time series
    """
    例如对于列变量为[Y_t, X_t]的数据，调用下面语句之后会列变量变成
    [Y_t Y_{t-1} Y_{t-2} ... Y_{t-lag} X_{t-1} X_{t-2} ... X_{t-lag}] 
     """
    dta = lagmat2ds(x, lag, trim="both", dropex=1)
    # add constant
    if addconst:
        dtaown = add_constant(dta[:, 1 : (lag + 1)], prepend=False, has_constant="add")
        dtajoint = add_constant(dta[:, 1:], prepend=False, has_constant="add")
    else:
        dtaown = dta[:, 1 : (lag + 1)]
        dtajoint = dta[:, 1:]
    if verbose:
        print("lagged transform:", x.shape, "-->", dta.shape)
    return dta, dtaown, dtajoint


def fit_regression_rolling(dta, dtaown, dtajoint, lag, step=1, rolling_method='pyc', 
    min_nobs=25):
    """
    在线更新的线性拟合方法

    Modify the RollingOLS process.
    In order to keep the two regression models use the same number of samples, we must set their min_obs to the same value explicitly. Otherwise it will be automatically set to different values. In addition, in our sliding window scheme, the first window is [0:step]. And the `min_nobs` is just the first regression window, we can set `min_obs` to `step-(dtaown.shape[1]-1)` to conduct the first regression. Here, we substract (dtaown.shape[1]-1) because these observations are trimmed when constructing the `dta,dtaown,dtajoint`.
    From the RollingOLS implementation, we get new constraints: `step-(dtaown.shape[1]-1) >= num_regressors`, i.e. `step-(dtaown.shape[1]-1) >= lag` and `step-(dtaown.shape[1]-1) >= 2*lag`
    Thus, step >= 3*lag.
    @Date: 2021-09-13 by pyc.
    """
    if rolling_method == 'pyc':
        res2down_all = RollingOLS(
            dta[:, 0], dtaown, window=len(dta), step=step, 
            min_nobs=step-(dtaown.shape[1]-1), 
            # min_nobs=20,
            expanding=True
        ).fit(num_trimmed=(dtaown.shape[1]-1))
        res2djoint_all = RollingOLS(
            dta[:, 0], dtajoint, window=len(dta), step=step, 
            min_nobs=step-(dtaown.shape[1]-1), 
            # min_nobs=20,
            expanding=True
        ).fit(num_trimmed=(dtaown.shape[1]-1))
    elif rolling_method == 'zyf':
        res2down_all = RollingOLS(
            dta[:, 0], dtaown, window=len(dta), step=step, 
            # min_nobs=step-(dtaown.shape[1]-1), 
            min_nobs=min_nobs,
            expanding=True
        ).fit_zyf(num_trimmed=(dtaown.shape[1]-1))
        res2djoint_all = RollingOLS(
            dta[:, 0], dtajoint, window=len(dta), step=step, 
            # min_nobs=step-(dtaown.shape[1]-1), 
            min_nobs=min_nobs,
            expanding=True
        ).fit_zyf(num_trimmed=(dtaown.shape[1]-1))
    else:
        raise NotImplemented("This rolling linear regression method is not implemented!")
    return res2down_all, res2djoint_all


def fit_regression(dta, dtaown, dtajoint):
    """
    针对部分模型和全模型进行两次线性拟合，并返回结果  
     """
    # Run ols on both models without and with lags of second variable

    res2down = OLS(dta[:, 0], dtaown).fit()
    res2djoint = OLS(dta[:, 0], dtajoint).fit()

    return res2down, res2djoint


def f_test(res2down, res2djoint, lag):
    """ 
     根据拟合结果进行F统计检验，返回统计值

     Returns: a dict {'ssr_ftest':
                            (F-statistics, 
                            stats.f.sf(fgc1, lag, res2djoint.df_resid)(p_value),
                            res2djoint.df_resid(完全模型剩余自由度)), 
                            lag)
                         }
     """
    result = {}

    # Granger Causality test using ssr (F statistic)
    # TODO: possible divide by 0
    fgc1 = (res2down.ssr - res2djoint.ssr) / res2djoint.ssr / lag * res2djoint.df_resid

    result["ssr_ftest"] = (
        fgc1,
        stats.f.sf(fgc1, lag, res2djoint.df_resid),
        res2djoint.df_resid,
        lag,
    )

    return result


def grangercausalitytests_with_model(
    res2down_all, res2djoint_all, lag, index, addconst=True, verbose=False
):
    fgc1 = (
        (res2down_all.ssr[index] - res2djoint_all.ssr[index])
        / res2djoint_all.ssr[index]
        / lag
        * res2djoint_all.df_resid[index]
    )
    p_value = stats.f.sf(fgc1, lag, res2djoint_all.df_resid[index])
    if int(verbose) == 2:
        print("sse_part={}\nsse_full={}\nlag={}\ndf_resid_full={}\nindex={}".format(
            res2down_all.ssr[index], res2djoint_all.ssr[index],
            lag, res2djoint_all.df_resid[index], index
        ))
    return p_value


def grangercausalitytests(x, lag, addconst=True, verbose=False):
    """
     采用自定义的方法进行Granger causality检验，只对lag进行f test。

     而statsmodels里的方法会对从1到lag的间隔都采取4种假设检验，效率较低。
     """
    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)

    res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)

    result = f_test(res2down, res2djoint, lag)

    p_value = result["ssr_ftest"][1]

    return p_value, res2down, res2djoint


def grangercausalitytests_check_F_upper_lower(
    x,
    lag,
    pre_res2down,
    pre_res2djoint,
    pre_res2down_ssr_upper,
    pre_res2down_ssr_lower,
    pre_res2djoint_ssr_upper,
    pre_res2djoint_ssr_lower,
    pre_res2djoint_df_resid,
    significant_thres,
    step,
    cnt,
    cnt_prune,
    addconst=True,
    verbose=False,
):

    dta, dtaown, dtajoint = get_lagged_data(x, lag, addconst, verbose)

    if cnt == -1:
        # initialization
        res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
        result = f_test(res2down, res2djoint, lag)
        p_value = result["ssr_ftest"][1]

        res2down_ssr_upper = res2down.ssr
        res2down_ssr_lower = res2down.ssr
        res2djoint_ssr_upper = res2djoint.ssr
        res2djoint_ssr_lower = res2djoint.ssr

        cnt_prune.cnt_initial += 1

        return (
            p_value,
            res2down,
            res2djoint,
            res2down_ssr_upper,
            res2down_ssr_lower,
            res2djoint_ssr_upper,
            res2djoint_ssr_lower,
            res2djoint.df_resid,
            0,
            cnt_prune,
        )
    else:
        # fit the new data
        # prune promising not
        res2down_fit_new_point_error = (
            np.dot(dtaown[-step:, :], pre_res2down.params) - dta[-step:, 0]
        )
        res2djoint_fit_new_point_error = (
            np.dot(dtajoint[-step:, :], pre_res2djoint.params) - dta[-step:, 0]
        )

        res2down_ssr_upper = (
            np.dot(res2down_fit_new_point_error, res2down_fit_new_point_error)
            + pre_res2down_ssr_upper
        )

        res2djoint_ssr_lower = 0
        res2down_ssr_lower = 0
        if len(dta) > lag * step * step:
            model_own, model_joint = fit_regression(
                dta[-step:, :], dtaown[-step:, :], dtajoint[-step:, :]
            )
            res2djoint_ssr_lower = pre_res2djoint_ssr_lower + model_joint.ssr
            res2down_ssr_lower = pre_res2down_ssr_lower + model_own.ssr
        else:
            res2djoint_ssr_lower = (
                pre_res2djoint_ssr_lower
            )  # + np.linalg.norm(np.cov(dtajoint[-step:, :])) / (step * step * lag * lag * 4)
            res2down_ssr_lower = (
                pre_res2down_ssr_lower
            )  # + np.linalg.norm(np.cov(dtaown[-step:, :])) / (step * step * lag * lag)

        res2djoint_ssr_upper = (
            np.dot(res2djoint_fit_new_point_error, res2djoint_fit_new_point_error)
            + pre_res2djoint_ssr_upper
        )

        non_zero_column = np.sum(np.sum(dtajoint[:, :], axis=0) != 0)
        res2djoint_df_resid = len(dtajoint) - non_zero_column

        # check F_upper
        F_upper = (
            (res2down_ssr_upper / res2djoint_ssr_lower - 1)
            * (res2djoint_df_resid)
            / lag
        )
        p_value_lower = 1 - stats.f.cdf(F_upper, lag, (res2djoint_df_resid))

        if p_value_lower >= significant_thres:  # promising not
            p_value = 1
            cnt_prune.cnt_promising_not += 1

            return (
                p_value,
                pre_res2down,
                pre_res2djoint,
                res2down_ssr_upper,
                res2down_ssr_lower,
                res2djoint_ssr_upper,
                res2djoint_ssr_lower,
                res2djoint_df_resid,
                cnt + 1,
                cnt_prune,
            )

        else:
            # check F_lower
            F_lower = (
                (res2down_ssr_lower / res2djoint_ssr_upper - 1)
                * (res2djoint_df_resid)
                / lag
            )
            p_value_upper = 1 - stats.f.cdf(F_lower, lag, (res2djoint_df_resid))

            if p_value_upper < significant_thres:
                # promising
                p_value = 0
                cnt_prune.cnt_promising += 1
                return (
                    p_value,
                    pre_res2down,
                    pre_res2djoint,
                    res2down_ssr_upper,
                    res2down_ssr_lower,
                    res2djoint_ssr_upper,
                    res2djoint_ssr_lower,
                    res2djoint_df_resid,
                    cnt + 1,
                    cnt_prune,
                )

            else:
                # not sure
                res2down, res2djoint = fit_regression(dta, dtaown, dtajoint)
                result = f_test(res2down, res2djoint, lag)
                p_value = result["ssr_ftest"][1]

                res2down_ssr_upper = res2down.ssr
                res2down_ssr_lower = res2down.ssr
                res2djoint_ssr_upper = res2djoint.ssr
                res2djoint_ssr_lower = res2djoint.ssr

                cnt_prune.cnt_not_sure += 1

                return (
                    p_value,
                    res2down,
                    res2djoint,
                    res2down_ssr_upper,
                    res2down_ssr_lower,
                    res2djoint_ssr_upper,
                    res2djoint_ssr_lower,
                    res2djoint.df_resid,
                    cnt + 1,
                    cnt_prune,
                )

