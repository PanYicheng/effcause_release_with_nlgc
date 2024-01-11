import logging
import os
import pickle
import datetime
from threading import local
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from multiprocessing import Manager

from tqdm.auto import tqdm

from dycause_lib.Granger_all_code_pair import loop_granger_pair
from dycause_lib.causal_graph_build import (
    get_segment_split,
    get_ordered_intervals,
    get_overlay_count,
)
from utility_funcs.proc_data import safe_dump_obj


class GrangerCausalRunner:
    """GrangerCausalRunner is the class used for running the Granger causal interval algorithm.
    This class holds most of the static parameters passed to underlying loop_granger function.
    """

    def __init__(
        self,
        dir_output,
        step,
        lag,
        significant_thres,
        max_segment_len,
        min_segment_len,
        method="fast_version_3",
        verbose=False,
        runtime_debug=False,
        *args,
        **kws,
    ):
        self.dir_output = dir_output
        self.step = step
        self.lag = lag
        self.significant_thres = significant_thres
        self.method = method
        self.max_segment_len = max_segment_len
        self.min_segment_len = min_segment_len
        self.verbose = verbose
        self.runtime_debug = runtime_debug
        self._args = args
        self._kws = kws

    def if_verbose(self, level):
        """Judge whether display info at this level. Affected by the verbose level.
        verbose 1) False: display minimal
                2) True: equal to 1.0
                3) Some integer > 1.
        Args:
            level ([int]): the verbose level of current info.
        Returns:
            True: should display
            False: should not display
        """
        if not self.verbose:
            return False
        elif self.verbose >= level:
            return True

    def pairwise_granger_intervals(self, array_feature, array_target):
        return loop_granger_pair(
            array_feature,
            array_target,
            self.dir_output,
            self.significant_thres,
            self.method,
            -1,
            self.lag,
            self.step,
            "simu",
            self.max_segment_len,
            self.min_segment_len,
            verbose=self.if_verbose(3),
            return_result=True,
        )

    def pairwise_process(self, param_dict, specific_params, result_dict):
        """The default process function that calcuates the Granger causal intervals of
        one pair. To enumerate all pairs, we must create N*(N-1) jobs running this function.
        """
        # try:
        # Load data from disk file.
        data_path = param_dict["data"]
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        x_i, y_i = specific_params["x_i"], specific_params["y_i"]
        ret = self.pairwise_granger_intervals(
            data[:, x_i].reshape(-1, 1), data[:, y_i].reshape(-1, 1)
        )
        result_dict["{}->{}".format(x_i, y_i)] = ret
        # resu_path = shared_params["resu"]
        # with open(os.path.join(resu_path, f"{x_i}->{y_i}.pkl"), "wb") as f:
        #     pickle.dump(ret, f)
        # except Exception as e:
        #     logging.error(e)
        #     from IPython.core.debugger import set_trace
        #     set_trace()
        return

    def allpairs_process(self, param_dict, proc_idx, result_dict):
        """A different multiprocess runner function that enumerate every pairs.
        In this implementation, the num of process is given and each process calculates
        many pairs.
        """
        # Load data from disk file.
        data_path = param_dict["data"]
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        N = param_dict["N"]
        max_workers = param_dict["max_workers"]
        i = 0
        # Only the first process is able to show the progress bar.
        if proc_idx == 0:
            pbar = tqdm(total=N * (N-1), ascii=True, desc=f"{max_workers} Processes")
        for x_i in range(N):
            for y_i in range(N):
                if x_i == y_i:
                    continue
                if i % max_workers == proc_idx:
                    # self.pairwise_process(param_dict, {'x_i': x_i, 'y_i': y_i}, result_dict)
                    ret = self.pairwise_granger_intervals(
                        data[:, x_i].reshape(-1, 1), data[:, y_i].reshape(-1, 1)
                    )
                    result_dict["{}->{}".format(x_i, y_i)] = ret
                if proc_idx == 0:
                    pbar.update(1)
                i += 1
        if proc_idx == 0:
            pbar.close()
        return

    def get_ordered_intervals(self, matrics, local_length):
        list_segment_split = get_segment_split(local_length, self.step)
        return get_ordered_intervals(
            matrics, self.significant_thres, list_segment_split
        )


class TemporalAnalyze:
    """TemporalAnalyze class manages the execution of the Granger causal interval algorithm in multiple processes.
    In order to work, analyze(...) function should only be executed in the process with __main__ name.
    """

    def __init__(
        self,
        dir_output,
        step,
        lag,
        significant_thres,
        max_segment_len,
        min_segment_len,
        method="fast_version_3",
        verbose=False,
        runtime_debug=False,
        *args,
        **kws,
    ):
        """[summary]

        Args:
            dir_output ([type]): [description]
            step ([type]): [description]
            lag ([type]): [description]
            significant_thres ([type]): [description]
            max_segment_len ([type]): [description]
            min_segment_len ([type]): [description]
            method (str, optional): [description]. Defaults to "fast_version_3".
            verbose (bool, optional): [description]. Defaults to False.
            runtime_debug (bool, optional): [description]. Defaults to False.
        """
        self.runner = GrangerCausalRunner(
            dir_output,
            step,
            lag,
            significant_thres,
            max_segment_len,
            min_segment_len,
            method=method,
            verbose=verbose,
            runtime_debug=runtime_debug,
            *args,
            **kws,
        )
        self.verbose = verbose
        self.runtime_debug = runtime_debug
        self.dir_output = dir_output

    def if_verbose(self, level):
        """Judge whether display info at this level. Affected by the verbose level.
        verbose 1) False: display minimal
                2) True: equal to 1.0
                3) Some integer > 1.
        Args:
            level ([int]): the verbose level of current info.
        Returns:
            True: should display
            False: should not display
        """
        if not self.verbose:
            return False
        elif self.verbose >= level:
            return True

    def granger_analyze(self, data, use_multiprocess=True, mp_mode=2, **kws):
        """Perform the Granger causality interval analysis.

        Args:
            data ([type]): Input multivariate time series data for analysis. 
                Shape [T, N]. T is time length, N is num of variables.
            use_multiprocess: whether use multiprocess to accelerate. The number of process is set by max_workers. Default to 5. 
            mp_mode: The mode of using multiprocessing library. mode 1 creates a job for each pair of the dynamic Granger test.
                mode 2 creates only a job for each process and iterates the pairs in each job. Then each process only calculates 
                its own pairs.
            kws : The coefficients of selecting data. Required only for caching analysis results to the disk.
        """
        tic = time.time()
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.if_verbose(1):
            print("{:.^80}".format("granger causal intervals"))
            print("{:<10}".format("") + f"Starting at {time_str:^80}")
        local_length, N = data.shape

        try:
            local_results_file_path = os.path.join(
                self.dir_output,
                "local-results",
                "aggregate-{}".format(kws["aggre_delta"]),
                "local_results"
                "_start{start}_bef{bef}_aft{aft}_lag{lag}_sig{sig}_step{step}_min{min}_max{max}.pkl".format(
                    start=kws["start"],
                    bef=kws["bef"],
                    aft=kws["aft"],
                    lag=self.runner.lag,
                    sig=self.runner.significant_thres,
                    step=self.runner.step,
                    min=self.runner.min_segment_len,
                    max=self.runner.max_segment_len,
                ),
            )
        except KeyError as e:
            logging.warning(
                "granger_analyze(...) doesn't have enough parameters to name the cache file. Disabling caching."
            )
            local_results_file_path = None
        except Exception as e:
            logging.error(f"Unknown error {e} occurs.")
        if (
            local_results_file_path is not None
            and not self.runtime_debug
            and os.path.exists(local_results_file_path)
        ):
            if self.if_verbose(1):
                # verbose level >= 3: print granger causal interval loading info
                print(
                    "{:^10}".format("")
                    + "Loading previous granger causal interval results:",
                    os.path.basename(local_results_file_path),
                )
            with open(local_results_file_path, "rb") as f:
                local_results = pickle.load(f)
        else:
            local_results = defaultdict(dict)
            num_jobs = [N * (N - 1)]
            max_workers = kws["max_workers"] if "max_workers" in kws else 5
            

            param_dict = {
                "data": os.path.join(self.dir_output, f"temp_data_{time_str}.pkl")
            }
            safe_dump_obj(data, param_dict["data"])
            if use_multiprocess and max_workers > 1:
                if mp_mode == 1:
                    # region Parallel method 1. Parallel each pair of Granger test.
                    # verbose level >= 2: print granger causal interval progress bar
                    if self.if_verbose(2):
                        pbar = tqdm(total=num_jobs[0], ascii=True)
                    manager = Manager()
                    result_dict = manager.dict()
                    if self.if_verbose(2):
                        pbar.desc = f"{max_workers} processes"
                    executor = ProcessPoolExecutor(max_workers=max_workers)
                    futures = []
                    for x_i in range(N):
                        for y_i in range(N):
                            if x_i == y_i:
                                continue
                            futures.append(
                                executor.submit(
                                    self.runner.pairwise_process,
                                    param_dict,
                                    {"x_i": x_i, "y_i": y_i},
                                    result_dict,
                                )
                            )
                    future_complete_time = []
                    if self.if_verbose(2):
                        for future in as_completed(futures):
                            pbar.update(1)
                            future_complete_time.append(time.time() - tic)
                        safe_dump_obj(
                            future_complete_time,
                            os.path.join(
                                self.dir_output,
                                "runtime-data",
                                f"future-complete-time-{time_str}.pkl",
                            ),
                        )
                    executor.shutdown(wait=True)
                    # verbose level >= 2: close progress bar in calculating granger causal interval
                    if self.if_verbose(2):
                        pbar.close()
                    # endregion
                elif mp_mode == 2:
                    # region Parallel method 2 (Insight from EffCause). Create n process and loop all pairs in each process.
                    # But only process only does its own jobs. This way may use less time in creating and destroying processes.
                    param_dict["N"] = N
                    param_dict["max_workers"] = max_workers
                    manager = Manager()
                    result_dict = manager.dict()
                    executor = ProcessPoolExecutor(max_workers=max_workers)
                    futures = []
                    for i in range(max_workers):
                        futures.append(
                            executor.submit(
                                self.runner.allpairs_process,
                                param_dict,
                                i,
                                result_dict,
                            )
                        )
                    executor.shutdown(wait=True)
                else:
                    raise NotImplementedError("multiprocess mode not supported!")
            else:
                # Single process version.
                # verbose level >= 2: print granger causal interval progress bar
                if self.if_verbose(2):
                    pbar = tqdm(total=num_jobs[0], ascii=True, desc="Single process")
                result_dict = {}
                for x_i in range(N):
                    for y_i in range(N):
                        if x_i == y_i:
                            continue
                        self.runner.pairwise_process(
                            param_dict, {"x_i": x_i, "y_i": y_i}, result_dict
                        )
                        if self.if_verbose(2):
                            pbar.update(1)
                # verbose level >= 2: close progress bar in calculating granger causal interval
                if self.if_verbose(2):
                    pbar.close()
            
            # Get results from the executor.
            time_stat_dict = {}
            if self.runtime_debug:
                time_stat_dict["time_total"] = 0
                time_stat_dict["time_OLS"] = []
                time_stat_dict["time_window"] = []
                time_stat_dict["time_granger"] = 0
                time_stat_dict["time_adf"] = []
            i = 0
            for x_i in range(N):
                for y_i in range(N):
                    if x_i == y_i:
                        continue
                    time_dict, array_results_YX, array_results_XY = result_dict["{}->{}".format(x_i, y_i)]
                    total_time = time_dict['total_time']
                    time_OLS = time_dict['time_OLS']
                    time_granger = time_dict['time_granger']
                    time_adf = time_dict['time_adf']
                    time_window = time_dict['time_window']
                    if self.runtime_debug:
                        time_stat_dict["time_total"] += total_time
                        time_stat_dict["time_OLS"].append(time_OLS)
                        time_stat_dict["time_window"].append(time_window)
                        time_stat_dict["time_granger"] += time_granger
                        time_stat_dict["time_adf"].append(time_adf)
                    if array_results_YX is None and array_results_XY is None:
                        if self.if_verbose(3):
                            print(
                                "Granger causal interval of:",
                                "%s->%s" % (x_i, y_i),
                                "Failed!",
                            )
                        # No intervals found. Maybe loop_granger has a bug or there does not exist an valid interval.
                        ordered_intervals = []
                    else:
                        matrics = [array_results_YX, array_results_XY]
                        ordered_intervals = self.runner.get_ordered_intervals(
                            matrics, local_length
                        )
                    local_results["%s->%s" % (x_i, y_i)][
                        "intervals"
                    ] = ordered_intervals
                    # result_YX is the Granger test results of: Y <-- X (target <-- feature).
                    # result_XY is the Granger test results of: X <-- Y (feature <-- target).
                    local_results["%s->%s" % (x_i, y_i)]["result_YX"] = array_results_YX
                    local_results["%s->%s" % (x_i, y_i)]["result_XY"] = array_results_XY
                    i = i + 1
            os.remove(param_dict["data"])
            safe_dump_obj(local_results, local_results_file_path)
        self.local_results = local_results
        if self.runtime_debug:
            toc = time.time()
            time_stat_dict["granger_analysis"] = toc - tic
            self.time_stat_dict = time_stat_dict
        return

    def generate_DCC(self, T, N):
        """Generate dynamic causality curve (DCC) between two services by overlaying intervals.
        Args:
            T: Data length in time dimension.
            N: Num of time series.
        """
        try:
            self.local_results
        except AttributeError:
            print(
                "No granger causal intervals available. Please run granger_analyze first."
            )
            return
        self.DCC = defaultdict(int)
        # verbose: >=2 print dynamic causality curve info
        if self.if_verbose(2):
            print("{:<10}{}".format("", "Generating dynamic causality curves..."))
        for x_i in range(N):
            for y_i in range(N):
                if y_i == x_i:
                    continue
                key = f"{x_i}->{y_i}"
                intervals = self.local_results[key]["intervals"]
                overlay_counts = get_overlay_count(T, intervals)
                self.DCC[key] = overlay_counts

