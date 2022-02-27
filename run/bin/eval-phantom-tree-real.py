#!/usr/bin/env python3

import os
import sys
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pprint import PrettyPrinter

from scipy.stats import friedmanchisquare

benchmark_list = ["base",
                  "enable-patch",
                  "disable-patch"]

@dataclass
class Param:
    dataset : str = ""
    exp_code : str = ""
    base_learner: str = "rf"
    noise: int = -1
    perf_window: int = -1
    num_phantom_branches: int = 5
    drift_loc: int = 50
    min_obs: str = ""
    conv_rate: str = ""

covtype_0_1= \
    Param(
        dataset="covtype",
        exp_code="covtype-0-1-73501",
        perf_window="5000",
        num_phantom_branches="30",
        min_obs="2000",
        conv_rate="0.15"
    )

covtype_1_2= \
    Param(
        dataset="covtype",
        exp_code="covtype-1-2-48334",
        perf_window="5000",
        num_phantom_branches="30",
        min_obs="2000",
        conv_rate="0.15"
    )

covtype_2_3= \
    Param(
        dataset="covtype",
        exp_code="covtype-2-3-120836",
        perf_window="5000",
        num_phantom_branches="30",
        min_obs="2000",
        conv_rate="0.15"
    )

covtype_3_4= \
    Param(
        dataset="covtype",
        exp_code="covtype-3-4-72502",
        perf_window="5000",
        num_phantom_branches="30",
        min_obs="2000",
        conv_rate="0.15"
    )

params = [covtype_0_1, covtype_1_2, covtype_2_3, covtype_3_4]


def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

for p in params:
    print(f"=================={p.exp_code}==================")

    apropo_region = []
    err_region = []
    full_region = []

    apropo_region_time = []
    err_region_time = []
    full_region_time = []

    for i in range(len(benchmark_list)):
        benchmark = benchmark_list[i]
        path = f"{p.dataset}/{benchmark_list[i]}/{p.base_learner}/{p.exp_code}"

        if benchmark == "enable-patch":
            path = f"{path}/" \
                   f"{p.perf_window}/" \
                   f"{p.num_phantom_branches}/" \
                   f"{p.min_obs}/" \
                   f"{p.conv_rate}"

        acc_list = []
        kappa_list = []
        acc_gain_list = []
        kappa_gain_list = []
        acc_gain_per_drift_list = []
        time_list = []

        switch_pos = []

        for seed in range(1):
            disable_output = f"{p.dataset}/base/{p.base_learner}/{p.exp_code}/result.csv"
            base_rf = pd.read_csv(disable_output)

            result_path = f"{path}/result.csv"

            benchmark_df = pd.read_csv(result_path)

            metrics = []

            acc_list.extend(benchmark_df["classifications correct (percent)"].to_list())
            kappa_list.extend(benchmark_df["Kappa Statistic (percent)"].to_list())

            runtime = 0
            if benchmark == "disable-patch" and benchmark == "transfer-only":
                runtime = \
                    (benchmark_df["evaluation time (cpu seconds)"].max() \
                    - benchmark_df["evaluation time (cpu seconds)"].iloc[p.drift_loc]) \
                    + (apropo_region_time[seed]+err_region_time[seed]+full_region_time[seed])/(1e+9)
                # nano seconds to seconds

            else:
                runtime = \
                    (benchmark_df["evaluation time (cpu seconds)"].max() \
                    - benchmark_df["evaluation time (cpu seconds)"].iloc[p.drift_loc])/60



            if benchmark == "enable-patch":
                apropo_region.append(benchmark_df["apropos region depth avg"].max())
                err_region.append(benchmark_df["error region depth avg"].max())
                full_region.append(benchmark_df["full region depth avg"].max())

                apropo_region_time.append(benchmark_df["apropos region time"].max())
                err_region_time.append(benchmark_df["error region time"].max())
                full_region_time.append(benchmark_df["full region time"].max())


                # switch_pos.append(benchmark_df["switchToNewClassifierPos"].max())
                switch_pos.append(benchmark_df.iloc[:,-1:].max())

            time_list.append(runtime)

            # benchmark_df["switchToNewClassifierPos"]

            if benchmark == "base":
                acc_gain_list.append(0)
                kappa_gain_list.append(0)
                acc_gain_per_drift_list.append(0)
            else:

                acc_gain = 0
                for i in range(p.drift_loc, len(benchmark_df)):
                    acc_gain += \
                        benchmark_df["classifications correct (percent)"][i] \
                        - base_rf["classifications correct (percent)"][i]
                acc_gain_list.append(acc_gain)

                acc_gain = 0
                for i in range(p.drift_loc, p.drift_loc+10):
                    acc_gain += \
                        benchmark_df["classifications correct (percent)"][i] \
                        - base_rf["classifications correct (percent)"][i]
                acc_gain_per_drift_list.append(acc_gain/10)

                # kappa_gain_list.append( \
                #     benchmark_df["Kappa Statistic (percent)"].sum() \
                #     - base_rf["Kappa Statistic (percent)"].sum())

        # acc = np.mean(acc_list[p.drift_loc:])
        # acc_std = np.std(acc_list[p.drift_loc:])
        # metrics.append(f"${acc:.2f}" + " \\pm " + f"{acc_std:.2f}$")

        # kappa = np.mean(kappa_list[p.drift_loc:])
        # kappa_std = np.std(kappa_list[p.drift_loc:])
        # metrics.append(f"${kappa:.2f}" + " \\pm " + f"{kappa_std:.2f}$")

        if benchmark == "base":
            metrics.append('-')
            metrics.append('-')
        else:
            # acc_gain_per_drift = np.mean(acc_gain_per_drift_list)
            # acc_gain_per_drift_std = np.std(acc_gain_per_drift_list)
            # metrics.append(f"${acc_gain_per_drift:.2f}" + " \\pm " + f"{acc_gain_per_drift_std:.2f}$")

            acc_gain = np.mean(acc_gain_list)
            acc_gain_std = np.std(acc_gain_list)
            metrics.append(f"${round(acc_gain)}" + " \\pm " + f"{round(acc_gain_std)}$")

            # kappa_gain = np.mean(kappa_gain_list)
            # kappa_gain_std = np.std(kappa_gain_list)
            # metrics.append(f"${round(kappa_gain)}" + " \\pm " + f"{round(kappa_gain_std)}$")


        time = np.mean(time_list)
        time_std = np.std(time_list)
        metrics.append(f"${time:.2f}" + " \\pm " + f"{time_std:.2f}$")

        print(" & ".join(metrics) + " \\\\")

    print(f"{benchmark}")
    print("depth")
    if len(apropo_region) != 0:
        # print(sum(apropo_region))
        # print(sum(err_region))
        # print(sum(full_region))

        # print(sum(apropo_region) / 10)
        # print(sum(err_region) / 10)
        # print(sum(full_region) / 10)

        print(np.mean(apropo_region), end="+")
        print(np.std(apropo_region))
        print(np.mean(err_region), end="+")
        print(np.std(err_region))
        print(np.mean(full_region), end="+")
        print(np.std(full_region))


    # print("time")
    # if len(apropo_region_time) != 0:
    #     # print((sum(apropo_region_time) + sum(err_region_time) + sum(full_region_time))/6e+10)

    #     print(sum(apropo_region_time)/10)
    #     print(sum(err_region_time)/10)
    #     print(sum(full_region_time)/10)
