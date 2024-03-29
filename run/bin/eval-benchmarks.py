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
                  "aotradaboost",
                  "ecpf",
                  "enable-patch",
                  "disable-patch"]


@dataclass
class Param:
    dataset : str = ""
    exp_code : str = ""
    base_learner: str = "rf"
    noise: int = -1
    perf_window: str = ""
    num_phantom_branches: int = 5
    drift_loc: int = 50
    min_obs: str = ""
    conv_rate: str = ""

tree_6_1 = \
    Param(
        dataset="tree",
        exp_code='6/1',
        # perf_window=500,
        num_phantom_branches="",
        perf_window="",
    )
tree_12_1 = \
    Param(
        dataset="tree",
        exp_code='12/1',
        num_phantom_branches="", # 5
        # perf_window=500,
        perf_window="",
    )

# tree_6_1 = \
#     Param(
#         dataset="tree",
#         exp_code='6/1',
#         perf_window="500",
#         num_phantom_branches="5",
#     )
# tree_12_1 = \
#     Param(
#         dataset="tree",
#         exp_code='12/1',
#         perf_window="500",
#         num_phantom_branches="5",
#     )

tree_6_3 = \
    Param(
        dataset="tree",
        exp_code='6/3',
        perf_window="1000",
        num_phantom_branches="5"
    )

tree_4842 = \
    Param(
        dataset="tree",
        exp_code='4/8/4/2',
        perf_window="1000",
        num_phantom_branches="10",
        min_obs="2000",
        conv_rate="0.15"
    )

tree_4844 = \
    Param(
        dataset="tree",
        exp_code='4/8/4/4',
        perf_window="1000",
        num_phantom_branches="10",
        min_obs="2000",
        conv_rate="0.15"
    )

tree_41242 = \
    Param(
        dataset="tree",
        exp_code='4/12/4/2',
        perf_window="1000",
        num_phantom_branches="10",
        min_obs="2000",
        conv_rate="0.15"
    )
tree_41244 = \
    Param(
        dataset="tree",
        exp_code='4/12/4/4',
        perf_window="1000",
        num_phantom_branches="10",
        min_obs="2000",
        conv_rate="0.15"
    )
# "4/8/4/2" "4/8/4/4" "4/12/4/2" "4/12/4/4"

tree_41244_9 = \
    Param(
        dataset="tree",
        exp_code='4/12/4/4/9',
        perf_window="1000",
        num_phantom_branches="10",
        min_obs="2000",
        conv_rate="0.15"
    )

#############################################
params=[]
for num_branches_to_prune in [1, 2, 3]:
    for subtree_max_tree_depth in [3, 6, 12]:
        params.append(
            Param(
                dataset="tree",
                exp_code=f'6/{subtree_max_tree_depth}/3/{num_branches_to_prune}/20',
                perf_window="1000",
                num_phantom_branches="30",
                min_obs="2000",
                conv_rate="0.15"
            )
        )

params=[]
n_classes=1024
for num_branches_to_prune in [8, 16, 24, 28]:
# for num_branches_to_prune in [28]:
    # for subtree_max_tree_depth in [9, 15]:
    # for subtree_max_tree_depth in [20]:
    for subtree_max_tree_depth in [6, 10, 20]:
        params.append(
            Param(
                dataset="tree",
                exp_code=f'6/{subtree_max_tree_depth}/6/{num_branches_to_prune}/{n_classes}',
                perf_window="1000",
                num_phantom_branches="30",
                min_obs="2000",
                conv_rate="0.15"
            )
        )

params=[]
n_classes=20
for num_branches_to_prune in [1, 2, 3]:
    for subtree_max_tree_depth in [3, 6, 10, 20]:
        params.append(
            Param(
                dataset="tree",
                exp_code=f'5/{subtree_max_tree_depth}/3/{num_branches_to_prune}/{n_classes}',
                perf_window="1000",
                num_phantom_branches="30",
                min_obs="2000",
                conv_rate="0.15"
            )
        )

params=[]
n_classes=20
for num_branches_to_prune in [1, 3, 5, 7]:
    for subtree_max_tree_depth in [4, 7, 10, 13]:
        exp_code_str=f'5/{subtree_max_tree_depth}/4/{num_branches_to_prune}/{n_classes}'
        if exp_code_str == "5/4/4/5/20" or exp_code_str=="5/10/4/7/20":
            continue
        if exp_code_str == '5/3/3/3/20' or exp_code_str=="5/4/4/7/20":
            continue
        params.append(
            Param(
                dataset="tree",
                exp_code=f'5/{subtree_max_tree_depth}/4/{num_branches_to_prune}/{n_classes}',
                perf_window="1000",
                num_phantom_branches="5",
                min_obs="2000",
                conv_rate="0.15"
            )
        )
##############################################

mnist_20 = \
    Param(
        dataset="mnist-max-pooling",
        exp_code='flip01',
        perf_window="10000",
        num_phantom_branches="5",
        drift_loc=70
    )
mnist_50 = \
    Param(
        dataset="mnist-max-pooling",
        exp_code='flip01234',
        perf_window="5000",
        num_phantom_branches="5",
        drift_loc=70
    )
mnist_80 = \
    Param(
        dataset="mnist-max-pooling",
        exp_code='flip01234567',
        perf_window="10000",
        num_phantom_branches="5",
        drift_loc=70
    )

##############################################

fashion_20 = \
    Param(
        dataset="fashion-mnist-max-pooling",
        exp_code='flip01',
        perf_window="100",
        num_phantom_branches="5",
        drift_loc=70
    )
fashion_50 = \
    Param(
        dataset="fashion-mnist-max-pooling",
        exp_code='flip01234',
        perf_window="5000",
        num_phantom_branches="5",
        drift_loc=70
    )
fashion_80 = \
    Param(
        dataset="fashion-mnist-max-pooling",
        exp_code='flip01234567',
        perf_window="10000",
        num_phantom_branches="20",
        drift_loc=70
    )

# params = [tree_6_1, tree_12_1, tree_6_3]
# params = [tree_41244_9]
# "4/8/4/2" "4/8/4/4" "4/12/4/2" "4/12/4/4"
# params = [mnist_20, mnist_50, mnist_80]
# params = [fashion_20, fashion_50, fashion_80]


def is_empty_file(fpath):
    return False if os.path.isfile(fpath) and os.path.getsize(fpath) > 0 else True

for p in params:
    print(f"{p.dataset}")
    print(f"{p.exp_code}")

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

        for seed in range(10):
        # for seed in range(1):
            disable_output = f"{p.dataset}/base/{p.base_learner}/{p.exp_code}/{seed}.csv"
            base_rf = pd.read_csv(disable_output)

            result_path = f"{path}/{seed}.csv"

            benchmark_df = pd.read_csv(result_path)

            metrics = []

            acc_list.extend(benchmark_df["classifications correct (percent)"].to_list())
            # kappa_list.extend(benchmark_df["Kappa Statistic (percent)"].astype(np.float).to_list())
            # count = 0
            # print(benchmark_df["Kappa Statistic (percent)"].to_list())
            # for v in benchmark_df["Kappa Statistic (percent)"].to_list():
            #     if v == '?':
            #         count+=1
            # print(f"count={count}")
            # benchmark_df["Kappa Statistic (percent)"] = pd.to_numeric(benchmark_df["Kappa Statistic (percent)"], errors='coerce').fillna(0)
            kappa_list.extend(benchmark_df["Kappa Statistic (percent)"].to_list())

            runtime = 0
            if benchmark == "disable-patch" or benchmark=="transfer-only":
                runtime = \
                    (benchmark_df["evaluation time (cpu seconds)"].max() \
                    - benchmark_df["evaluation time (cpu seconds)"].iloc[p.drift_loc]) \
                    + (apropo_region_time[seed]+err_region_time[seed]+full_region_time[seed])/(1e+9)
                runtime /= 60
            else:
                # print(len(benchmark_df["evaluation time (cpu seconds)"]))
                runtime = \
                    (benchmark_df["evaluation time (cpu seconds)"].max() \
                    - benchmark_df["evaluation time (cpu seconds)"].iloc[p.drift_loc])/60
            if benchmark == "enable-patch":
                apropo_region_time.append(benchmark_df["apropos region time"].max())
                err_region_time.append(benchmark_df["error region time"].max())
                full_region_time.append(benchmark_df["full region time"].max())

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

                kappa_gain_list.append( \
                    benchmark_df["Kappa Statistic (percent)"].sum() \
                    - base_rf["Kappa Statistic (percent)"].sum())

        acc = np.mean(acc_list[p.drift_loc:])
        acc_std = np.std(acc_list[p.drift_loc:])
        metrics.append(f"${acc:.2f}" + " \\pm " + f"{acc_std:.2f}$")

        kappa = np.mean(kappa_list[p.drift_loc:])
        kappa_std = np.std(kappa_list[p.drift_loc:])
        metrics.append(f"${kappa:.2f}" + " \\pm " + f"{kappa_std:.2f}$")

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

            kappa_gain = np.mean(kappa_gain_list)
            kappa_gain_std = np.std(kappa_gain_list)
            metrics.append(f"${round(kappa_gain)}" + " \\pm " + f"{round(kappa_gain_std)}$")


        time = np.mean(time_list)
        time_std = np.std(time_list)
        metrics.append(f"${time:.2f}" + " \\pm " + f"{time_std:.2f}$")

        print(" & ".join(metrics) + " \\\\")
