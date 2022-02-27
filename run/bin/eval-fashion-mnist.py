#!/usr/bin/env python

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)


base_learner = "rf"
# base_learner = "ozaboost"

# dataset_name = "mnist-max-pooling"
dataset_name = "fashion-mnist-max-pooling"

# dataset_prefix = "flip01"
# # perf_window = "10000"
# # num_phantom_branches = "20"
# perf_window = "100"
# num_phantom_branches = "10"

dataset_prefix = "flip01234"
perf_window = "5000"
num_phantom_branches = "10"

# dataset_prefix = "flip01234567"
# perf_window = "10000"
# num_phantom_branches = "10"




seed=2

result_str = f"{dataset_prefix}/{seed}"

dataset = f"{dataset_name}/ecpf/{base_learner}/{result_str}.csv"
ecpf = pd.read_csv(dataset, index_col=0, sep=',')

dataset = f"{dataset_name}/aotradaboost/{base_learner}/{result_str}.csv"
aotradaboost= pd.read_csv(dataset, index_col=0, sep=',')

# dataset = f"{dataset_name}/enable-patch/{base_learner}/{result_str}.csv"
# dataset = f"{dataset_name}/enable-patch/{base_learner}/{dataset_prefix}/5000/{seed}.csv"
#  dataset = f"{dataset_name}/enable-patch/{base_learner}/{dataset_prefix}/100/{seed}.csv"
dataset = f"{dataset_name}/enable-patch/{base_learner}/{dataset_prefix}/{perf_window}/{num_phantom_branches}/{seed}.csv"
enable_patching = pd.read_csv(dataset, index_col=0, sep=',')

dataset = f"{dataset_name}/disable-patch/{base_learner}/{result_str}.csv"
disable_patching = pd.read_csv(dataset, index_col=0, sep=',')

dataset = f"{dataset_name}/base/{base_learner}/{result_str}.csv"
base = pd.read_csv(dataset, index_col=0, sep=',')

plt.plot(ecpf["classifications correct (percent)"], label="ECPF")
plt.plot(aotradaboost["classifications correct (percent)"], label="AOTrAdaBoost")
plt.plot(enable_patching["classifications correct (percent)"], label="OPERA (patching)")
plt.plot(disable_patching["classifications correct (percent)"], label="OPERA (new model)", linestyle="--")
plt.plot(base["classifications correct (percent)"], label="base", linestyle="--")

print("runtime")
print((ecpf["evaluation time (cpu seconds)"].max()
       - ecpf["evaluation time (cpu seconds)"].iloc[70]) / 60)
print((aotradaboost["evaluation time (cpu seconds)"].max()
       - aotradaboost["evaluation time (cpu seconds)"].iloc[70]) / 60)
print((enable_patching["evaluation time (cpu seconds)"].max()
       - enable_patching["evaluation time (cpu seconds)"].iloc[70]) / 60)
print((disable_patching["evaluation time (cpu seconds)"].max()
       - disable_patching["evaluation time (cpu seconds)"].iloc[70]) / 60)


print("correct:")
print(enable_patching["apropos region depth avg"].max())
print("error:")
print(enable_patching["error region depth avg"].max())
print("full:")
print(enable_patching["full region depth avg"].max())

obs_period = str(enable_patching["instance store size"].max())
print(f"obs period:{obs_period}")


plt.legend()
plt.xlabel("no. of instances")
plt.ylabel("accuracy")

plt.show()

# plt.savefig('tree-results.png', bbox_inches='tight', dpi=100)
# plt.savefig(f'/home/hwu344/img/{base_learner}-{result_str}.png', bbox_inches='tight', dpi=100)
