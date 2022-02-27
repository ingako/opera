#!/usr/bin/env python

import copy
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["backend"] = "Qt4Agg"
plt.rcParams["figure.figsize"] = (20, 10)

dataset = "./hoeffding-tree-result.csv"
df = pd.read_csv(dataset, index_col=None, sep=',')
T = df["learning evaluation instances"]
power = df["classifications correct (percent)"]

xnew = np.linspace(T.min(), T.max(), 25)
spl = make_interp_spline(T, power, k=3)  # type=BSpline
power_smooth = spl(xnew)

# +1 due to the fact that diff reduces the original index number
locs = (np.diff(np.sign(np.diff(power_smooth))) > 0).nonzero()[0] + 1         # local min

# for v in locs:
#     print(xnew[v])
print(xnew[locs])

# plot
# plt.figure(figsize=(12, 5))
# plt.plot(xnew, power_smooth, color='grey')
# plt.plot(xnew[locs], power_smooth[locs], "o", label="min", color='r')
# plt.show()

header_filename = "./covtype-header.arff"
with open(header_filename, "r") as f:
    header = f.readlines()

data_filename = "./covtype-data.arff"
data = []
with open(data_filename, "r") as f:
    data = f.readlines()

print("chunk sizes")
prev = 0
chunks = []
for loc in xnew[locs]:
    loc = int(loc)
    chunks.append(data[prev:loc])
    # print(f"{prev}:{loc}")
    print(loc-prev)
    prev = loc

# for i in range(len(chunks)-1):
#     for j in range(i+1, len(chunks)):
#         chunk = header.copy()
#         chunk.extend(chunks[i])
#         chunk.extend(chunks[j])
# 
#         drift_loc = len(chunks[i])
#         output_filename = f"covtype-{i}-{j}-{drift_loc}.arff"
#         print(output_filename)
#         with open(output_filename, "w") as f:
#             f.write("".join(chunk))
