#!/usr/bin/env python3

import copy
import os
import logging
from pathlib import Path

from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from stream_generator import RecurrentDriftStream
import pandas as pd

flip_percentage = 25

df = pd.read_csv("../data/mnist/mnist.csv", header=0)

print(df.head())
n_row, n_col = df.shape
print(f"row={n_row}, col={n_col}")

# flip
for i in range(1, int(flip_percentage/100 * n_col)):
    df.iloc[:,i] = df.iloc[:,i].mul(-1).add(255)

# shuffle
df = df.sample(frac = 1)

# append label column to the end
label_col = df.iloc[:,0].copy()
df = df.drop("label", 1)
df.insert(n_col-1, "label", label_col)

print(df.head())
n_row, n_col = df.shape
print(f"row={n_row}, col={n_col}")

output_dir = f'../data/mnist/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_filename = f'{output_dir}/flip{flip_percentage}.arff'
df.to_csv(output_filename, sep=',', header=False)

# insert arff header
header = []
for i in range(n_col):
    header.append(f"@attribute a{i} numeric")

header.append(f"@attribute class {{0,1,2,3,4,5,6,7,8,9}}")
header.append("@data\n")

with open(output_filename, "r") as f:
    data = f.readlines()

data.insert(0, "\n".join(header))

with open(output_filename, "w") as f:
    data = "".join(data)
    f.write(data)
