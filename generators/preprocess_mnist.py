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
n_row, n_col = df.shape
print(n_row)
print(n_col)

print(df.head())

for i in range(1, n_col):
    df.iloc[:,i] = df.iloc[:,i].mul(-1).add(255)
print()
print(df.head())


header = []
for i in range(n_col-1):
    header.append(f"@attribute a{i} numeric")

header.append(f"@attribute class {{0,1,2,3,4,5,6,7,8,9}}")
header.append("@data\n")

output_dir = f'../data/mnist/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_filename = f'{output_dir}/mnist-flip{flip_percentage}.arff'
df.to_csv(output_filename, sep=',')
