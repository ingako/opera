#!/usr/bin/env python3

import copy
import os
import logging
import sys
from pathlib import Path

from stream_generator import RecurrentDriftStream
import pandas as pd

def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all()

# flip_digits = [0, 1]
# flip_digits = [0, 1, 2, 3, 4]
flip_digits = [0, 1, 2, 3, 4, 5, 6, 7]

data_name = sys.argv[1] # mnist, fashion-mnist

df = pd.read_csv(f"../data/{data_name}/{data_name}.csv", header=0)

# print(df.head())
n_row, n_col = df.shape
print(f"row={n_row}, col={n_col}")

df_values = df.values

# max pooling
# 28*28 784
max_pool_values = []
for r in range(0, len(df_values)):
    max_pool_row = [df_values[r][0]] # label
    for i in range(1, n_col):
        if i % (28*2) != 1:
            continue

        for j in range(i, i + 28, 2):
            max_pool_row.append(max(df_values[r][j], df_values[r][j+1], df_values[r][j+28], df_values[r][j+29]))
    max_pool_values.append(max_pool_row)
n_col = len(max_pool_values[0])
col_names = [v for v in range(n_col)]
col_names[0] = "label"
df = pd.DataFrame(max_pool_values, columns=col_names)


# remove useless attributes
df_new = pd.DataFrame(df.iloc[:,0])
count = 1
for i in range(1, n_col):
    if not is_unique(df.iloc[:,i]):
        df_new.insert(count, str(i), df.iloc[:,i])
        count += 1

## 
df = df_new
df_orig = df.copy()
df_orig = df_orig.sample(frac=1, random_state=0) # shuffle

n_row, n_col = df.shape
print(f"row={n_row}, col={n_col}")


# flip
rows_flip = df.label.isin(flip_digits).astype(bool)
# print(f"row={rows_flip}")
for col_name in list(df):
    if col_name == "label": continue
    df.loc[rows_flip, col_name] = df.loc[rows_flip, col_name].mul(-1).add(255)

df = df.sample(frac=1, random_state=1) # shuffle

# concat
df = df_orig.append(df, ignore_index=True)
# df = df.append(df_orig, ignore_index=True)

# append label column to the end
label_col = df.iloc[:,0].copy()
df = df.drop("label", 1)
df.insert(n_col-1, "label", label_col)

output_dir = f'../data/{data_name}-max-pooling/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
output_filename = f'{output_dir}/flip{"".join([str(d) for d in flip_digits])}.arff'
df.to_csv(output_filename, sep=',', header=False, index=False)


# insert arff header
header = []
# attributes = [str(j) for j in range(n_col-1)]
# attributes_str = ",".join(attributes)
for i in range(n_col-1):
    header.append(f"@attribute a{i} numeric")
    # header.append(f"@attribute a{i} {{{attributes_str}}}")

header.append(f"@attribute class {{0,1,2,3,4,5,6,7,8,9}}")
header.append("@data\n")

with open(output_filename, "r") as f:
    data = f.readlines()

data.insert(0, "\n".join(header))

with open(output_filename, "w") as f:
    data = "".join(data)
    f.write(data)
