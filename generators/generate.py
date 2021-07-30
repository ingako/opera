#!/usr/bin/env python3

import os
from skmultiflow.data.random_tree_generator import RandomTreeGenerator

num_sample = 10000
seed = 42
delim = ','
output_filename = f'../data/tree/{seed}.arff'

n_cat_features = 5
n_categories_per_cat_feature=2
n_classes = 2
header = []

for i in range(n_cat_features * n_categories_per_cat_feature):
    header.append(f"@attribute a{i} {{0,1}}")

header.append(f"@attribute class {{0,1}}")
header.append("@data\n")


# Setting up the stream
stream = RandomTreeGenerator(
            tree_random_state=seed,
            n_classes=n_classes,
            n_cat_features=n_cat_features,
            n_num_features=0,
            n_categories_per_cat_feature=n_categories_per_cat_feature,
            max_tree_depth=5,
            min_leaf_depth=3,
            fraction_leaves_per_level=0.15)

print(f'generating {output_filename}...')

with open(output_filename, 'w') as out:
    out.write('\n'.join(header))
    for _ in range(num_sample):
        X, y = stream.next_sample()

        out.write(delim.join(str(int(v)) for v in X[0]))
        out.write(f'{delim}{int(y[0])}')
        out.write('\n')
