#!/usr/bin/env python3

import os
from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from pathlib import Path

num_sample = 100000
seed = 42
delim = ','

max_tree_depths=[3]
# min_leaf_depths=[1, 10, 20]
min_leaf_depths=[3]
n_cat_features = 10
n_categories_per_cat_feature= 2
n_classes = 2


for max_tree_depth in max_tree_depths:
    for min_leaf_depth in min_leaf_depths:

        header = []

        for i in range(n_cat_features * n_categories_per_cat_feature):
            header.append(f"@attribute a{i} {{0,1}}")

        header.append(f"@attribute class {{0,1}}")
        header.append("@data\n")

        output_dir = f'../data/tree/{n_cat_features}/{max_tree_depth}/{min_leaf_depth}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_filename = f'{output_dir}/{seed}.arff'

        stream = RandomTreeGenerator(
                    tree_random_state=seed,
                    n_classes=n_classes,
                    n_cat_features=n_cat_features,
                    n_num_features=0,
                    n_categories_per_cat_feature=n_categories_per_cat_feature,
                    max_tree_depth=max_tree_depth,
                    min_leaf_depth=min_leaf_depth,
                    fraction_leaves_per_level=0.15)

        print(f'generating {output_filename}...')
        stream.get_depth_info()

        with open(output_filename, 'w') as out:
            out.write('\n'.join(header))
            for _ in range(num_sample):
                X, y = stream.next_sample()

                out.write(delim.join(str(int(v)) for v in X[0]))
                out.write(f'{delim}{int(y[0])}')
                out.write('\n')
