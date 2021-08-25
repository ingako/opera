#!/usr/bin/env python3

import copy
import os
import logging
from pathlib import Path

from skmultiflow.data.random_tree_generator import RandomTreeGenerator
from stream_generator import RecurrentDriftStream

formatter = logging.Formatter('%(message)s')

def setup_logger(name, log_file, level=logging.INFO):
  handler = logging.FileHandler(log_file, mode='w')
  handler.setFormatter(formatter)

  logger = logging.getLogger(name)
  logger.setLevel(level)
  logger.addHandler(handler)

  return logger


num_sample = 40000
seed = 42
delim = ','

max_tree_depths=[10]
# min_leaf_depths=[1, 10, 20]
min_leaf_depths=[10]
n_cat_features = 20
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
        stable_period_logger = setup_logger(f'drift-{seed}', f'{output_dir}/drift-{seed}.log')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_filename = f'{output_dir}/{seed}.arff'

        full_tree_stream = RandomTreeGenerator(
                    tree_random_state=seed,
                    n_classes=n_classes,
                    n_cat_features=n_cat_features,
                    n_num_features=0,
                    n_categories_per_cat_feature=n_categories_per_cat_feature,
                    max_tree_depth=max_tree_depth,
                    min_leaf_depth=min_leaf_depth,
                    fraction_leaves_per_level=0.15)

        print(f'generating {output_filename}...')
        full_tree_stream.get_depth_info()

        pruned_tree_stream = copy.deepcopy(full_tree_stream)
        pruned_tree_stream.prune_subtrees(prune_level=3, prune_percentage=0.5)

        print("after prunning")
        pruned_tree_stream.get_depth_info()


        stream = RecurrentDriftStream(stable_period_logger=stable_period_logger)
        stream.prepare_for_use([pruned_tree_stream, full_tree_stream])

        with open(output_filename, 'w') as out:
            out.write('\n'.join(header))
            for _ in range(num_sample):
                X, y = stream.next_sample()

                out.write(delim.join(str(int(v)) for v in X[0]))
                out.write(f'{delim}{int(y[0])}')
                out.write('\n')
