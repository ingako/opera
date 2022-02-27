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


stable_period=50000
delim = ','
prune_level=3
num_branches_to_prune_list=[1, 2, 3]
num_branches_to_prune_list=[1]
max_tree_depth = 8
min_leaf_depth = max_tree_depth
# subtree_max_tree_depths = [6, 12, 18]
subtree_max_tree_depths = [3]
n_cat_features = 20
n_categories_per_cat_feature= 2
n_classes = 3


for seed in range(10):
    for num_branches_to_prune in num_branches_to_prune_list:
        for subtree_max_tree_depth in subtree_max_tree_depths:

            header = []

            for i in range(n_cat_features * n_categories_per_cat_feature):
                header.append(f"@attribute a{i} {{0,1}}")

            class_str = ",".join([str(v) for v in range(n_classes)])
            header.append(f"@attribute class {{{class_str}}}")
            header.append("@data\n")

            output_dir = \
                f'../data/tree/{subtree_max_tree_depth}/{num_branches_to_prune}'
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            stable_period_logger = setup_logger(f'drift-{seed}', f'{output_dir}/drift-{seed}.log')
            output_filename = f'{output_dir}/{seed}.arff'

            full_tree_stream = RandomTreeGenerator(
                        tree_random_state=seed,
                        sample_random_state=seed,
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
            pruned_tree_stream.max_tree_depth = subtree_max_tree_depth
            pruned_tree_stream.min_leaf_depth = subtree_max_tree_depth

            pruned_tree_stream.prune_subtrees(prune_level=prune_level,
                                              num_branches_to_prune=num_branches_to_prune)

            print("after prunning")
            pruned_tree_stream.get_depth_info()


            # stream = RecurrentDriftStream(stable_period=stable_period,
            #                               position=stable_period,
            #                               stable_period_logger=stable_period_logger)
            # stream.prepare_for_use([full_tree_stream, pruned_tree_stream])

            with open(output_filename, 'w') as out:
                out.write('\n'.join(header))
                for _ in range(stable_period):
                    X, y = full_tree_stream.next_sample()
                    out.write(delim.join(str(int(v)) for v in X[0]))
                    out.write(f'{delim}{int(y[0])}')
                    out.write('\n')
                for _ in range(stable_period):
                    X, y = pruned_tree_stream.next_sample()
                    out.write(delim.join(str(int(v)) for v in X[0]))
                    out.write(f'{delim}{int(y[0])}')
                    out.write('\n')
