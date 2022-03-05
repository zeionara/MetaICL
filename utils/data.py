# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import csv
import json
import string
import numpy as np
import torch

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, max_examples_per_task=None, shuffle_examples=True, shuffle_examples_seed=0):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = []
    for dataset in datasets:
        if dataset.endswith('.jsonl'):
            data_path = dataset
        else:
            data_path = os.path.join("data", dataset,
                                    "{}_{}_{}_{}.jsonl".format(dataset, k, seed if split=="train" else 100,
                                                            "test" if split is None else split))
        with open(data_path, "r") as f:
            lines = f.readlines()
            if shuffle_examples:
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(lines)
            for idx, line in enumerate(lines):
                if max_examples_per_task is not None:
                    if idx >= max_examples_per_task:
                        break
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
    return data

