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
from pathlib import Path

def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, max_examples_per_task=None, shuffle_examples=True, shuffle_examples_seed=0):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", task+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    # Support paths to directories containing jsonl files
    datasets_expanded = []
    for dataset in datasets:
        if Path(dataset).is_dir():
            for p in Path(dataset).glob("*.jsonl"):
                datasets_expanded.append(str(p))
        else:
            datasets_expanded.append(dataset)

    data = []
    for task_idx, dataset in enumerate(datasets_expanded):
        if dataset.endswith('.jsonl'):
            data_path = dataset
        else:
            data_path = os.path.join("data", dataset,
                                    "{}_{}_{}_{}.jsonl".format(dataset, k, seed if split=="train" else 100,
                                                            "test" if split is None else split))
        with open(data_path, "r") as f:
            if max_examples_per_task is None:
                lines = f.readlines()
            else:
                lines = f.readlines()[:max_examples_per_task]
            if shuffle_examples:
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(lines)
            for line in lines:
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
    return data
