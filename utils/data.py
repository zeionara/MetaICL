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
from datasets import load_dataset


def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False):
    if task.startswith('huggingface:'):
        hf_dataset_path = task[len('huggingface:'):]
        data = []
        dataset = load_dataset(hf_dataset_path)
        for row in dataset['train']:
            data.append(row)
        return data
    else:
        if config_split is None:
            config_split = split

        if datasets is None:
            with open(os.path.join("config", task+".json"), "r") as f:
                config = json.load(f)
            datasets = config[config_split]

        data = []
        for dataset in datasets:
            data_path = os.path.join("data", dataset,
                                    "{}_{}_{}_{}.jsonl".format(dataset, k, seed, split))

            print('data path = ', data_path)

            with open(data_path, "r") as f:
                for line in f:
                    dp = json.loads(line)
                    if is_null:
                        dp["input"] = "N/A"
                    data.append(dp)
        return data

