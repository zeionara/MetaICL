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

def load_data_from_clusters(task, max_tasks_per_cluster=None, max_examples_per_task=None, shuffle_examples=True, shuffle_examples_seed=0, cluster_idxs=None):
    jsonl_files = []
    with open(Path("config") / f"{task}.json") as f:
        obj = json.load(f)
        for cluster_name, cluster_file_list in obj.items():
            if cluster_idxs: # Whitelist to only train on particular cluster idxs
                # cluster_name is something like "cluster50_idx11_strain_ATCC_S_c_Escherichia"
                idxpart = [part for part in cluster_name.split('_') if part.startswith('idx')]
                cluster_idx = int(idxpart[0][len('idx'):])
                if cluster_idx not in cluster_idxs:
                    continue

            if shuffle_examples:
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(cluster_file_list)
            if max_tasks_per_cluster:
                cluster_file_list = cluster_file_list[:max_tasks_per_cluster]
            jsonl_files += cluster_file_list

    data = []
    for task_idx, jsonl_file in enumerate(jsonl_files):
        assert jsonl_file.endswith('.jsonl')
        assert Path(jsonl_file).exists()

        with open(jsonl_file) as f:
            lines = f.readlines()

            if shuffle_examples:
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(lines)
            if max_examples_per_task:
                lines = lines[:max_examples_per_task]

            for line in lines:
                dp = json.loads(line)
                data.append(dp)
    return data


def load_data(task, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, max_examples_per_task=None, shuffle_examples=True, shuffle_examples_seed=0, 
              is_cluster_dataset=0, max_tasks_per_cluster=None, cluster_idxs=None):
    if is_cluster_dataset:
        if split != 'train':
            raise NotImplementedError('Cluster dataset only supported for training.')
        return load_data_from_clusters(
            task, 
            max_tasks_per_cluster=max_tasks_per_cluster, 
            max_examples_per_task=max_examples_per_task, 
            shuffle_examples=shuffle_examples, 
            shuffle_examples_seed=shuffle_examples_seed,
            cluster_idxs=cluster_idxs)

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
            lines = f.readlines()

            if shuffle_examples:
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(lines)
            if max_examples_per_task:
                lines = lines[:max_examples_per_task]

            for line in lines:
                dp = json.loads(line)
                if split == 'test' and len(dp['options']) < 2: # Only multiple-choice tasks accepted for test evaluation
                    raise NotImplementedError(f"{dataset} is missing options for test evaluation; only multi-choice questions are supported for test evaluation!")
                if is_null:
                    dp["input"] = "N/A"
                data.append(dp)
    return data

def load_data_by_task(config_name, split, k, seed=0, config_split=None, datasets=None,
              is_null=False, max_examples_per_task=None, shuffle_examples=True, shuffle_examples_seed=0):
    if config_split is None:
        config_split = split

    if datasets is None:
        with open(os.path.join("config", config_name+".json"), "r") as f:
            config = json.load(f)
        datasets = config[config_split]

    data = {}
    for dataset in datasets:
        data[dataset] = []
        if dataset.endswith('.jsonl'):
            data_path = dataset
        else:
            data_path = os.path.join("data", dataset,
                                    "{}_{}_{}_{}.jsonl".format(dataset, k, seed if split=="train" else 100,
                                                            "test" if split is None else split))
        with open(data_path, "r") as f:
            lines = f.readlines()
            if shuffle_examples:
                print("Shuffling examples with seed", shuffle_examples_seed)
                np.random.seed(shuffle_examples_seed)
                np.random.shuffle(lines)
            for idx, line in enumerate(lines):
                if max_examples_per_task is not None:
                    if idx >= max_examples_per_task:
                        break
                dp = json.loads(line)
                if is_null:
                    dp["input"] = "N/A"
                data[dataset].append(dp)
    return data

