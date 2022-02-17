# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from utils.data import load_data

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="SST-2")

    args = parser.parse_args()

    for cfg_file in Path("./config").glob("*.json"):
        try:
            train_data = load_data(cfg_file.stem, "train", 16384, seed=100)
        except FileNotFoundError as e:
            continue
        print(f"{cfg_file.stem:40} {len(train_data)} examples")