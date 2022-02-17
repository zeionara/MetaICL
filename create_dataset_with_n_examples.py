import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--num_examples", type=int, required=True)
    parser.add_argument("--verbose", '-v', type=str, default=False)

    args = parser.parse_args()
    num_examples = args.num_examples

    config_path = Path(args.config)
    if not config_path.exists() or not config_path.is_file():
        print(f"File does not exist: {config_path}")
        sys.exit()

    np.random.seed(0)
    with open(config_path) as f:
        cfg = json.load(f)

        print(config_path.stem)
        
        # Point directly to the jsonl files
        train_truncated_files = []
        output_dir = Path("data") / f"{config_path.stem}_k{num_examples}"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=False)
        for idx, train_item in enumerate(cfg['train']):
            train_truncated = output_dir / f"{train_item}_{num_examples}_100_train.jsonl"
            print(train_truncated)

            if not train_item.endswith('.jsonl'):
                train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
            train_truncated_files.append(str(train_truncated))
            
            with open(train_item) as f_in:
                lines = f_in.readlines()
                with open(train_truncated, 'w') as f_out:
                    # for line in itertools.islice(f_in, num_examples):
                    #     f_out.write(line)
                    for line in np.random.choice(lines, num_examples, replace=False):
                        f_out.write(line)
        
        out_cfg_path = Path('config') / f"{config_path.stem}_k{num_examples}.json"
        with open(out_cfg_path, 'w') as f_out:
            cfg['train'] = train_truncated_files
            json.dump(cfg, f_out, indent=4, sort_keys=True)
        print(f"Written cfg file to {out_cfg_path}")