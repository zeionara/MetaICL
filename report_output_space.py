import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()


    config_path = Path(args.config)
    if not config_path.exists() or not config_path.is_file():
        print(f"File does not exist: {config_path}")
        sys.exit()

    # print(config_path.stem)
    with open(config_path) as f:
        cfg = json.load(f)
        datasets = cfg[args.split]

        out_obj = {}
        for idx, train_item in enumerate(datasets):
            if not train_item.endswith('.jsonl'):
                if args.split == 'train':
                    train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
                elif args.split == 'test':
                    train_item = Path("data") / train_item / f"{train_item}_16_13_test.jsonl"

            with open(train_item) as f:
                line = f.readline()
                obj = json.loads(line)
                out_obj[obj['task']] = {
                    'options': obj['options']
                }
    print(json.dumps(out_obj, indent=4, sort_keys=True))