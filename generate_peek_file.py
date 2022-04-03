import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
    

def write_samples(list_of_paths, write_samples_to, num_peek=None):
    # Check if there are repeated tasks:
    if len(list_of_paths) != len(set(list_of_paths)):
        num_repeats = len(list_of_paths) - len(set(list_of_paths))
        print(f"WARNING: There are {num_repeats}/{len(list_of_paths)} repeated tasks in the training set.")

    print('\nWriting to file...')

    with open(write_samples_to, 'w') as f_out:
        if num_peek is not None:
            list_of_paths = list_of_paths[:num_peek]
        for jsonl_path in tqdm(list_of_paths):
            with open(jsonl_path) as f:
                lines = f.readlines()
                if len(lines) == 0:
                    continue
                print(f"{json.loads(lines[0])['task']} ({len(lines)} examples)", file=f_out)
                for line in lines[:3]:
                    obj = json.loads(line)
                    print("---", file=f_out)
                    print(f"input: {obj['input']}", file=f_out)
                    print(f"output: {obj['output']}", file=f_out)
            print("\n==================\n", file=f_out)
    print("Wrote examples printout to", write_samples_to)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./debug/')
    parser.add_argument("--num_peek", type=str, default=None)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    assert out_dir.is_dir(), out_dir
    outfile = out_dir / f"{Path(args.config).stem}.txt"

    config_path = Path(args.config)
    if not config_path.exists() or not config_path.is_file():
        print(f"File does not exist: {config_path}")
        sys.exit()

    with open(config_path) as f:
        cfg = json.load(f)

        print(config_path.stem)
        
        # Point directly to the jsonl files
        num_examples = 50
        train_files = []

        datasets = cfg['train']

        # Support paths to directories containing jsonl files
        datasets_expanded = []
        for dataset in datasets:
            if Path(dataset).is_dir():
                for p in Path(dataset).glob("*.jsonl"):
                    datasets_expanded.append(str(p))
            else:
                datasets_expanded.append(dataset)
        
        for idx, train_item in enumerate(datasets_expanded):
            if not train_item.endswith('.jsonl'):
                train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
            train_files.append(train_item)

        print("Inspecting")
        write_samples(train_files, outfile, num_peek=args.num_peek)