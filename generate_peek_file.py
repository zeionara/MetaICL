import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
from utils.data import load_data


# def write_samples(list_of_paths, write_samples_to, max_tasks=None, max_examples_per_task=None):
#     # Check if there are repeated tasks:
#     if len(list_of_paths) != len(set(list_of_paths)):
#         num_repeats = len(list_of_paths) - len(set(list_of_paths))
#         print(f"WARNING: There are {num_repeats}/{len(list_of_paths)} repeated tasks in the training set.")

#     print('\nWriting to file...')

#     with open(write_samples_to, 'w') as f_out:
#         if max_tasks is not None:
#             list_of_paths = list_of_paths[:max_tasks]
#         for jsonl_path in tqdm(list_of_paths):
#             with open(jsonl_path) as f:
#                 lines = f.readlines()
#                 if max_examples_per_task is not None:
#                     lines = lines[:max_examples_per_task]
#                 if len(lines) == 0:
#                     continue
#                 print(f"{json.loads(lines[0])['task']} ({len(lines)} examples)", file=f_out)
#                 for line in lines:
#                     obj = json.loads(line)
#                     print("---", file=f_out)
#                     print(f"input: {obj['input']}", file=f_out)
#                     print(f"output: {obj['output']}", file=f_out)
#             print("\n==================\n", file=f_out)
#     print("Wrote examples printout to", write_samples_to)

def write_json_samples(list_of_paths, write_samples_to, max_tasks=None, max_examples_per_task=None):
    # Check if there are repeated tasks:
    if len(list_of_paths) != len(set(list_of_paths)):
        num_repeats = len(list_of_paths) - len(set(list_of_paths))
        print(f"WARNING: There are {num_repeats}/{len(list_of_paths)} repeated tasks in the training set.")

    print('\nWriting to file...')

    out_obj = {}
    if max_tasks is not None:
        list_of_paths = list_of_paths[:max_tasks]
    for jsonl_path in tqdm(list_of_paths):
        task_name = jsonl_path.parent.stem
        jsonl_path = str(jsonl_path)
        out_obj[task_name] = []
        with open(jsonl_path) as f:
            lines = f.readlines()
            if max_examples_per_task is not None:
                lines = lines[:max_examples_per_task]
            if len(lines) == 0:
                continue
            # print(f"{json.loads(lines[0])['task']} ({len(lines)} examples)", file=f_out)
            for line in lines:
                obj = json.loads(line)
                out_obj[task_name].append(obj)
                # print(line, file=f_out)
                # print(f"input: {obj['input']}", file=f_out)
                # print(f"output: {obj['output']}", file=f_out)
    #         print("\n==================\n", file=f_out)
    with open(write_samples_to, 'w') as f_out:
        json.dump(out_obj, f_out, indent=4, sort_keys=True)
    print("Wrote examples printout to", write_samples_to)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./debug/')
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--max_examples_per_task", type=int, default=200)
    parser.add_argument("--split", type=str, default='train')

    # parser.add_argument("--k", type=int, default=16384)
    # parser.add_argument("--seed", type=int, default=100)
    # parser.add_argument("--max_examples_per_task", type=int, default=None)
    # parser.add_argument("--shuffle", type=int, default=1)
    # parser.add_argument("--shuffle_examples_seed", type=int, default=0)
    # parser.add_argument("--is_cluster_dataset", type=int, default=0)
    # parser.add_argument("--cluster_idxs", type=str, default=None)
    # parser.add_argument("--max_tasks_per_cluster", type=int, default=None)

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    assert out_dir.is_dir(), out_dir
    outfile = out_dir / f"{Path(args.config).stem}.json"

    # train_data = load_data(args.task, "train", args.k, seed=args.seed,
    #     max_examples_per_task=args.max_examples_per_task,
    #     shuffle_examples=args.shuffle,
    #     shuffle_examples_seed=args.shuffle_examples_seed,
    #     is_cluster_dataset=args.is_cluster_dataset,
    #     max_tasks_per_cluster=args.max_tasks_per_cluster,
    #     cluster_idxs=[int(idx) for idx in str(args.cluster_idxs).split(',')] if args.cluster_idxs else None,
    #     )
    # print(f"Loaded {len(train_data)} datapoints.")

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

        datasets = cfg[args.split]

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
                if args.split == 'train':
                    train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
                elif args.split == 'test':
                    train_item = Path("data") / train_item / f"{train_item}_16_13_test.jsonl"
            train_files.append(train_item)

        print("Inspecting")
        # write_samples(train_files, outfile, max_tasks=args.max_tasks, max_examples_per_task=args.max_examples_per_task)
        write_json_samples(train_files, outfile, max_tasks=args.max_tasks, max_examples_per_task=args.max_examples_per_task)