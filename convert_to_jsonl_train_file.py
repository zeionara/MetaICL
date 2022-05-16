import argparse
import json
import sys
from pathlib import Path
from utils.data import load_anydata


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
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)

    # Train args
    parser.add_argument("--do_tensorize", default=True, action="store_true")
    parser.add_argument("--tensorize_dir", type=str, default="tensorized")
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--n_process", type=int, default=4)
    parser.add_argument("--max_length_per_example", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=256)

    parser.add_argument("--use_demonstrations", default=True, action="store_true")
    parser.add_argument("--log_file", default=None, type=str)
    parser.add_argument("--debug_data_order", type=int, default=0)
    parser.add_argument("--repeat_batch", type=int, default=1)

    parser.add_argument("--num_training_steps", type=int, default=1000000)
    parser.add_argument("--validation_split", type=float, default=0.001)
    parser.add_argument("--save_period", type=int, default=10000)
    parser.add_argument("--log_period", type=int, default=2000)

    parser.add_argument("--train_algo", type=str, default=None)
    # parser.add_argument("--task", type=str, default="SST-2")
    parser.add_argument("--k", type=int, default=16384)
    parser.add_argument("--test_k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--train_seed", type=int, default=1)

    parser.add_argument("--max_examples_per_task", type=int, default=None)
    parser.add_argument("--shuffle", type=int, default=1)
    parser.add_argument("--shuffle_examples_seed", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument("--init_checkpoint", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--no_masking", default=False, action="store_true")

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--method", type=str, default="direct", choices=["direct", "channel"])
    parser.add_argument("--gpt2", type=str, default="gpt2-large")

    parser.add_argument("--optimization", type=str, default="8bit-adam")
    parser.add_argument("--fp16", default=True, action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")

    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--disable_wandb", default=False, action="store_true")
    parser.add_argument("--verbose_train", type=int, default=0)

    parser.add_argument("--max_examples_per_test", type=int, default=100)
    parser.add_argument("--test_tasks", type=str, default=None) # 'all_tasks_test', 
    parser.add_argument("--is_cluster_dataset", type=int, default=0)
    parser.add_argument("--cluster_idxs", type=str, default=None)
    parser.add_argument("--max_tasks_per_cluster", type=int, default=None)

    parser.add_argument("--use_random_label", type=int, default=0)
    parser.add_argument("--predict_last_word", type=int, default=0)
    parser.add_argument("--swap_input_output", type=int, default=0)

    parser.add_argument("--target_num_examples", type=int, default=8000)
    parser.add_argument("--task_ratios", type=str, default=None)

    args = parser.parse_args()

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path("./export_jsonl/") / args.task
    outdir.mkdir(parents=True, exist_ok=True)

    train_data = load_anydata(args)
    print(f"Num examples in the final set: {len(train_data)}")
    # Train data is a flat list of [json_obj, json_obj, json_obj, ...] where each json_obj is an example from relevant train.jsonl files
    all_tasks = sorted(list(set([dp["task"] for dp in train_data])))
    num_tasks = len(all_tasks)
    print(f"Num tasks in the final set: {num_tasks}")
    
    for idx, task in enumerate(all_tasks):
        outfile = outdir / f"{idx:03}.jsonl"
        print(outfile)
        task_text = ''
        with open(outfile, 'w') as f:
            for dp in train_data:
                if task != dp['task']:
                    continue
                out_obj = {
                    'task': dp['task'],
                    'input': dp['input'],
                    'output': dp['output'],
                    'options': dp['options'],
                }
                print(json.dumps(out_obj), file=f) # Dump with newline
    print("Exported to", outfile)

    # train_counter = Counter()
    # for dp in train_data:
    #     train_counter[dp["task"]] += 1
    # if args.local_rank <= 0:
    #     for k, v in train_counter.items():
    #         logger.info("[Train] %s\t%d" % (k, v))
    #     logger.info("%s on %s (%d train)" % (args.method, args.task, len(train_counter)))


    # config_path = Path(args.config)
    # if not config_path.exists() or not config_path.is_file():
    #     print(f"File does not exist: {config_path}")
    #     sys.exit()

    # with open(config_path) as f:
    #     cfg = json.load(f)

    #     print(config_path.stem)
        
    #     # Point directly to the jsonl files
    #     num_examples = 50
    #     train_files = []

    #     datasets = cfg[args.split]

    #     # Support paths to directories containing jsonl files
    #     datasets_expanded = []
    #     for dataset in datasets:
    #         if Path(dataset).is_dir():
    #             for p in Path(dataset).glob("*.jsonl"):
    #                 datasets_expanded.append(str(p))
    #         else:
    #             datasets_expanded.append(dataset)
        
    #     for idx, train_item in enumerate(datasets_expanded):
    #         if not train_item.endswith('.jsonl'):
    #             if args.split == 'train':
    #                 train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
    #             elif args.split == 'test':
    #                 train_item = Path("data") / train_item / f"{train_item}_16_13_test.jsonl"
    #         train_files.append(train_item)

    #     print("Inspecting")
    #     # write_samples(train_files, outfile, max_tasks=args.max_tasks, max_examples_per_task=args.max_examples_per_task)
    #     write_json_samples(train_files, outfile, max_tasks=args.max_tasks, max_examples_per_task=args.max_examples_per_task)