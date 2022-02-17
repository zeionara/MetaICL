import argparse
import numpy as np
import json
import spacy
from collections import Counter
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def count_examples(jsonl_path):
    with open(jsonl_path) as f:
        lines = f.readlines()
    return sum([1 for line in lines])

def uppercase_word_density(string):
    words = [word for word in string.split() if word[0].isalpha()]
    if len(words) <= 0:
        return 1
    return sum([1 for word in words if word[0].isupper()]) / len(words)

def count_alpha_words(string):
    return sum([1 for word in string.split() if word[0].isalpha()])

def extract_input_output(json_string):
    obj = json.loads(json_string)
    return obj['input'] + ' ' + obj['output']

def get_avg_example_length(jsonl_path):
    with open(jsonl_path) as f:
        lines = f.readlines()
    if len(lines) == 0:
        return 0
    return np.mean([len(extract_input_output(line)) for line in lines])

def get_parts_of_speech_bag(text):
    doc = nlp(text)
    pos_dict = Counter([token.pos_ for token in doc])
    return pos_dict

def inspect_dataset(list_of_paths, num_peek=10, write_samples_to=None, verbose=False):
    if verbose:
        print("# tasks:", len(list_of_paths))
        num_examples_for_tasks = [count_examples(path) for path in list_of_paths]
        print("# empty tasks:", sum([1 for count in num_examples_for_tasks if count == 0]))
        print("avg # examples per task:", np.mean(num_examples_for_tasks))
        print("avg length of examples:", np.mean([get_avg_example_length(path) for path in list_of_paths]))
        # Check if there are repeated tasks:
        if len(list_of_paths) != len(set(list_of_paths)):
            num_repeats = len(list_of_paths) - len(set(list_of_paths))
            print(f"WARNING: There are {num_repeats}/{len(list_of_paths)} repeated tasks in the training set.")

    print('\nWriting to file...')

    if write_samples_to is not None:
        with open(write_samples_to, 'w') as f_out:
            np.random.seed(0)
            np.random.shuffle(list_of_paths)
            for jsonl_path in tqdm(list_of_paths[:num_peek]):
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
                        text = extract_input_output(line)
                        pos_dict = get_parts_of_speech_bag(text)
                        print(pos_dict, file=f_out)
                    mean_pos_types = np.mean([len(get_parts_of_speech_bag(extract_input_output(text))) for text in lines[:5]])
                    print("\nmean_pos_types", mean_pos_types, file=f_out)
                print("\n\n==================\n\n", file=f_out)
        print("Wrote examples printout to", write_samples_to)

import argparse
import sys
from pathlib import Path

def filter_paths(input_paths):
    output_paths = []
    for path in tqdm(input_paths):
        with open(path) as f:
            lines = f.readlines()

            if len(lines) == 0:
                continue

            text_lines = [extract_input_output(line) for line in lines]

            # line_count = sum(1 for line in lines)
            # if line_count <= 10:
            #     continue
            
            # median_alpha_words = np.median([count_alpha_words(line) for line in text_lines])
            # if median_alpha_words <= 5:
            #     continue

            # uppercase_density = np.mean([uppercase_word_density(line) for line in text_lines])
            # if uppercase_density < 0.5:
            #     continue

            mean_pos_types = np.mean([len(get_parts_of_speech_bag(text)) for text in text_lines[:5]])
            if mean_pos_types < 10:
                continue
        
        output_paths.append(path)
    print(f"Clean paths: ({len(output_paths)} / {len(input_paths)})")
    return output_paths

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--write_samples_to", type=str, default='samples.txt')
    parser.add_argument("--verbose", '-v', type=str, default=False)

    args = parser.parse_args()

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
        for idx, train_item in enumerate(cfg['train']):

            if not train_item.endswith('.jsonl'):
                train_item = Path("data") / train_item / f"{train_item}_16384_100_train.jsonl"
            train_files.append(train_item)
            
        print("Filtering")
        train_files = filter_paths(train_files)
        print("Inspecting")
        inspect_dataset(train_files, num_peek=100, write_samples_to=args.write_samples_to, verbose=args.verbose)