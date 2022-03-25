import argparse
import json
import numpy as np
import pandas as pd
import shutil
import spacy
import tarfile
import time

from collections import Counter
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.width', 1000)

nlp = spacy.load("en_core_web_sm")
is_valid_pos = {
    "ADJ": True, # adjective
    "ADP": True, # adposition
    "ADV": True, # adverb
    "AUX": True, # auxiliary
    "CCONJ": True, # coordinating conjunction
    "DET": True, # determiner
    "INTJ": True, # interjection
    "NOUN": True, # noun
    "NUM": False, # numeral
    "PART": True, # particle
    "PRON": True, # pronoun
    "PROPN": False, # proper noun
    "PUNCT": False, # punctuation
    "SCONJ": True, # subordinating conjunction
    "SYM": False, # symbol
    "VERB": True, # verb
    "X": False, # other
    "SPACE": False,
}

def measure_proseness(text):
    if len(text) == 0:
        return 0
    doc = nlp(text)
    total_count = 0
    valid_count = 0
    for token in doc:
        if is_valid_pos[token.pos_]:
            valid_count += 1
        total_count += 1
    assert total_count == len(doc)
    return valid_count / len(doc)

def convert_to_df(table, header_row_idx=None):
    assert all([len(row) == len(table[0]) for row in table])
    df = pd.DataFrame(table)
    if header_row_idx is not None:
        header = df.iloc[header_row_idx] #grab the first row for the header
        new_header = []
        for idx, name in enumerate(header):
            # Replace any empty column names with the column index
            if not name:
                name = f"col_{idx}"
            if name in new_header:
                name = f"{name}_{idx}"
            new_header.append(name)
        df = df[header_row_idx + 1:] # take the data less the header row
        df.columns = new_header # set the header row as the df header
    df = df.drop_duplicates()
    return df

def is_mostly_valid_text(df, min_proseness=0.7):
    enumerated_sample_rows = islice(df.iterrows(), 1, 4) # Skip the first row (probably header), take 3 rows
    score = np.mean([measure_proseness(' '.join(row)) for idx, row in enumerated_sample_rows])
    if score >= min_proseness:
        return True
    else:
        return False

def find_diverse_text_columns(df, min_proseness=0.7):
    diverse_cols = []
    for col_name in df.columns:
        if len(df[col_name].unique()) >= 0.9 * len(df[col_name]): # We want diverse columns
            if measure_proseness(' '.join(df[col_name])) >= min_proseness: # We only want text columns
                diverse_cols.append({
                    'column': col_name,
                    'labels': [], # No labels for generative output
                })
    if len(diverse_cols) < 2: # We need at least two good columns to form an input->output mapping
        return []
    return diverse_cols

def measure_class_balance(counter: Counter):
    """
    Shannon entropy-based measure of class balance:
    https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
    """
    n = np.sum([c for c in counter.values()])
    k = len(counter)
    numerator = -np.sum([(c / n) * np.log(c / n) for c in counter.values()])
    return numerator / np.log(k)

def find_categorical_columns(df):
    cat_cols = []
    for col in df.columns:
        col_values = df[col].to_list()
        col_values = [val for val in col_values if val.strip()]# Remove empty strings
        class_labels = list(set(col_values))

        if not len(class_labels) >= 2:
            continue

        labels_fewer_than_rows = len(class_labels) < 0.7 * len(col_values)
        if not labels_fewer_than_rows:
            continue

        label_counter = Counter(col_values)
        classes_are_balanced = measure_class_balance(label_counter) < 0.8
        if not classes_are_balanced:
            continue

        labels_are_english = measure_proseness(' '.join(class_labels)) > 0.8
        if not labels_are_english:
            continue

        cat_cols.append({
            'column': col,
            'labels': class_labels,
        })

    return cat_cols


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tarfile", type=str, required=True)
    parser.add_argument("--min_rows", type=int, default=10)
    
    args = parser.parse_args()

    export_slice = Path(args.tarfile).stem
    out_dir = Path(args.tarfile).parent / export_slice
    out_dir.mkdir(parents=True, exist_ok=False)

    longlistfile = out_dir / "longlist.jsonl" # Starting point for human annotators to keep track of high quality tables
    peekfile = out_dir / "peek.txt" # For human annotators to peek at table contents during annotation
    assert not longlistfile.exists()
    assert not peekfile.exists()

    assert tarfile.is_tarfile(args.tarfile)
    print("Starting extraction...")
    with tarfile.open(args.tarfile, "r") as file:
        hits = 0
        t_start = time.time()
        for idx, member in tqdm(enumerate(file)):
            if idx % 1000 == 0:
                print(f"Progress: {hits}/{idx} hits after {time.time() - t_start}s")

            f = file.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf-8")
            obj = json.loads(content)

            # We only want relational tables
            if (obj['tableType'] != 'RELATION'):
                continue

            # Load table data
            table = obj['relation']

            # Transpose table
            if obj['tableOrientation'] == 'HORIZONTAL':
                table = list(map(list, zip(*table)))
            
            # Convert to dataframe
            header_row_idx = obj['headerRowIndex'] if obj['hasHeader'] else None
            df = convert_to_df(table, header_row_idx=header_row_idx)

            if df.shape[0] < args.min_rows:
                continue

            if not is_mostly_valid_text(df):
                continue
            
            diverse_cols = find_diverse_text_columns(df)
            cat_cols = find_categorical_columns(df)
            if not (diverse_cols or cat_cols):
                continue

            title = f"{obj['pageTitle']} || {obj['title']}"
            # for row in table[:10]:
            #   print(row)
            # print("cat_cols", cat_cols)
            # print("diverse_cols", diverse_cols)
            # display(df)
            # break
            
            # Save table
            for output_col in diverse_cols + cat_cols:
                save_obj = {
                    "title": title,
                    "output_column": output_col['column'],
                    "labels": output_col['labels'],
                    "file": member.name,
                }
                with open(longlistfile, 'a') as f:
                    json.dump(save_obj, f)
                    f.write('\n')
            
            with open(peekfile, 'a') as f:
                print(title, file=f)
                print(member.name, file=f)
                print(df, file=f)
                # for col in diverse_cols + cat_cols:
                #     print(col, file=f)
                print("\n", file=f)

            hits += 1